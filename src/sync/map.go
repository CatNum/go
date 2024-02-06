// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"sync/atomic"
)

// Map is like a Go map[interface{}]interface{} but is safe for concurrent use
// by multiple goroutines without additional locking or coordination.
// Loads, stores, and deletes run in amortized constant time.
//
// The Map type is specialized. Most code should use a plain Go map instead,
// with separate locking or coordination, for better type safety and to make it
// easier to maintain other invariants along with the map content.
//
// The Map type is optimized for two common use cases: (1) when the entry for a given
// key is only ever written once but read many times, as in caches that only grow,
// or (2) when multiple goroutines read, write, and overwrite entries for disjoint
// sets of keys. In these two cases, use of a Map may significantly reduce lock
// contention compared to a Go map paired with a separate Mutex or RWMutex.
//
// The zero Map is empty and ready for use. A Map must not be copied after first use.
//
// In the terminology of the Go memory model, Map arranges that a write operation
// “synchronizes before” any read operation that observes the effect of the write, where
// read and write operations are defined as follows.
// Load, LoadAndDelete, LoadOrStore, Swap, CompareAndSwap, and CompareAndDelete
// are read operations; Delete, LoadAndDelete, Store, and Swap are write operations;
// LoadOrStore is a write operation when it returns loaded set to false;
// CompareAndSwap is a write operation when it returns swapped set to true;
// and CompareAndDelete is a write operation when it returns deleted set to true.
type Map struct {
	mu Mutex

	// read contains the portion of the map's contents that are safe for
	// concurrent access (with or without mu held).
	//
	// The read field itself is always safe to load, but must only be stored with
	// mu held.
	//
	// Entries stored in read may be updated concurrently without mu, but updating
	// a previously-expunged entry requires that the entry be copied to the dirty
	// map and unexpunged with mu held.
	// read 包含 map 中可以并发读取的部分
	// read 字段本身可以安全的读取，但是在存储时必须加锁
	// 存储在 read 中的元素可以在无锁的情况下同时修改，但是修改一个之前删除的元素需要先将元素赋值到 dirty map
	// 并在持有锁的情况下进行解除清除（unexpunged）
	read atomic.Pointer[readOnly]

	// explain：这里的 clean map 指的应该是 read map，因为对应的是 dirty map
	// dirty contains the portion of the map's contents that require mu to be
	// held. To ensure that the dirty map can be promoted to the read map quickly,
	// it also includes all of the non-expunged entries in the read map.
	//
	// Expunged entries are not stored in the dirty map. An expunged entry in the
	// clean map must be unexpunged and added to the dirty map before a new value
	// can be stored to it.
	//
	// If the dirty map is nil, the next write to the map will initialize it by
	// making a shallow copy of the clean map, omitting stale entries.
	// dirty 包含了需要持有 mu 锁的 map 内容部分。为了确保 dirty map 可以快速晋升为
	// read map，它还包括了 read map 中所有未被清除的条目。
	dirty map[any]*entry

	// misses counts the number of loads since the read map was last updated that
	// needed to lock mu to determine whether the key was present.
	//
	// Once enough misses have occurred to cover the cost of copying the dirty
	// map, the dirty map will be promoted to the read map (in the unamended
	// state) and the next store to the map will make a new dirty copy.

	// 存储自read最新一次更新之后需要去加锁判断key是否存在的次数
	// 一旦这个数值超过某个阈值，dirty会升级为read（处于未修订状态），而且下一次存储操作会创建一个新的dirty副本
	misses int
}

// readOnly is an immutable struct stored atomically in the Map.read field.
// readOnly 是以原子方式存储在 Map.read 字段中的不可变结构。
type readOnly struct {
	m map[any]*entry
	// 如果 dirty 中有 read 中没有的 key，则为true
	// 换句话说，如果 read 不完整，返回 true；完整，返回 false
	amended bool // true if the dirty map contains some key not in m.
}

// expunged is an arbitrary pointer that marks entries which have been deleted
// from the dirty map.
// Expunged 是一个任意指针，用于标记从 dirty map 已删除的条目。
var expunged = new(any)

// An entry is a slot in the map corresponding to a particular key.
type entry struct {
	// p points to the interface{} value stored for the entry.
	//
	// If p == nil, the entry has been deleted, and either m.dirty == nil or
	// m.dirty[key] is e.
	//
	// If p == expunged, the entry has been deleted, m.dirty != nil, and the entry
	// is missing from m.dirty.
	//
	// Otherwise, the entry is valid and recorded in m.read.m[key] and, if m.dirty
	// != nil, in m.dirty[key].
	//
	// An entry can be deleted by atomic replacement with nil: when m.dirty is
	// next created, it will atomically replace nil with expunged and leave
	// m.dirty[key] unset.
	//
	// An entry's associated value can be updated by atomic replacement, provided
	// p != expunged. If p == expunged, an entry's associated value can be updated
	// only after first setting m.dirty[key] = e so that lookups using the dirty
	// map find the entry.

	// p 是指向实际存储 value 值的地方的指针
	// 如果 p == nil，那么该 entry 已被删除，并且 m.dirty == nil 或者 m.dirty[key] 是 e。
	//
	// 如果 p == expunged，则该 entry 已被删除，m.dirty != nil，并且 entry 在 m.dirty 中缺失。
	//
	// 否则，该 entry 有效并记录在 m.read.m[key] 中，如果 m.dirty != nil，则也记录在 m.dirty[key] 中。
	//
	// 一个 entry 可以通过使用原子替换为 nil 来删除：当 m.dirty 下次被创建时，
	// 它将以原子方式将 nil 替换为 expunged，并保持 m.dirty[key] 未设置。
	//
	// 一个entry的关联值可以通过原子替换进行更新，前提是 p != expunged。
	// 如果 p == expunged，则只有在首先设置 m.dirty[key] = e 后才能更新条目的关联值，
	// 这样使用 dirty map 进行查找时就能找到该条目。
	p atomic.Pointer[any]
}

func newEntry(i any) *entry {
	e := &entry{}
	e.p.Store(&i)
	return e
}

func (m *Map) loadReadOnly() readOnly {
	if p := m.read.Load(); p != nil {
		return *p
	}
	return readOnly{}
}

// Load returns the value stored in the map for a key, or nil if no
// value is present.
// The ok result indicates whether value was found in the map.
// 返回存储的值，如果没有则返回nil
// ok 的结果是该值是否在 map 中被找到
func (m *Map) Load(key any) (value any, ok bool) {
	read := m.loadReadOnly()
	// read优先：先从 read 中取值
	e, ok := read.m[key]
	if !ok && read.amended {
		// 如果不存在且 read 不完整，则加锁访问
		m.mu.Lock()
		// Avoid reporting a spurious miss if m.dirty got promoted while we were
		// blocked on m.mu. (If further loads of the same key will not miss, it's
		// not worth copying the dirty map for this key.)
		// 尽量避免在我们被m.mu阻塞时，如果m.dirty被提升，导致不必要的缺失报告。
		// 如果相同键的后续加载不会缺失，那么为了这个键复制 dirty 是不值得的。
		// 双检查机制：先加锁再访问一遍read
		// 如果read还没有再访问dirty
		read = m.loadReadOnly()
		e, ok = read.m[key]
		if !ok && read.amended {
			e, ok = m.dirty[key]
			// Regardless of whether the entry was present, record a miss: this key
			// will take the slow path until the dirty map is promoted to the read
			// map.
			// 记录misses：无论 entry 是否存在，都记录一次缺失：该键将采取缓慢的路径，直到 dirty 提升到 read。
			m.missLocked()
		}
		m.mu.Unlock()
	}
	if !ok {
		return nil, false
	}
	return e.load()
}

func (e *entry) load() (value any, ok bool) {
	p := e.p.Load()
	if p == nil || p == expunged {
		return nil, false
	}
	return *p, true
}

// Store sets the value for a key.
// 为 key 设置一个 value
func (m *Map) Store(key, value any) {
	_, _ = m.Swap(key, value)
}

// tryCompareAndSwap compare the entry with the given old value and swaps
// it with a new value if the entry is equal to the old value, and the entry
// has not been expunged.
//
// If the entry is expunged, tryCompareAndSwap returns false and leaves
// the entry unchanged.
//
// tryCompareAndSwap 比较给定的旧值与 entry，如果 entry 等于旧值且未被删除，则将其与新值交换。
//
// 如果 entry 已删除，则 tryCompareAndSwap 返回 false 并保持条目不变。
func (e *entry) tryCompareAndSwap(old, new any) bool {
	p := e.p.Load()
	if p == nil || p == expunged || *p != old {
		return false
	}

	// Copy the interface after the first load to make this method more amenable
	// to escape analysis: if the comparison fails from the start, we shouldn't
	// bother heap-allocating an interface value to store.
	// question 不是很理解
	// 在第一次加载后复制 interface，以使此方法更易于逃逸分析：
	// 如果从一开始就失败了比较，我们就不应该费心分配堆上的 interface value 来存储。
	nc := new
	for {
		if e.p.CompareAndSwap(p, &nc) {
			return true
		}
		p = e.p.Load()
		if p == nil || p == expunged || *p != old {
			return false
		}
	}
}

// unexpungeLocked ensures that the entry is not marked as expunged.
//
// If the entry was previously expunged, it must be added to the dirty map
// before m.mu is unlocked.
// unexpungeLocked 确保 entry 未标记为 expunged。
// 将 expunged 状态更换为 nil
// 如果 entry 以前被清除，必须在解锁 m.mu 之前将其添加到 dirty map 中。
func (e *entry) unexpungeLocked() (wasExpunged bool) {
	return e.p.CompareAndSwap(expunged, nil)
}

// swapLocked unconditionally swaps a value into the entry.
//
// The entry must be known not to be expunged.
// swapLocked 无条件的更换一个值到 entry
// 必须知道这个 entry 不是 expunged（被删除的）
func (e *entry) swapLocked(i *any) *any {
	return e.p.Swap(i)
}

// LoadOrStore returns the existing value for the key if present.
// Otherwise, it stores and returns the given value.
// The loaded result is true if the value was loaded, false if stored.
// LoadOrStore 如果键存在，则返回该键的现有值。
// 否则，会存储该值并返回这个值
// 如果获取到了值则 loaded 为 true；存进去了值为 false
func (m *Map) LoadOrStore(key, value any) (actual any, loaded bool) {
	// Avoid locking if it's a clean hit.
	// 如果可以 clean 命中，避免锁定
	// read优先：优先在 read 中进行尝试
	read := m.loadReadOnly()
	if e, ok := read.m[key]; ok {
		// 尝试获取或更新 value
		actual, loaded, ok := e.tryLoadOrStore(value)
		if ok {
			return actual, loaded
		}
	}

	// 如果在 read 中没有获取到值
	// 则加锁
	m.mu.Lock()
	// 双检查机制：防止在锁阻塞的时候，dirty 升级为 read
	// 再次在 read 中尝试获取
	read = m.loadReadOnly()
	if e, ok := read.m[key]; ok {
		// 如果 key 被标记为 expunged，则添加到 dirty 中
		if e.unexpungeLocked() {
			// 这里加入 dirty 的 e 的状态是 nil，而不是 expunged
			m.dirty[key] = e
		}
		// 尝试获取或者存储 value
		actual, loaded, _ = e.tryLoadOrStore(value)
	} else if e, ok := m.dirty[key]; ok {
		// 如果 read 中不存在，则直接获取 dirty 中
		actual, loaded, _ = e.tryLoadOrStore(value)
		// 记录misses
		m.missLocked()
	} else {
		// 如果 dirty 中也没有 且 read 是完整的
		if !read.amended {
			// We're adding the first new key to the dirty map.
			// Make sure it is allocated and mark the read-only map as incomplete.
			// 我们将添加一个新 key 到 dirty 中
			// 确保它被分配（存储）并且标记 read 为不完整
			// 将 read 复制给 dirty
			m.dirtyLocked()
			// 标记 read 为不完整
			m.read.Store(&readOnly{m: read.m, amended: true})
		}
		// 在 dirty 中新增一个 entry
		m.dirty[key] = newEntry(value)
		actual, loaded = value, false
	}
	m.mu.Unlock()

	return actual, loaded
}

// tryLoadOrStore atomically loads or stores a value if the entry is not
// expunged.
//
// If the entry is expunged, tryLoadOrStore leaves the entry unchanged and
// returns with ok==false.
// tryLoadOrStore 原子的获取或存储一个值，如果 entry 不是 expunged 状态
// 如果 entry 是 expunged 状态，tryLoadOrStore 不改变 entry 并返回 ok 为 false
func (e *entry) tryLoadOrStore(i any) (actual any, loaded, ok bool) {
	p := e.p.Load()
	// 先判断是否 expunged 状态（已删除）
	if p == expunged {
		return nil, false, false
	}
	// 非 expunged 状态，则判断是否为空（也是一种已删除）
	// 非 nil，则返回对应的值
	if p != nil {
		return *p, true, true
	}

	// Copy the interface after the first load to make this method more amenable
	// to escape analysis: if we hit the "load" path or the entry is expunged, we
	// shouldn't bother heap-allocating.
	// 在第一次加载后复制接口，以使这种方法更容易进行逃逸分析：
	// 如果我们进入了 “load” 路径或 entry 被删除，就不需要在堆上进行分配。
	ic := i
	for {
		// 这里的 old 传 nil，是因为之前的条件判断现在的值是 nil
		if e.p.CompareAndSwap(nil, &ic) {
			return i, false, true
		}
		p = e.p.Load()
		if p == expunged {
			return nil, false, false
		}
		if p != nil {
			return *p, true, true
		}
	}
}

// LoadAndDelete deletes the value for a key, returning the previous value if any.
// The loaded result reports whether the key was present.
// LoadAndDelete 如果之前存在值，返回之前的值，并删除指定 key 的 value，
// loaded 代表这个 key 之前是否存在值
func (m *Map) LoadAndDelete(key any) (value any, loaded bool) {
	read := m.loadReadOnly()
	// read优先：优先从 read 中读取值
	e, ok := read.m[key]
	if !ok && read.amended {
		// 如果 read 中不存在且 read 不完整，加锁
		m.mu.Lock()
		// 双检查机制：再次读一遍 read
		read = m.loadReadOnly()
		e, ok = read.m[key]
		if !ok && read.amended {
			// 如果 read 中不存在，且 read 不完整
			// 从 dirty 中读
			e, ok = m.dirty[key]
			// 删除 dirty 中的 key
			delete(m.dirty, key)
			// Regardless of whether the entry was present, record a miss: this key
			// will take the slow path until the dirty map is promoted to the read
			// map.
			// 记录 misses
			m.missLocked()
		}
		m.mu.Unlock()
	}
	// 如果 read 存在，获取值并将 read 中的 entry 状态置为 nil（即是删除）
	if ok {
		return e.delete()
	}
	return nil, false
}

// Delete deletes the value for a key.
func (m *Map) Delete(key any) {
	m.LoadAndDelete(key)
}

// 用于将 read 中的 entry 进行删除
func (e *entry) delete() (value any, ok bool) {
	for {
		p := e.p.Load()
		if p == nil || p == expunged {
			return nil, false
		}
		if e.p.CompareAndSwap(p, nil) {
			return *p, true
		}
	}
}

// trySwap swaps a value if the entry has not been expunged.
//
// If the entry is expunged, trySwap returns false and leaves the entry
// unchanged.
// 如果 entry 没有被删除，则更新值
// 如果被删除，则返回 false 且不改变 entry
func (e *entry) trySwap(i *any) (*any, bool) {
	for {
		p := e.p.Load()
		if p == expunged {
			return nil, false
		}
		// 这里是直接更改的
		if e.p.CompareAndSwap(p, i) {
			return p, true
		}
	}
}

// Swap swaps the value for a key and returns the previous value if any.
// The loaded result reports whether the key was present.
// 如果key存在值，swap 更新 key 的值并返回之前的旧值
// loaded 代表 key 之前是否存在值
func (m *Map) Swap(key, value any) (previous any, loaded bool) {
	read := m.loadReadOnly()
	// read优先：先查看read中是否存在key的值，
	if e, ok := read.m[key]; ok {
		// 如果read中存在，尝试更新值
		if v, ok := e.trySwap(&value); ok {
			if v == nil {
				return nil, false
			}
			return *v, true
		}
	}

	// 双检查机制：如果不存在，则加锁再查看一次read中是否存在
	m.mu.Lock()
	read = m.loadReadOnly()
	if e, ok := read.m[key]; ok {
		if e.unexpungeLocked() {
			// The entry was previously expunged, which implies that there is a
			// non-nil dirty map and this entry is not in it.
			// 这个 entry 已经删除（expunged）了，这意味着有一个 non-nil 的 dirty 且 这个 entry 不在 dirty 里面
			m.dirty[key] = e
		}
		// 更新 entry 的值
		if v := e.swapLocked(&value); v != nil {
			loaded = true
			previous = *v
		}
	} else if e, ok := m.dirty[key]; ok {
		// 如果 read 中不存在， dirty 中存在
		// 更新 dirty 中的值
		if v := e.swapLocked(&value); v != nil {
			loaded = true
			previous = *v
		}
	} else {
		// 如果都不存在，判断 read 是否完整
		if !read.amended {
			// 如果 read 完整
			// We're adding the first new key to the dirty map.
			// Make sure it is allocated and mark the read-only map as incomplete.
			// 添加新 key 到 dirty
			// 确保它被分配且标记 read 是不完整的

			// 将 read 中的未删除的 entry 复制到 dirty
			m.dirtyLocked()
			// 更新 read 的状态为不完整
			m.read.Store(&readOnly{m: read.m, amended: true})
		}
		// 创建新 entry 加入到 dirty
		m.dirty[key] = newEntry(value)
	}
	m.mu.Unlock()
	return previous, loaded
}

// CompareAndSwap swaps the old and new values for key
// if the value stored in the map is equal to old.
// The old value must be of a comparable type.
// CompareAndSwap CAS机制
// old value必须是可比较的类型
func (m *Map) CompareAndSwap(key, old, new any) bool {
	read := m.loadReadOnly()
	// read优先：先看 read 中是否存在，存在就更新
	// 如果不存在，就判断 read 是否完整，完整则返回 false
	if e, ok := read.m[key]; ok {
		return e.tryCompareAndSwap(old, new)
	} else if !read.amended {
		return false // No existing value for key.
	}

	// read 不完整 就加锁
	m.mu.Lock()
	defer m.mu.Unlock()
	// 双检查机制：加锁之后再看一遍 read
	read = m.loadReadOnly()
	swapped := false
	if e, ok := read.m[key]; ok {
		// 如果 read 中存在则更新 read 中的值
		swapped = e.tryCompareAndSwap(old, new)
	} else if e, ok := m.dirty[key]; ok {
		// 否则就看 dirty 中是否存在
		// 存在就尝试更新 dirty 中的
		swapped = e.tryCompareAndSwap(old, new)
		// We needed to lock mu in order to load the entry for key,
		// and the operation didn't change the set of keys in the map
		// (so it would be made more efficient by promoting the dirty
		// map to read-only).
		// Count it as a miss so that we will eventually switch to the
		// more efficient steady state.
		// 记录 misses
		m.missLocked()
	}
	return swapped
}

// CompareAndDelete deletes the entry for key if its value is equal to old.
// The old value must be of a comparable type.
//
// If there is no current value for key in the map, CompareAndDelete
// returns false (even if the old value is the nil interface value).
// CompareAndDelete 方法会删除键 key 对应的 entry，如果其值等于 old。
// old 值必须是可比较的类型。
// 如果 map 中不存在键 key 的当前值，CompareAndDelete 会返回 false（即使 old 值是 nil 接口值）。
func (m *Map) CompareAndDelete(key, old any) (deleted bool) {
	read := m.loadReadOnly()
	// read优先：会先从 read 中查找
	e, ok := read.m[key]
	if !ok && read.amended {
		// 如果没找到且 read 是不完整的，加锁
		m.mu.Lock()
		read = m.loadReadOnly()
		// 双检查机制：再在 read 中查一遍
		e, ok = read.m[key]
		if !ok && read.amended {
			// 如果还没找到，且 read 是不完整的
			e, ok = m.dirty[key]
			// Don't delete key from m.dirty: we still need to do the “compare” part
			// of the operation. The entry will eventually be expunged when the
			// dirty map is promoted to the read map.
			//
			// Regardless of whether the entry was present, record a miss: this key
			// will take the slow path until the dirty map is promoted to the read
			// map.
			// 不要从 dirty 中删除：我们仍然需要执行操作中的 “比较” 部分。这个 entry 最终会在 dirty
			// 升级为 read 时被删除。
			// 记录 misses
			m.missLocked()
		}
		m.mu.Unlock()
	}
	// 如果 read 中存在
	for ok {
		p := e.p.Load()
		// 查看是否被删除或者不等于 old
		if p == nil || p == expunged || *p != old {
			return false
		}
		// 比较并置为nil（删除）
		if e.p.CompareAndSwap(p, nil) {
			return true
		}
	}
	return false
}

// Range calls f sequentially for each key and value present in the map.
// If f returns false, range stops the iteration.
//
// Range does not necessarily correspond to any consistent snapshot of the Map's
// contents: no key will be visited more than once, but if the value for any key
// is stored or deleted concurrently (including by f), Range may reflect any
// mapping for that key from any point during the Range call. Range does not
// block other methods on the receiver; even f itself may call any method on m.
//
// Range may be O(N) with the number of elements in the map even if f returns
// false after a constant number of calls.

// Range 依次对 map 中每个键值对调用函数 f。如果 f 返回 false，range 将停止迭代。
//
// Range 不一定对应于 Map 内容的任何一致快照：不会多次访问同一个键，但是如果任何键的值同时被存储或删除（包括在 f 中），
// Range 可能反映 Range 调用期间该键的任何映射。Range 不会阻止接收器上的其他方法；即使 f 本身也可以调用 m 上的任何方法。
//
// Range 的时间复杂度可能为 O(N)，其中 N 是 map 中的元素数量，即使在 f 返回 false 后只经过常数次调用。
func (m *Map) Range(f func(key, value any) bool) {
	// We need to be able to iterate over all of the keys that were already
	// present at the start of the call to Range.
	// If read.amended is false, then read.m satisfies that property without
	// requiring us to hold m.mu for a long time.
	// 我们需要能够迭代 Range 调用开始时已经存在的所有键。
	// 如果 read.amended 为 false，则 read.m 满足该属性，而无需我们长时间持有 m.mu。
	read := m.loadReadOnly()
	// read优先：判断 read 是否不完整
	if read.amended {
		// m.dirty contains keys not in read.m. Fortunately, Range is already O(N)
		// (assuming the caller does not break out early), so a call to Range
		// amortizes an entire copy of the map: we can promote the dirty copy
		// immediately!
		// m.dirty 包含不在 read.m 中的键。幸运的是，Range 已经是 O(N)（假设调用者不会早退出），
		// 因此对 Range 的调用会摊销整个 map 的复制：我们可以立即提升 dirty！
		// 不完整，则加锁
		m.mu.Lock()
		read = m.loadReadOnly()
		// 双检查机制：再次判断 read 是否不完整
		if read.amended {
			// read 不完整
			// 升级 dirty 到 read，将 dirty 复制到 read，并重置 dirty
			read = readOnly{m: m.dirty}
			copyRead := read
			m.read.Store(&copyRead)
			m.dirty = nil
			m.misses = 0
		}
		m.mu.Unlock()
	}

	// 遍历 read
	for k, e := range read.m {
		v, ok := e.load()
		if !ok {
			continue
		}
		if !f(k, v) {
			break
		}
	}
}

// misses++ 并判断 misses 是否达到阈值
func (m *Map) missLocked() {
	m.misses++
	if m.misses < len(m.dirty) {
		return
	}
	// 将 dirty 复制给 read
	m.read.Store(&readOnly{m: m.dirty})
	// dirty 置空
	m.dirty = nil
	m.misses = 0
}

// 将 read 中的 entry 复制到 dirty 中（剔除 read 中为 nil 和 expunged 的）
// 经过这个操作，read 中所有的 nil 都变成了 expunged
func (m *Map) dirtyLocked() {
	if m.dirty != nil {
		return
	}

	read := m.loadReadOnly()
	m.dirty = make(map[any]*entry, len(read.m))
	// 将 read 中的 entry 复制到 dirty 中
	for k, e := range read.m {
		// 剔除 read 中已经被删除的 entry
		if !e.tryExpungeLocked() {
			m.dirty[k] = e
		}
	}
}

// 用来判断 read 中的 entry 是否是 nil 或者 expunged 状态，是就返回 true
// 总的来说，就是在将 read 中的 entry 复制到 dirty 时，剔除被删除的 entry
func (e *entry) tryExpungeLocked() (isExpunged bool) {
	p := e.p.Load()
	for p == nil {
		// 如果为 nil，则进行删除标记
		if e.p.CompareAndSwap(nil, expunged) {
			return true
		}
		p = e.p.Load()
	}
	return p == expunged
}
