"""Real data structures used by AV-ASOC. Each class says where the pipeline uses it."""
from collections import deque, OrderedDict
import heapq, itertools, time
import numpy as np


# ---- Sliding window: audio segmentation ------------------------------------------
def sliding_windows(n, size, hop):
    """Yield (start, end) sample indices of overlapping windows. O(n/hop) time, O(1) space.
    A clip shorter than one window yields a single (0, n) window; a trailing remainder is
    covered by one extra window aligned to the end."""
    if n <= size:
        yield (0, n); return
    last = 0
    for s in range(0, n - size + 1, hop):
        last = s; yield (s, s + size)
    if last + size < n:
        yield (n - size, n)


# ---- Queue / deque: live stream buffer -------------------------------------------
class StreamBuffer:
    """FIFO of incoming PCM chunks. push() is O(1); advance() drops from the left in O(chunks)."""
    def __init__(self):
        self.q, self.count = deque(), 0
    def push(self, chunk):
        self.q.append(chunk); self.count += len(chunk)
    def peek(self, size):
        if self.count < size: return None
        out, need = [], size
        for c in self.q:
            out.append(c[:need]); need -= len(out[-1])
            if need == 0: break
        return np.concatenate(out)
    def advance(self, n):
        while n > 0 and self.q:
            c = self.q[0]
            if len(c) <= n: n -= len(c); self.count -= len(c); self.q.popleft()
            else: self.q[0] = c[n:]; self.count -= n; n = 0


# ---- Priority queue / binary heap: alerts ----------------------------------------
class PriorityAlerts:
    """Min-heap keyed (-priority, -confidence, timestamp): most severe, most confident,
    earliest alert pops first. push/pop O(log n)."""
    def __init__(self):
        self.h, self.c = [], itertools.count()
    def push(self, priority, confidence, timestamp, alert):
        heapq.heappush(self.h, (-priority, -confidence, timestamp, next(self.c), alert))
    def pop(self):
        return heapq.heappop(self.h)[-1]
    def drain(self):
        out = []
        while self.h: out.append(self.pop())
        return out


# ---- Trie: transcript keyword lookup ---------------------------------------------
class Trie:
    """Prefix tree over lowercase words. insert O(L); find_all scans text in O(len(text) * L)."""
    def __init__(self, words=()):
        self.root = {}
        for w in words: self.insert(w)
    def insert(self, w):
        n = self.root
        for ch in w.lower(): n = n.setdefault(ch, {})
        n["$"] = w
    def find_all(self, text):
        """Return (start, end, word) for whole-word matches."""
        t, res = text.lower(), []
        for i in range(len(t)):
            if i and t[i - 1].isalnum(): continue
            n, j = self.root, i
            while j < len(t) and t[j] in n:
                n = n[t[j]]; j += 1
                if "$" in n and (j == len(t) or not t[j].isalnum()):
                    res.append((i, j, n["$"]))
        return res


# ---- String matching (KMP): multi-word phrases -----------------------------------
def kmp_search(text, pat):
    """All start indices of pat in text. O(n + m) time, O(m) space."""
    text, pat = text.lower(), pat.lower()
    if not pat: return []
    fail, k = [0] * len(pat), 0
    for i in range(1, len(pat)):
        while k and pat[i] != pat[k]: k = fail[k - 1]
        k += pat[i] == pat[k]; fail[i] = k
    res, k = [], 0
    for i, ch in enumerate(text):
        while k and ch != pat[k]: k = fail[k - 1]
        k += ch == pat[k]
        if k == len(pat): res.append(i - k + 1); k = fail[k - 1]
    return res


# ---- Sorting: timeline / history -------------------------------------------------
def merge_sort(items, key=lambda x: x):
    """Stable O(n log n) merge sort. O(n) extra space."""
    if len(items) < 2: return list(items)
    m = len(items) // 2
    a, b = merge_sort(items[:m], key), merge_sort(items[m:], key)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if key(b[j]) < key(a[i]): out.append(b[j]); j += 1
        else: out.append(a[i]); i += 1
    return out + a[i:] + b[j:]


# ---- Binary search: timestamp -> window ------------------------------------------
def window_at(starts, t):
    """Index of the last window whose start <= t, or -1. `starts` must be sorted. O(log n)."""
    lo, hi = 0, len(starts) - 1; ans = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if starts[mid] <= t: ans, lo = mid, mid + 1
        else: hi = mid - 1
    return ans


# ---- Hash map (+ LRU list): session / result lookup with retention ----------------
class ResultStore:
    """dict gives O(1) lookup by id; OrderedDict order gives O(1) LRU eviction.
    Entries also expire after `ttl` seconds, so audio-derived data is not kept forever."""
    def __init__(self, max_items=50, ttl=3600):
        self.d, self.max, self.ttl = OrderedDict(), max_items, ttl
    def put(self, k, v):
        self.d[k] = (time.time(), v); self.d.move_to_end(k)
        while len(self.d) > self.max: self.d.popitem(last=False)
    def get(self, k):
        e = self.d.get(k)
        if not e: return None
        if time.time() - e[0] > self.ttl: del self.d[k]; return None
        self.d.move_to_end(k); return e[1]
    def ids(self): return list(self.d)
