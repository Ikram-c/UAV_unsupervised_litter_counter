class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [None] * (2 * self.n)
        for i in range(self.n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self._merge(self.tree[2 * i], self.tree[2 * i + 1])
    
    def _merge(self, left, right):
        if left is None:
            return right
        if right is None:
            return left
        return left if left[2] < right[2] else right

    def update(self, index, value):
        pos = index + self.n
        self.tree[pos] = value
        while pos > 1:
            pos //= 2
            self.tree[pos] = self._merge(self.tree[2 * pos], self.tree[2 * pos + 1])
    
    def query(self, left, right):
        res = None
        left += self.n
        right += self.n
        while left < right:
            if left % 2 == 1:
                res = self._merge(res, self.tree[left]) if res is not None else self.tree[left]
                left += 1
            if right % 2 == 1:
                right -= 1
                res = self._merge(res, self.tree[right]) if res is not None else self.tree[right]
            left //= 2
            right //= 2
        return res