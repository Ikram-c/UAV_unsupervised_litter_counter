class _pairs_seg_tree:
    def __initsubclass__(self, data=None):
        self.n = len(data)
        self.tree = [None] * (2 * self.n)
        # Build the segment tree
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



class main_sorter:

    def __init__(self, pairs):
        self.pairs = pairs
        self.result_dict = {}
        self.remaining = set(self.pairs)
        self.list_num = 1

    def get_score(self, pair, swap, last_pos):
        first, second = pair if not swap else (pair[1], pair[0])
        score = 0
        if first in last_pos and last_pos[first] == 0:
            score += 1
        if second in last_pos and last_pos[second] == 1:
            score += 1
        return score
    
    def try_build_list(self, remaining_pairs, last_pos):
        if not remaining_pairs:
            return []
            
        tree_data = []
        for pair in remaining_pairs:
            normal_score = self.get_score(pair, False, last_pos)
            swapped_score = self.get_score(pair, True, last_pos)
            
            if normal_score <= swapped_score:
                tree_data.append((pair, False, normal_score))
            else:
                tree_data.append((pair, True, swapped_score))

        tree = _pairs_seg_tree(tree_data)
        result = []
        used_pairs = set()
        
        for _ in range(len(remaining_pairs)):
            best = tree.query(0, len(remaining_pairs))
            if best is None or best[2] > 0:  # If best score > 0, we have a conflict
                break
                
            pair, should_swap, _ = best
            chosen_pair = pair if not should_swap else (pair[1], pair[0])
            
            # Find and mark this pair as used
            for i, (p, swap, score) in enumerate(tree_data):
                if p == pair and tree.tree[tree.n + i] is not None:
                    tree.update(i, None)
                    used_pairs.add(pair)
                    break
            
            result.append(chosen_pair)
            last_pos[chosen_pair[0]] = 0
            last_pos[chosen_pair[1]] = 1
            
            # Update scores for remaining pairs
            for i, (p, _, _) in enumerate(tree_data):
                if p not in used_pairs:
                    normal_score = self.get_score(p, False, last_pos)
                    swapped_score = self.get_score(p, True, last_pos)
                    if normal_score <= swapped_score:
                        tree.update(i, (p, False, normal_score))
                    else:
                        tree.update(i, (p, True, swapped_score))
        
        return result, used_pairs



    def process_pairs(self):
        
        while self.remaining:
            current_list, used = self.try_build_list(list(self.remaining), {})
            if not current_list:
                break

            self.result_dict[f"list_{self.list_num}"] = current_list
            self.remaining -= used
            self.list_num += 1

        if self.remaining:
            self.result_dict[f"list_{self.list_num}"] = list(self.remaining)
            

        return self.result_dict