from utils.helper_functions._odm_out_seg_tree import SegmentTree


class PairSorter:
    def __init__(self, pairs):
        self.pairs = pairs
    
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

        tree = SegmentTree(tree_data)
        result = []
        used_pairs = set()
        
        for _ in range(len(remaining_pairs)):
            best = tree.query(0, len(remaining_pairs))
            if best is None or best[2] > 0:  # If best score > 0, we have a conflict
                break
                
            pair, should_swap, _ = best
            chosen_pair = pair if not should_swap else (pair[1], pair[0])
            
            for i, (p, swap, score) in enumerate(tree_data):
                if p == pair and tree.tree[tree.n + i] is not None:
                    tree.update(i, None)
                    used_pairs.add(pair)
                    break
            
            result.append(chosen_pair)
            last_pos[chosen_pair[0]] = 0
            last_pos[chosen_pair[1]] = 1
            
            for i, (p, _, _) in enumerate(tree_data):
                if p not in used_pairs:
                    normal_score = self.get_score(p, False, last_pos)
                    swapped_score = self.get_score(p, True, last_pos)
                    if normal_score <= swapped_score:
                        tree.update(i, (p, False, normal_score))
                    else:
                        tree.update(i, (p, True, swapped_score))
        
        return result, used_pairs

    def sort_pairs_into_lists(self):
        if not self.pairs:
            return {}
        
        result_list = []
        remaining = set(self.pairs)
        list_num = 1
        
        while remaining:
            current_list, used = self.try_build_list(list(remaining), {})
            if not current_list:  
                break
            
            result_list.append(current_list)
            remaining -= used
            list_num += 1
        
        # If there are still remaining pairs, add them to a final list
        if remaining:
            result_list.append(remaining)
        
        return result_list