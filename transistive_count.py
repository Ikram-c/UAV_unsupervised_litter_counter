from typing import Dict, List, Set, Union
import networkx as nx
from collections import defaultdict
from utils.helper_functions import coco_help
from operator import itemgetter
import numpy as np



class FinalCounter:
    """Optimized counter class for processing annotations with overlap handling."""
    
    def __init__(self, overlaps: List[tuple], json_path: str):
        """
        Initialize the counter with optimized data structures.
        
        Args:
            overlaps: List of tuple pairs representing overlapping annotations
            json_path: Path to COCO annotation file
            table_dict: Lookup table mapping annotation IDs to category IDs
        """
        self.overlaps = overlaps
        self.coco_helper = coco_help.COCOAnnotationHelper(annotation_file=json_path)
        self.table_dict = self._build_lookup_table()
        self.count_dict = defaultdict(int)  # Maps category_id -> count
    
    
    def _build_lookup_table(self) -> Dict[int, int]:
        """
        Build lookup table mapping annotation IDs to category IDs from COCO annotations.
        Uses optimized methods depending on the data size and structure.
        
        Returns:
            Dictionary mapping annotation IDs to category IDs
        """
        annotations = self.coco_helper.coco.dataset['annotations']
        
        if len(annotations) < 1000:
            return {ann['id']: ann['category_id'] for ann in annotations}
        
        try:
            ann_array = np.array([(ann['id'], ann['category_id']) for ann in annotations],
                               dtype=[('id', np.int32), ('category_id', np.int32)])
            
            ann_array.sort(order='id')
            
            return dict(zip(ann_array['id'], ann_array['category_id']))
            
        except (MemoryError, ValueError):
            lookup_table = {}
            chunk_size = 10000
            
            for i in range(0, len(annotations), chunk_size):
                chunk = annotations[i:i + chunk_size]
                id_getter = itemgetter('id')
                cat_getter = itemgetter('category_id')
                
                lookup_table.update(
                    zip(map(id_getter, chunk), map(cat_getter, chunk))
                )
            
            return lookup_table
        
    def _count_helper(self, ann_id: int) -> None:
        """
        Increment count for category corresponding to annotation ID.
        
        Args:
            ann_id: Annotation ID to process
        """
        category_id = self.table_dict.get(ann_id)
        if category_id is not None:  # Only count if we have a valid category mapping
            self.count_dict[category_id] += 1
    
    def _filter_dataset_annotation_id_list(self, annotations_to_remove: Set[int]) -> List[int]:
        """
        Filter out removed annotation IDs efficiently using sets.
        
        Args:
            annotations_to_remove: Set of annotation IDs to exclude
            
        Returns:
            List of remaining annotation IDs
        """
        all_annotation_ids = set(self.table_dict.keys())  # All possible annotation IDs
        return list(all_annotation_ids - annotations_to_remove)
    
    def count_annotations(self) -> Dict[int, int]:
        """
        Count annotations handling overlapping cases.
        
        Returns:
            Dictionary mapping category IDs to their counts
        """
        g = nx.Graph(self.overlaps)
        
        annotations_to_remove = set()
        
        for component in nx.connected_components(g):
            ann_ids = list(component)
            if ann_ids:  # Make sure we have at least one annotation
                self._count_helper(int(ann_ids[0]))
                annotations_to_remove.update(ann_ids)
        
        filtered_ids = self._filter_dataset_annotation_id_list(annotations_to_remove)
        for ann_id in filtered_ids:
            self._count_helper(ann_id)
        
        return dict(self.count_dict)  