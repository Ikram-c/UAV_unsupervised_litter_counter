
from pycocotools.coco import COCO
from heapq import heapify
import os
class COCOAnnotationHelper:
    def __init__(self, annotation_file):
        self.coco = COCO(annotation_file)

    def get_image_id_from_filename(self, filename):
        for img in self.coco.dataset['images']:
            if img['file_name'] == os.path.basename(filename):
                return img['id']
        return None

    def get_image_annotations(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        return self.coco.loadAnns(ann_ids)

    def get_bounding_boxes(self, image_id):
        return [tuple(ann['bbox']) for ann in self.get_image_annotations(image_id)]
    
    def get_annotation_ids_heap(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        heapify(ann_ids)  # Convert list to min-heap in-place
        return ann_ids
    
    def get_category_id_from_annotation(self, annotation_id):
        """
        Get the category ID for a given annotation ID.
        
        Args:
            annotation_id (int): The ID of the annotation
            
        Returns:
            int: The category ID of the annotation, or None if annotation not found
        """
        ann = self.coco.loadAnns([annotation_id])
        if ann and len(ann) > 0:
            return ann[0]['category_id']
        return None