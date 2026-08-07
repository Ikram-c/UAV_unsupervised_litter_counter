##########################################################################
import cv2
import numpy as np
from utils.helper_functions import coco_help
from collections import defaultdict


class OverlapInitializer:
    
    
    def __init__(self, img1_id, img2_id, M, json_file_path):
        self.coco_helper = coco_help.COCOAnnotationHelper(json_file_path)
        self.M = M
        self.anns1 = self.coco_helper.get_image_annotations(img1_id)
        self.anns2 = self.coco_helper.get_image_annotations(img2_id)
        self.ann2_by_category, self.anns2_tup= self._category_lookup(self.anns2)
        
    
    
    def _category_lookup(self, anns):
        """
        Organize annotations by category and transform bounding boxes into a standard format.
        
        Args:
            anns: List of annotation dictionaries containing 'bbox' and 'id'
            
        Returns:
            tuple: (
                dict: Annotations organized by category ID,
                list: List of tuples containing (x_min, y_min, width, height, category_id, annotation_id)
            )
        """
        ann2_by_category = {}
        anns2_tuple_list = []
        
        for ann in anns:
            if 'bbox' in ann and 'id' in ann:
                cat_id = self.coco_helper.get_category_id_from_annotation(ann['id'])
                if cat_id:
                    if cat_id not in ann2_by_category:
                        ann2_by_category[cat_id] = []
                    ann2_by_category[cat_id].append(ann)
                    
                    # Get bbox coordinates
                    # COCO format bbox is [x, y, width, height]
                    bbox = ann['bbox']
                    x_min = bbox[0]
                    y_min = bbox[1]
                    width = bbox[2]
                    height = bbox[3]
                    bbox_temp = (x_min, y_min, x_min + width, y_min + height)
                    rounded_bbox = tuple(map(lambda x: int(np.round(x)), bbox_temp))
                    anns2_tuple_list.append([rounded_bbox[0], 
                                             rounded_bbox[1], 
                                             rounded_bbox[2], 
                                             rounded_bbox[3],
                                             cat_id, 
                                             ann['id']])

        return ann2_by_category, anns2_tuple_list



    def transformed_anns1(self):
        transformed_ann1_bbox_list = []
        for ann1 in self.anns1:
            if 'bbox' in ann1 and 'id' in ann1:
                category_id_1 = self.coco_helper.get_category_id_from_annotation(ann1['id'])
                
                # Ensure category_id_1 exists and is valid
                if category_id_1 is not None and category_id_1 in self.ann2_by_category:
                    bbox1 = ann1['bbox']

                    # Define corners of the bounding box (top-left, top-right, bottom-left, bottom-right)
                    corners = np.array([
                        [bbox1[0], bbox1[1]],  # top-left
                        [bbox1[0] + bbox1[2], bbox1[1]],  # top-right
                        [bbox1[0], bbox1[1] + bbox1[3]],  # bottom-left
                        [bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]  # bottom-right
                    ]).astype(np.float32).reshape(-1, 1, 2)

                    # Apply perspective transformation
                    transformed_corners = cv2.perspectiveTransform(corners, self.M).reshape(-1, 2)

                    # Calculate new bounding box coordinates
                    x_min, y_min = np.min(transformed_corners, axis=0)
                    x_max, y_max = np.max(transformed_corners, axis=0)

                    # Calculate width and height
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    bbox_temp = (x_min, y_min, x_min + width, y_min + height)
                    rounded_bbox = tuple(map(lambda x: int(np.round(x)), bbox_temp))

                    # Append transformed bounding box to the list
                    transformed_ann1_bbox_list.append([rounded_bbox[0], 
                                                       rounded_bbox[1], 
                                                       rounded_bbox[2], 
                                                       rounded_bbox[3], 
                                                       category_id_1, 
                                                       ann1['id']])

                    
        return transformed_ann1_bbox_list
    

        
    

    def compare_rectangle_sets(self, set1_rects, set2_rects):
        """
        Compare two sets of rectangles and find all overlaps between them, considering category IDs.
        
        Args:
            set1_rects: List of [x0, y0, x1, y1, category_id, annotation_id] rectangles from the first set
            set2_rects: List of [x0, y0, x1, y1, category_id, annotation_id] rectangles from the second set
        
        Returns:
            overlaps: List of tuples (annotation_id_1, annotation_id_2) for overlapping rectangles with matching category IDs
        """
        if not set1_rects or not set2_rects:
            return []
        
        # Combine all rectangles but keep track of which set they're from
        events = defaultdict(list)
        
        # Add rectangles from set 1 (marked with set_id=0)
        for i, r in enumerate(set1_rects):
            events[r[0]].append((r[1], r[3], 1, i, 0))    # Start event
            events[r[2]].append((r[1], r[3], -1, i, 0))   # End event
        
        # Add rectangles from set 2 (marked with set_id=1)
        for i, r in enumerate(set2_rects):
            events[r[0]].append((r[1], r[3], 1, i, 1))    # Start event
            events[r[2]].append((r[1], r[3], -1, i,1))   # End event
        
        # Get sorted x-coordinates
        xs = sorted(events.keys())
        
        # Create arrays to track active rectangles from each set
        max_y = max(max(r[3] for r in set1_rects), max(r[3] for r in set2_rects)) + 1
        set1_arr = np.zeros(max_y + 1, "int16")
        set2_arr = np.zeros(max_y + 1, "int16")
        
        # Dictionary to store active rectangles at each y position
        active_rects = defaultdict(lambda: {"set1": set(), "set2": set()})
        
        # List to store overlaps
        overlaps = []
        
        x0 = xs[0]  # Previous x coordinate
        
        # Process all events
        for x in xs[1:]:
            # Process events at current x
            for y0, y1, sign, rect_id, set_id in events[x0]:
                # Update active rectangles
                # Clamp y0 and y1 to ensure they're within bounds
                y0_clamped = max(0, min(y0, max_y - 1))
                y1_clamped = max(0, min(y1, max_y - 1))
                for y in range(y0_clamped, y1_clamped):
                    if set_id == 0:
                        if sign == 1:
                            active_rects[y]["set1"].add(rect_id)
                        else:
                            active_rects[y]["set1"].discard(rect_id)
                        set1_arr[y] += sign
                    else:
                        if sign == 1:
                            active_rects[y]["set2"].add(rect_id)
                        else:
                            active_rects[y]["set2"].discard(rect_id)
                        set2_arr[y] += sign
            
            # Check for overlaps in the current x-segment
            width = x - x0
            if width > 0:  # Only process if there's a non-zero width
                for y in range(max_y):
                    if set1_arr[y] > 0 and set2_arr[y] > 0:
                        # There's an overlap at this position
                        # Add overlapping pairs with matching category IDs
                        for rect1 in active_rects[y]["set1"]:
                            for rect2 in active_rects[y]["set2"]:
                                if set1_rects[rect1][4] == set2_rects[rect2][4]:  # Check if category IDs match
                                    annotation_id_1 = set1_rects[rect1][5]
                                    annotation_id_2 = set2_rects[rect2][5]
                                    pair = tuple(map(lambda x: str(x), (annotation_id_1, annotation_id_2)))
                                    if pair not in overlaps:
                                        overlaps.append(pair)
            
            x0 = x  
        
        return overlaps


    def run_overlap_for_two_images(self):
        """This function will run the methods within the class and returns a list of tuple pairs

        Returns:
            _type_: _description_
        """
        
        transformed_ann1_bbox_list = self.transformed_anns1()
        return self.compare_rectangle_sets(transformed_ann1_bbox_list, 
                                               self.anns2_tup)