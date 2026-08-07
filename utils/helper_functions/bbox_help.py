from shapely.geometry import Polygon
import numpy as np


class BoundingBoxHelper:
    @staticmethod
    def bbox_overlap(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        
        return intersection / min(area1, area2)

    @staticmethod
    def polygon_in_mask(polygon, mask):
        mask_polygon = Polygon(np.column_stack(np.where(mask > 0)))
        return polygon.intersects(mask_polygon)