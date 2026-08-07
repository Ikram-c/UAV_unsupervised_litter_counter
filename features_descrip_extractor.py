##########################################################################
import cv2
import numpy as np
from utils.helper_functions import coco_help
import os
from utils import plotting_utilities
from valid_overlaps import OverlapInitializer 

class img_feature_matcher:

    def __init__(self, img_sub_dir, sorted_img_pairs, json_path,
                 im1_path=None, im2_path=None):
        """
        Initialize the image feature matcher.

        Args:
            img_sub_dir (str): Directory containing image subsets.
            sorted_img_pairs (list): List of tuples for sorted image pairs.
            json_path (str): Path to the COCO JSON annotation file.
            im1_path (str, optional): Path to the first image 
            (Default to None).
            im2_path (str, optional): Path to the second image 
            (Defaults to None).
        """

        self.json_path = json_path
        self.coco_helper = coco_help.COCOAnnotationHelper(
            annotation_file = json_path)
        self.img_sub_dir = img_sub_dir
        self.sorted_img_pairs = sorted_img_pairs
        self.im1_path = im1_path
        self.im2_path = im2_path
        self.overlaps_list = []
    
    
    def _get_bounding_boxes(self, coco_json, image_id):
        """
        Get bounding boxes for a given image ID from COCO annotations.

        Args:
            coco_json (dict): COCO JSON object containing annotations.
            image_id (int): ID of the image.

        Returns:
            list: List of bounding box coordinates.
        """
        return [ann['bbox'] for ann in coco_json['annotations'] 
                if ann['image_id'] == image_id]

    def _load_images(self):
        """
        Load grayscale versions of the images specified by im1_path and im2_path.

        Returns:
            tuple: Two grayscale images as NumPy arrays.
        """
        im_grey_1 = cv2.imread(self.im1_path,cv2.IMREAD_GRAYSCALE)
        im_grey_2 = cv2.imread(self.im2_path,cv2.IMREAD_GRAYSCALE)
        return im_grey_1, im_grey_2


    def get_image_ids(self):
        """
        Retrieve the COCO image IDs for the current image paths.

        Raises:
            ValueError: If image IDs cannot be found in the COCO dataset.

        Returns:
            tuple: Image IDs for the two images.
        """
        img1_id = self.coco_helper.get_image_id_from_filename(
            self.im1_path)
        img2_id = self.coco_helper.get_image_id_from_filename(
            self.im2_path)
        if img1_id is None or img2_id is None:
            raise ValueError("Could not find image IDs in the COCO dataset")
            
        
        return img1_id, img2_id
    

    def img_akaze_matches(self):
        """
        Perform feature matching between the two images using AKAZE.

        Returns:
            dict or None: Dictionary containing matched features and keypoints 
                          or None if insufficient matches are found.
        """
        
        img1_grey, img2_grey = self._load_images()

        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(img1_grey, None)
        kp2, des2 = akaze.detectAndCompute(img2_grey, None)

        matcher = cv2.DescriptorMatcher_create(
            cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

        nn_matches = matcher.knnMatch(des1, des2, 2)

        # 4. Apply Lowe's ratio test
        good_matches = []
        for m, n in nn_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        features_dict = {
            "good_matches": good_matches,
            "kp1": kp1,
            "kp2":kp2
        }
        return features_dict if len(good_matches) > 10 else None


    def find_homography(self, features_dict):
        """
        Compute the homography matrix based on matched features.

        Args:
            features_dict (dict): Dictionary containing good matches and keypoints.

        Returns:
            numpy.ndarray: Homography matrix.
        """

        good_matches = features_dict.get("good_matches")
        kp1 = features_dict.get("kp1")
        kp2 = features_dict.get("kp2")
        
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
        return M
    
    
    
    def optional_align_image_and_plot(self, M, transformed_anns1, anns2):
        """
        Align the first image to the second using the homography matrix and plot the result.

        Args:
            M (numpy.ndarray): Homography matrix.
            transformed_anns1 (list): Transformed annotations for the first image.
            anns2 (list): Annotations for the second image.

        Returns:
            None
        """
        
        
        img1_grey, img2_grey = self._load_images(self.im1_path, 
                                                 self.im2_path)
        h1, w1 = img1_grey.shape[:2]
        h2, w2 = img2_grey.shape[:2]
        pts1 = np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(-1,1,2)
        
        # Transform img1 corners to img2 space
        dst1 = cv2.perspectiveTransform(pts1, M)
        mask1 = np.zeros((h2,w2), dtype=np.uint8)
        mask2 = np.zeros((h1,w1), dtype=np.uint8)
        
        
        img1_aligned = cv2.warpPerspective(cv2.imread(self.im1_path), M, (w2, h2))
        mask1_aligned = cv2.warpPerspective(mask1, M, (w2, h2))
        
        
        overlap_img1 = cv2.bitwise_and(img1_aligned, img1_aligned, mask=mask1_aligned)
        image_name_1 = os.path.basename[0](self.im1_path)
        image_name_2 = os.path.basename[0](self.im2_path)
        plotting_utilities.ImagePlotter.plot_annotations_on_image(
            img1_aligned, transformed_anns1, anns2, 
            os.path.join("output_plots", 
                         f'{image_name_1}_aligned_with{image_name_2}.jpg'))
        return None
    
    
    def process_img_pairs(self, pair):
        """
        Process a pair of images to find overlapping features and annotations.

        Args:
            pair (tuple): Tuple containing filenames of two images.

        Returns:
            list or None: List of overlapping annotation IDs or None if no overlaps are found.
        """
        
        self.im1_path = f"{self.img_sub_dir}/{pair[0]}"
        self.im2_path = f"{self.img_sub_dir}/{pair[1]}"
        
        features_dict = self.img_akaze_matches()
        if not features_dict:
            return None
        
        
        img1_id, img2_id = self.get_image_ids()
        M = self.find_homography(features_dict)
        overlap_finder = OverlapInitializer(img1_id, img2_id, M, 
                                            self.json_path)
        return overlap_finder.run_overlap_for_two_images()

    
    def feature_and_img_overlap_finder(self):
        """
        Iterate through the dataset to find overlapping annotation IDs for image pairs.

        Returns:
            list: List of tuples containing overlapping annotation IDs.
        """
        
        
        # iterate through each pair
        for pair in self.sorted_img_pairs:
            overlaps = self.process_img_pairs(pair)
            if overlaps:
                self.overlaps_list.extend(overlaps)
        
        return self.overlaps_list