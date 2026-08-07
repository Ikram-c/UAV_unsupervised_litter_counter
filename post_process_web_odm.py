
from odm_output_pair_sorter import PairSorter
import re
from collections import defaultdict
import os

class extract_from_odm_files:

    def __init__(self, odm_output_file, image_dir=None):
        self.success_pattern = r'Matching (DJI_\d+\.JPG) and (DJI_\d+\.JPG).*Success: True'
        self.failed_pattern = r'Matching (DJI_\d+\.JPG) and (DJI_\d+\.JPG).*(?:Success: False|Matches: FAILED)'
        self.image_dir = image_dir
        self.img_list = self.get_image_list() if image_dir else []
        self.odm_output_file = odm_output_file
    
    
    def get_image_list(self, directory=None):
        """Get list of images from the specified directory or the default directory."""
        dir_to_use = directory or self.image_dir
        if not dir_to_use:
            raise ValueError("No directory specified. Please provide a directory.")
        return [f for f in os.listdir(dir_to_use) if f.lower().endswith(('.jpg', '.jpeg'))]

    def parse_image_matching_log(self, log_content):
        """Parse the log content to find matching and failed pairs."""
        matching_pairs = re.findall(self.success_pattern, log_content)
        failed_pairs = re.findall(self.failed_pattern, log_content)
        return matching_pairs, failed_pairs

    def find_unmatched_images(self, matching_pairs, failed_pairs, img_list=None):
        """Find images that don't appear in matching or failed pairs."""
        if img_list is None:
            img_list = self.img_list
        
        matched_images = set(img for pair in matching_pairs + failed_pairs for img in pair)
        return [img for img in img_list if img not in matched_images]

    def get_matched_images(self, matching_pairs):
        """Get a dictionary of matched images."""
        matched_images = defaultdict(list)
        for img1, img2 in matching_pairs:
            matched_images[img1].append(img2)
            matched_images[img2].append(img1)
        return dict(matched_images)

    def get_no_match_images(self, matching_pairs, failed_pairs):
        """Find images that were attempted but had no successful matches."""
        image_status = defaultdict(lambda: {'successful': False, 'attempted': False})

        # Process matching pairs
        for img1, img2 in matching_pairs:
            image_status[img1]['successful'] = True
            image_status[img2]['successful'] = True
            image_status[img1]['attempted'] = True
            image_status[img2]['attempted'] = True

        # Process failed pairs
        for img1, img2 in failed_pairs:
            image_status[img1]['attempted'] = True
            image_status[img2]['attempted'] = True

        # Find images that were attempted but had no successful matches
        no_match_img = [img for img, status in image_status.items() 
                        if status['attempted'] and not status['successful']]

        return no_match_img

    def process_web_odm_output_txt_file(self, full_info=None):
        """Process the entire log file and return comprehensive results."""
        with open(self.odm_output_file, 'r') as file:
            odm_output_file = file.read()

        matching_pairs, failed_pairs = self.parse_image_matching_log(odm_output_file)
        unmatched_images = self.find_unmatched_images(matching_pairs, failed_pairs)
        matched_images_dict = self.get_matched_images(matching_pairs)
        no_match_images = self.get_no_match_images(matching_pairs, failed_pairs)


        initial_sorter = PairSorter(matching_pairs)
        sorted_pairs = initial_sorter.sort_pairs_into_lists()
        
        
        if full_info:
            return {
                'matching_pairs': sorted_pairs[0],
                'failed_pairs': failed_pairs,
                'unmatched_images': unmatched_images,
                'matched_images': matched_images_dict,
                'no_match_images': no_match_images
            }

        else:
            return sorted_pairs[0]