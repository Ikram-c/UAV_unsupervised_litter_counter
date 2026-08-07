import json        
import os        
import sys
from pathlib import Path
from arguments_parser import parse_arguments
from post_process_web_odm import extract_from_odm_files 
from features_descrip_extractor import img_feature_matcher
from transistive_count import FinalCounter
from tqdm import tqdm
from typing import Dict, Any

def process_single_dataset_steps(txt_file_path: str, 
                                 img_sub_dir: str, 
                                 json_file_path: str, 
                                 pbar) -> bool:
    """
    Process a single dataset through all steps with progress tracking
        
    Args:
        txt_file_path (str): Path to ODM output text file
        img_sub_dir (str): Directory containing image dataset
        json_file_path (str): Path to JSON file with feature descriptions
        pbar (tqdm): Progress bar object for step tracking
        
    Returns:
        bool: Success status of processing
    """
    odm_output_sorter = extract_from_odm_files(txt_file_path, image_dir=img_sub_dir)
    sorted_img_pairs = odm_output_sorter.process_web_odm_output_txt_file()
    pbar.update(1)
    
    feature_matcher = img_feature_matcher(img_sub_dir, sorted_img_pairs, json_path=json_file_path)
    overlap_ids = feature_matcher.feature_and_img_overlap_finder()
    pbar.update(1)
    
    annocounter = FinalCounter(overlap_ids, json_file_path)
    final_counts = annocounter.count_annotations
    pbar.update(1)
    
    dataset_name = os.path.basename(img_sub_dir)
    
    output_path = Path(f'output/{dataset_name}_counts.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    
    if not output_path.exists():
        with open(output_path, 'w') as fp:
            json.dump(final_counts, fp)
    else:
        # If the JSON file already exists, read it, update it, and save it again
        with open(output_path, 'r') as fp:
            existing_data = json.load(fp)
        
        existing_data.update(final_counts)
        
        with open(output_path, 'w') as fp:
            json.dump(existing_data, fp)
    
    pbar.update(1)
    
    return True

def main(txt_file_path: str, img_sub_dir: str, json_file_path: str) -> bool:
    """
    Main execution function that processes a single dataset
        
    Args:
        txt_file_path (str): Path to ODM output text file
        img_sub_dir (str): Directory containing image dataset
        json_file_path (str): Path to JSON file with feature descriptions
        
    Returns:
        bool: Success status of processing
    """
    steps = ['Processing ODM output', 'Matching features', 
             'Counting overlaps', 'Saving results']
    with tqdm(total=len(steps), desc=f"Processing {img_sub_dir}", leave=False) as pbar:
        return process_single_dataset_steps(txt_file_path, img_sub_dir, 
                                          json_file_path, pbar)

def validate_batch_paths(user_inputs: Dict[str, Any]) -> bool:
    """
    Validate existence of all required batch processing directories
        
    Args:
        user_inputs (Dict[str, Any]): Dictionary containing super directory paths
        
    Returns:
        bool: True if all directories exist, False otherwise
    """
    required_dirs = ['img_super_dir', 'json_super_dir', 'txt_super_dir']
    return all(os.path.exists(user_inputs[dir_]) for dir_ in required_dirs)

def validate_single_paths(user_inputs: Dict[str, Any]) -> bool:
    """
    Validate existence of all required single processing paths
        
    Args:
        user_inputs (Dict[str, Any]): Dictionary containing file and directory paths
        
    Returns:
        bool: True if all paths exist, False otherwise
    """
    required_paths = ['txt_file_path', 'img_sub_dir', 'json_file_path']
    return all(os.path.exists(user_inputs[path]) for path in required_paths)

def process_batch(user_inputs: Dict[str, Any]) -> int:
    """
    Process multiple datasets in batch mode
        
    Args:
        user_inputs (Dict[str, Any]): Dictionary containing super directory paths
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        img_super_dir = Path(user_inputs['img_super_dir'])
        json_super_dir = Path(user_inputs['json_super_dir'])
        txt_super_dir = Path(user_inputs['txt_super_dir'])
        
        img_dir_list = sorted(os.listdir(img_super_dir))
        json_dir_list = sorted(os.listdir(json_super_dir))
        txt_dir_list = sorted(os.listdir(txt_super_dir))
        
        if not (len(img_dir_list) == len(json_dir_list) == len(txt_dir_list)):
            print("Error: Mismatch in number of files across directories")
            return 1
            
        successful = 0
        failed = 0
        
        with tqdm(total=len(img_dir_list), desc="Overall Progress", position=0) as pbar:
            for txt_file, img_dir, json_file in zip(txt_dir_list, img_dir_list, json_dir_list):
                txt_full_path = str(txt_super_dir / txt_file)
                img_full_path = str(img_super_dir / img_dir)
                json_full_path = str(json_super_dir / json_file)
                
                if main(txt_full_path, img_full_path, json_full_path):
                    successful += 1
                else:
                    failed += 1
                    print(f"Failed to process: {img_dir}")
                
                pbar.update(1)
        
        print(f"\nProcessing complete: {successful} successful, {failed} failed")
        return 0 if failed == 0 else 1
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return 1

def process_single(user_inputs: Dict[str, Any]) -> int:
    """
    Process a single dataset
        
    Args:
        user_inputs (Dict[str, Any]): Dictionary containing file and directory paths
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    success = main(user_inputs['txt_file_path'], 
                  user_inputs['img_sub_dir'], 
                  user_inputs['json_file_path'])
    
    status = "Successfully processed" if success else "Failed to process"
    print(f"{status} dataset: {os.path.basename(user_inputs['img_sub_dir'])}")
    return 0 if success else 1

if __name__ == "__main__":
    user_inputs = parse_arguments()
    
    processing_modes = {
        True: (validate_batch_paths, process_batch),
        False: (validate_single_paths, process_single)
    }

    validate_fn, process_fn = processing_modes[user_inputs['batch_processing']]

    if not validate_fn(user_inputs):
        print("Error: One or more required paths do not exist")
        sys.exit(1)
    
    sys.exit(process_fn(user_inputs))