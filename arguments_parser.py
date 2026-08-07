from argparse import ArgumentParser
from typing import Dict, Any

def parse_arguments() -> Dict[str, Any]:
    """_summary_
        Parse command line arguments with mutually exclusive groups for 
        single vs batch processing
        
    Args:
        None
        
    Returns:
        Dict[str, Any]: Dictionary of parsed arguments based on processing mode
    """
    parser = ArgumentParser(description='Process image datasets')
    parser.add_argument('--batch_processing', action='store_true',
                      help='Enable batch processing mode')
    
    # Create mutually exclusive argument groups
    single_mode = parser.add_argument_group('Single Dataset Processing')
    batch_mode = parser.add_argument_group('Batch Processing')
    
    # Single dataset arguments
    single_mode.add_argument('--txt_file_path', type=str,
                          help='Path to single ODM output text file',
                          default="")
    single_mode.add_argument('--img_sub_dir', type=str,
                          help='Path to single image directory',
                          default="")
    single_mode.add_argument('--json_file_path', type=str,
                          help='Path to single JSON file',
                          default="")
    
    # Batch processing arguments
    batch_mode.add_argument('--img_super_dir', type=str,
                         help='Directory containing multiple image dataset folders')
    batch_mode.add_argument('--json_super_dir', type=str,
                         help='Directory containing multiple JSON files')
    batch_mode.add_argument('--txt_super_dir', type=str,
                         help='Directory containing multiple ODM output text files')
    
    args = parser.parse_args()
    
    # Validate required arguments based on mode
    if args.batch_processing:
        required = ['img_super_dir', 'json_super_dir', 'txt_super_dir']
        missing = [arg for arg in required if getattr(args, arg) is None]
        if missing:
            parser.error(f"Batch processing requires: {', '.join(missing)}")
    else:
        required = ['txt_file_path', 'img_sub_dir', 'json_file_path']
        missing = [arg for arg in required if getattr(args, arg) is None]
        if missing:
            parser.error(f"Single dataset processing requires: {', '.join(missing)}")
    
    return vars(args)



# # Single dataset mode
# python script.py --txt_file_path path/to/text.txt --img_sub_dir path/to/images --json_file_path path/to/data.json --dataset_name dataset1

# # Batch processing mode
# python script.py --batch_processing --img_super_dir path/to/img_folders --json_super_dir path/to/json_files --txt_super_dir path/to/text_files