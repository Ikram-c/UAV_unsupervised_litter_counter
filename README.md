Alpha 0.1 Iteration of the a notation count method, this was developed in November 2024, evidence of this is found in figure 1. 
![figure_1: date modified marks records for a sample of the files, from windows explorer](image.png)

This iteration relied on the use of OpenDroneMap to find the initial overlapping regions, the pairs would then be output in a txt file hence the overlapping pairs.

Note that this version being released publicly (as an alpha model) is cleared up version of the Official early release in the official github repo. To keep it consistent, none the less, bad/previous coding patterns that I once had were left within the script. Understand that several methods were used for the official approach, each with several nuanced methods of achieving both the initial over arching problem and the subsequent secondary problems.


# Installation
Installation:
Install UV: https://docs.astral.sh/uv/getting-started/installation/
Then run the command: uv init
The virtual environment should initialize
To get all the libraries simply type the following two commands:
uv add -r requirements.txt
uv sync




# Running Instructions

## Single dataset mode
 python script.py --txt_file_path path/to/text.txt --img_sub_dir path/to/images --json_file_path path/to/data.json --dataset_name dataset1

## Batch processing mode
 python script.py --batch_processing --img_super_dir path/to/img_folders --json_super_dir path/to/json_files --txt_super_dir path/to/text_files