import json
import os
import glob

def count_annotations(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Assuming the JSON structure has a key 'segments' that contains the list of annotations
    if 'segments' in data:
        return len(data['segments'])
    else:
        print("No annotations found in the JSON file.")
        return 0
    
if __name__ == "__main__":
    json_files = glob.glob(os.path.join(os.getcwd(), 'Dataset/**', '*.json'), recursive=True)
 
    tot_count = 0
    for json_file in json_files:
        print(f"Processing file: {json_file}")
        count = count_annotations(json_file)
        print(f"Number of annotations in {json_file}: {count}")
        tot_count += count  

    print(f"Total number of annotations across all files: {tot_count}")s