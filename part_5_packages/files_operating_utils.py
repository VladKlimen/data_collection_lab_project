"""Files operating utilities"""
import os
import pickle
from pathlib import Path

def save_to_pickle(data, file_name, dest_dir="/Workspace/Users/vladklim@campus.technion.ac.il/Project/models/"):
    Path("/Workspace/Users/vladklim@campus.technion.ac.il/Project/models/").mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(dest_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


# This used to split files larger than 10mb, because databricks didn't allow to download large files 
def split(source, dest_dir, files_name, write_size=10**7):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    else:
        for file in os.listdir(dest_dir):
            os.remove(os.path.join(dest_dir, file))
    part_num = 0
    
    with open(source, 'rb') as input_file:
        while True:
            chunk = input_file.read(write_size)
            if not chunk:
                break
            
            part_num += 1
            file_path = os.path.join(dest_dir, files_name + str(part_num))
            
            with open(file_path, 'wb') as dest_file:
                dest_file.write(chunk)
    
    print(f"Partitions created: {part_num}")


# Used to join the splitted files back
def join(source_dir, dest_file, read_size):
    with open(dest_file, 'wb') as output_file:
        for path in os.listdir(source_dir):
            with open(path, 'rb') as input_file:
                while True:
                    bytes = input_file.read(read_size)
                    if not bytes:
                        break
                output_file.write(bytes)