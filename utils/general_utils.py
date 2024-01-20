import subprocess
import re
from typing import List, Union, Dict, Any
import time
import datetime
import numpy as np
from torch import Tensor
import torch
import hashlib
import pickle



# UL2 two gpu mapping
MODULES_ON_FIRST_GPU = [
    'shared','decoder.embed_tokens', 'encoder', 'lm_head', 'decoder.block.0', 'decoder.block.1', 'decoder.block.2',
]

MODULES_ON_SECOND_GPU = [
     'decoder.block.3', 'decoder.block.4', 'decoder.block.5', 'decoder.block.6', 'decoder.block.7', 'decoder.block.8', 'decoder.block.9', 'decoder.block.10', 'decoder.block.11', 'decoder.block.12', 'decoder.block.13', 'decoder.block.14', 'decoder.block.15', 'decoder.block.16', 'decoder.block.17', 'decoder.block.18', 'decoder.block.19', 'decoder.block.20', 'decoder.block.21', 'decoder.block.22', 'decoder.block.23', 'decoder.block.24', 'decoder.block.25', 'decoder.block.26', 'decoder.block.27', 'decoder.block.28', 'decoder.block.29', 'decoder.block.30', 'decoder.block.31', 'decoder.final_layer_norm', 'decoder.dropout'
]

def get_ul2_device_map(device_ids:str):
    device_ids = device_ids.split(',')
    # to int
    device_ids = [int(x) for x in device_ids]
    if len(device_ids) != 2:
        raise ValueError(f"len(device_ids) != 2")
    else:
        device_map = {}
        for module in MODULES_ON_FIRST_GPU:
            device_map[module] = device_ids[0]
        for module in MODULES_ON_SECOND_GPU:
            device_map[module] = device_ids[1]
        return device_map



# Function to get processes from nvidia-smi
def get_gpu_processes(gpu_id):
    result = subprocess.run(['nvidia-smi', '--id=' + gpu_id, '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True)
    return re.findall(r'\d+', result.stdout)

# Function to kill processes
def kill_process(pid):
    subprocess.run(['kill', '-9', pid])


def kill_gpu_process(target_gpus: Union[List[str],str]):
    if isinstance(target_gpus, str):
        target_gpus = target_gpus.split(',')
    for gpu in target_gpus:
        processes = get_gpu_processes(gpu)
        for pid in processes:
            kill_process(pid)
            print(f"Killed process {pid} on GPU {gpu}")


'''human readable utf+8 time'''
import datetime
def get_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')




def to_numpy(data: Dict[Any, List[Tensor]]) -> Dict[Any, List[np.ndarray]]:
    ''' Convert Dict[Any, List[Tensor]] to Dict[Any, List[np.ndarray]] '''
    return {
        key: [
            tensor.cpu().numpy() for tensor in tensor_list
        ] 
        for key, tensor_list in data.items()
    }

def to_tensor(data: Dict[Any, List[np.ndarray]]) -> Dict[Any, List[Tensor]]:
    ''' Convert Dict[Any, List[np.ndarray]] to Dict[Any, List[Tensor]] '''
    return {
        key: [
            torch.from_numpy(array) for array in array_list
        ]
        for key, array_list in data.items()
    }


# Function to check the remaining GPU memory for each device
def check_gpu_memory():
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory
        reserved_memory = torch.cuda.memory_reserved(i)
        free_memory = total_memory - reserved_memory
        gpu_memory.append({
            'device': i,
            'total_memory_GB': total_memory / (1024**3),
            'reserved_memory_GB': reserved_memory / (1024**3),
            'free_memory_GB': free_memory / (1024**3)
        })
    return gpu_memory

def find_string_in_file(search_string, file_path):
    """
    Searches for a specific string in a text file and prints out each line containing the string, along with the line number.

    :param file_path: Path to the text file.
    :param search_string: String to search for in the file.
    # Example usage:
    # find_string_in_file('path/to/your/file.txt', 'your_search_string')
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            found = False
            for line_number, line in enumerate(file, start=1):
                if search_string in line:
                    print(f"String found in file {file_path} on line {line_number}: {line.strip()}")
                    found = True
    except FileNotFoundError:
        print("The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Walk through the directory for all .py files
def find_py_files(directory_path):
    import os
    py_files = []
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            # Construct the full file path
            file_path = os.path.join(dirpath, filename)
            # print(file_path)
            # Check if the file is a python file
            if file_path.endswith('.py'):
                py_files.append(file_path)
    return py_files

def hash_object(obj):
    # Serialize the object using pickle
    serialized_obj = pickle.dumps(obj)
    
    # Create a SHA-256 hash of the serialized object
    hash_obj = hashlib.sha256(serialized_obj).hexdigest()
    
    return hash_obj
