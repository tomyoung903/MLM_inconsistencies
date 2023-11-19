import subprocess
import re
from typing import List, Union, Dict, Any
import time
import datetime
import numpy as np
from torch import Tensor
import torch

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