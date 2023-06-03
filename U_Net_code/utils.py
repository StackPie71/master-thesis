# Imports
import torch
import math
import nibabel
from scipy import ndimage
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

def extract_patient_number(patient):
    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    patient_num = ''
    for char in patient:
        if char in num:
            patient_num += char
        if char == '_' or char == '.':
            break     
    return patient_num

def load_file(file_path):
    data = nibabel.load(file_path) # load file
    data = nibabel.as_closest_canonical(data) # transform into RAS orientation
    data = np.array(data.dataobj) # fetch data as float 32bit
    data = np.float32(data)
    return data


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        return iter(range(iter_start, iter_end))