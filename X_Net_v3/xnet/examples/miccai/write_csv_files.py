"""Script for writing cvs files
"""

import os
import csv
import pandas as pd
import random
from random import shuffle

# Vérifier ce que ça donne ici 
def create_csv_file(data_root, output_file, fields_ct, fields_mr):
    """
    create a csv file to store the paths of files for each patient
    """
    filenames_ct = []
    filenames_mr = []
    patient_names = os.listdir(data_root + '/' + fields_ct[1])
    print('total number of images {0:}'.format(len(patient_names)))
    for patient_name in patient_names:
        patient_image_names_ct = []
        patient_image_names_mr = []
        for field in fields_ct:
            image_name = data_root + '/' + field + '/' + patient_name
            if(field == 'data'):
                image_name = image_name.replace('_seg.', '.')
            image_name = image_name[len(data_root) + 1 :]
            patient_image_names_ct.append(image_name)
        filenames_ct.append(patient_image_names_ct)
        for field in fields_mr:
            image_name = data_root + '/' + field + '/' + patient_name
            if(field == 'data'):
                image_name = image_name.replace('_seg.', '.')
            image_name = image_name[len(data_root) + 1 :]
            patient_image_names_mr.append(image_name)
        filenames_mr.append(patient_image_names_mr)

    with open(output_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        fields = [fields_ct[0], fields_ct[1], fields_mr[0], fields_mr[1]]
        csv_writer.writerow(fields)
        for item_ct, item_mr in zip(filenames_ct, filenames_mr):
            item = [item_ct[0], item_ct[1], item_mr[0], item_mr[1]]
            csv_writer.writerow(item)

def random_split_dataset():
    random.seed(2019)
    input_file = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/examples/miccai/config/image_all.csv'
    train_names_file = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/examples/miccai/config/image_train.csv'
    valid_names_file = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/examples/miccai/config/image_valid.csv'
    test_names_file  = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/examples/miccai/config/image_test.csv'

    
    # TODO: Vérifier que les dataset sont les même pour ct et mr (patient présent et ordre !!)
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    shuffle(data_lines)
    train_lines  = data_lines[:16]
    valid_lines  = data_lines[16:]
    test_lines   = data_lines[16:]
    
    with open(train_names_file, 'w') as f:
        f.writelines(lines[:1] + train_lines)
    with open(valid_names_file, 'w') as f:
        f.writelines(lines[:1] + valid_lines)
    with open(test_names_file, 'w') as f:
        f.writelines(lines[:1] + test_lines)
    

def obtain_patient_names():
    """
    extract the patient names from csv files
    """
    split_names = ['train', 'valid', 'test']
    for split_name in split_names:
        csv_file = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/examples/miccai/config/image_{0:}.csv'.format(split_name)
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        data_lines = lines[1:]
        patient_names = []
        for data_line in data_lines:
            patient_name = data_line.split(',')[0]
            patient_name = patient_name[6:-7]
            print(patient_name)
            patient_names.append(patient_name)
        output_filename = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/examples/miccai/config/image_{0:}_names.txt'.format(split_name)
        with open(output_filename, 'w') as f:
            for patient_name in patient_names:
                f.write('{0:}\n'.format(patient_name))
        
if __name__ == "__main__":
    # create cvs file for fetal MR dataset
    data_root  = '/auto/home/users/n/b/nboulang/X_Net_v3/xnet/alldataset/origindata'
    # output_file = 'config/image_all.csv'
    output_file = "/auto/home/users/n/b/nboulang/X_Net_v3/xnet/examples/miccai/config/image_all.csv"
    fields_ct      = ['data_ct', 'label_ct']
    fields_mr      = ['data_mr', 'label_mr']
    create_csv_file(data_root, output_file, fields_ct, fields_mr)

    # split fetal MR dataset in to training, validation and testing
    random_split_dataset()

    # obtain image names the splitted dataset
    # obtain_patient_names() -> not usefull I guess 
