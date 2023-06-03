# Imports
import os
import torchio as tio
import torch
import numpy as np
import json
from numpyencoder import NumpyEncoder
from model import UNet
from utils import extract_patient_number
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from IPython.display import HTML
from celluloid import Camera

# Dataset creation ##############################################################################

# Loading data
print("Loading data...")
path_ct = "/auto/home/users/n/b/nboulang/X_Net/raw_data/Task001_HNC_segm/imagesTr_CT/"
# path_mr = "/auto/home/users/n/b/nboulang/X_Net/raw_data/Task001_HNC_segm/imagesTr_MR/"
subjects_paths_ct = os.listdir(path_ct)
# subjects_paths_mr = os.listdir(path_mr)
paths_ct = "/auto/home/users/n/b/nboulang/X_Net/raw_data/Task001_HNC_segm/labelsTr_CT/"
# paths_mr = "/auto/home/users/n/b/nboulang/X_Net/raw_data/Task001_HNC_segm/labelsTr_MR/"
labels_paths_ct = os.listdir(paths_ct)
# labels_paths_mr = os.listdir(paths_mr)


subjects_ct = []
# subjects_mr = []
for label in labels_paths_ct:
    for patient in subjects_paths_ct:
        if extract_patient_number(label) == extract_patient_number(patient):
            print("patient number:", extract_patient_number(patient))
            subject_ct = tio.Subject({"CT":tio.ScalarImage(os.path.join(path_ct, patient)), "Label":tio.LabelMap(os.path.join(paths_ct, label))})
            print("CT patient: ", patient)
            subjects_ct.append(subject_ct)
    
# for label in labels_paths_ct:
#     for patient in subjects_paths_mr:
#         if extract_patient_number(label) == extract_patient_number(patient):
#             print("patient number:", extract_patient_number(patient))
#             subject_mr = tio.Subject({"MR":tio.ScalarImage(os.path.join(path_mr, patient)), "Label":tio.LabelMap(os.path.join(paths_mr, label))})
#             print("CT patient: ", patient)
#             subjects_mr.append(subject_mr)

# Testing one queue/loader: 1 patient with multiple images and multiple labels => could be usefull to apply the same transform maybe
# subjects_paths_ct.sort()
# subjects_paths_mr.sort()
# labels_paths_ct.sort()
# labels_paths_mr.sort()
# subjects = []
# for image, label in zip(subjects_paths_ct, labels_paths_ct):
#     subject = tio.Subject({'CT': tio.ScalarImage(os.path.join(path_ct, image)), 'MR': tio.ScalarImage(os.path.join(path_mr, image)), "label_ct": tio.LabelMap(os.path.join(paths_ct, label)), "label_ct": tio.LabelMap(os.path.join(paths_ct, label))})

for subject in subjects_ct:
    assert subject["CT"].orientation == ("R", "A", "S")
# for subject in subjects_mr:
#     assert subject["MR"].orientation == ("R", "A", "S")


# Augmentation ##############################################################################

# process = tio.Compose([
#             tio.CropOrPad((512, 512, 160)),
#             tio.RescaleIntensity((0, 1))
#             ])

# augmentation = tio.RandomAffine(scales=(0.9, 1.1), degrees=(-10, 10))
# val_transform = process
# train_transform = tio.Compose([process, augmentation])

# Data split
train_ct, val_ct = train_test_split(subjects_ct, test_size=0.2, random_state=42)
# train_mr, val_mr = train_test_split(subjects_mr, test_size=0.2, random_state=42)

del subjects_ct
# del subjects_mr

# val_dataset_ct = tio.SubjectsDataset(val_ct, transform=val_transform)
# val_dataset_mr = tio.SubjectsDataset(val_mr, transform=val_transform)
val_dataset_ct = tio.SubjectsDataset(val_ct)
# val_dataset_mr = tio.SubjectsDataset(val_mr)

del val_ct
# del val_mr

# Creating queue to draw patches from ####################################################

patch_size = (128, 128, 40)
sampler = tio.data.UniformSampler(patch_size=patch_size)

val_patches_queue_ct = tio.Queue(
     val_dataset_ct,
     max_length=64,
     samples_per_volume=16,
     sampler=sampler,
     num_workers=1,
     shuffle_subjects=False,
     shuffle_patches=False,
     verbose=True
    )
print("DONE")

# val_patches_queue_mr = tio.Queue(
#      val_dataset_mr,
#      max_length=64,
#      samples_per_volume=16,
#      sampler=sampler,
#      num_workers=1,
#      shuffle_subjects=False,
#      shuffle_patches=False,
#      verbose=True
#     )

# print("DONE (2/2)")

# Define train and val loader
print("Define train and val loader...")
batch_size = 1
val_loader_ct = DataLoader(val_patches_queue_ct, batch_size=batch_size, num_workers=0)
# val_loader_mr = DataLoader(val_patches_queue_mr, batch_size=batch_size, num_workers=0)


# Evalutation ##########################################################################

# Function to compute dice accuracy
def dice(pred, mask):
    seg = pred.numpy()
    gt = mask.numpy()
    dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))
    print('Dice similarity score is {}'.format(dice))

# Load the model and place it on the gpu
print("Evaluation...")
model = UNet()

PATH = "model_5_epoch.pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

dataiter_ct = iter(val_loader_ct)
# dataiter_mr = iter(val_loader_mr)
with torch.no_grad():
    for i in range(4):
        print("Evaluation patient ", i)

        # # Select a validation subject and extract the images and segmentation for evaluation
        mask_ct = val_dataset_ct[i]["Label"][tio.DATA]
        # mask_mr = val_dataset_mr[i]["Label"][tio.DATA]
        imgs_ct = val_dataset_ct[i]["CT"][tio.DATA]
        # imgs_mr = val_dataset_mr[i]["MR"][tio.DATA]

        # GridSampler
        grid_sampler_ct = tio.inference.GridSampler(val_dataset_ct[i], patch_size, (8, 8, 8))
        # grid_sampler_mr = tio.inference.GridSampler(val_dataset_mr[i], patch_size, (8, 8, 8))

        # GridAggregator
        aggregator_ct = tio.inference.GridAggregator(grid_sampler_ct)
        # aggregator_mr = tio.inference.GridAggregator(grid_sampler_mr)

        # DataLoader for speed up
        patch_loader_ct = torch.utils.data.DataLoader(grid_sampler_ct, batch_size=batch_size)
        # patch_loader_mr = torch.utils.data.DataLoader(grid_sampler_mr, batch_size=batch_size)

        # # Prediction
        with torch.no_grad():
            for patches_batch_ct in patch_loader_ct:
                input_tensor_ct = patches_batch_ct['CT'][tio.DATA].to(device)  # Get batch of patches
                # input_tensor_mr = patches_batch_mr['MR'][tio.DATA].to(device)  # Get batch of patches
                locations_ct = patches_batch_ct[tio.LOCATION].to(device)  # Get locations of patches
                # locations_mr = patches_batch_mr[tio.LOCATION].to(device)  # Get locations of patches
                pred_ct = model(input_tensor_ct)  # Compute prediction
                aggregator_ct.add_batch(pred_ct, locations_ct)  # Combine predictions to volume
                # aggregator_mr.add_batch(pred_mr, locations_mr)  # Combine predictions to volume

        # Extract the volume prediction + accuracy
        output_tensor_ct = aggregator_ct.get_output_tensor()
        # output_tensor_mr = aggregator_mr.get_output_tensor()

        print(output_tensor_ct)
        # print("CT:")
        # dice(output_tensor_ct, mask_ct)
        # print("MR:")
        # dice(output_tensor_mr, mask_mr)

        # # Save results 
        # img_name = 'image'
        # mask_name = 'mask'
        # pred_name = 'prediction'
        # # CT
        # results_ct = {img_name: imgs_ct.numpy(), mask_name: mask_ct.numpy(), pred_name: output_tensor_ct.numpy()}
        # path = 'predictions/results_5_ct' + str(i) + '.json'
        # js_ct = json.dumps(results_ct, cls=NumpyEncoder)
        # fp = open(path, 'a')
        # fp.write(js_ct)
        # fp.close()
        # # MR
        # results_mr = {img_name: imgs_mr.numpy(), mask_name: mask_mr.numpy(), pred_name: output_tensor_mr.numpy()}
        # path = 'predictions/results_5_mr' + str(i) + '.json'
        # js_mr = json.dumps(results_mr, cls=NumpyEncoder)
        # fp = open(path, 'a')
        # fp.write(js_mr)
        # fp.close()
        # print("Results saved in ", path)
        
        # Iteration to next patient
        i += 1

print("Evaludation done")

# # Visualize the prediction
fig = plt.figure()
camera = Camera(fig)  # create the camera object from celluloid
pred = output_tensor_ct.argmax(0)

for i in range(0, output_tensor_ct.shape[3], 2):  # axial view
    plt.imshow(imgs_ct[0,:,:,i], cmap="bone")
    mask_ = np.ma.masked_where(pred[:,:,i]==0, pred[:,:,i])
    label_mask = np.ma.masked_where(mask_ct[0,:,:,i]==0, mask_ct[0,:,:,i])
    plt.imshow(mask_, alpha=0.1, cmap="autumn")
    plt.imshow(label_mask, alpha=0.5, cmap="jet")  # Uncomment if you want to see the label
    if i == 86:
        plt.savefig('test_countour_4297.png')
    # plt.axis("off")
    camera.snap()  # Store the current slice

animation = camera.animate()  # create the animation
  # convert the animation to a video
#HTML(animation.to_html5_video())
with open("myvideo.html", "w") as f:
    print(animation.to_html5_video(), file=f)