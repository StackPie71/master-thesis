print("Starting X-Net")

# Imports
import os
import torchio as tio
import torch
from model import UNet
from utils import extract_patient_number
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Loss import DiceCELoss

# Dataset creation ##############################################################################

# Loading data
print("Loading data...")
path_ct = "/auto/home/users/n/b/nboulang/X_Net/raw_data/Task001_HNC_segm/imagesTr_CT/"
subjects_paths_ct = os.listdir(path_ct)
paths_ct = "/auto/home/users/n/b/nboulang/X_Net/raw_data/Task001_HNC_segm/labelsTr_CT/"
labels_paths_ct = os.listdir(paths_ct)


subjects_ct = []
for label in labels_paths_ct:
    for patient in subjects_paths_ct:
        if extract_patient_number(label) == extract_patient_number(patient):
            print("patient number:", extract_patient_number(patient))
            subject_ct = tio.Subject({"CT":tio.ScalarImage(os.path.join(path_ct, patient)), "Label":tio.LabelMap(os.path.join(paths_ct, label))})
            subjects_ct.append(subject_ct)
    

for subject in subjects_ct:
    assert subject["CT"].orientation == ("R", "A", "S")

print("len(subjects_ct): ", len(subjects_ct))   # 20
    
# Augmentation 
process = tio.Compose([
            tio.CropOrPad((512, 512, 160)),
            tio.RescaleIntensity((0, 1))
            ])

augmentation = tio.RandomAffine(scales=(0.9, 1.1), degrees=(-10, 10))
val_transform = process
train_transform = tio.Compose([process, augmentation])

# Data split
train_ct, val_ct = train_test_split(subjects_ct, test_size=0.2, random_state=42)
del subjects_ct

train_dataset_ct = tio.SubjectsDataset(train_ct, transform=train_transform)

del train_ct
del val_ct

sampler = tio.data.UniformSampler(patch_size=(128, 128, 40))

# Creating queue to draw patches from
print("Creating queues...")
train_patches_queue_ct = tio.Queue(
     train_dataset_ct,
     max_length=64,
     samples_per_volume=16,
     sampler=sampler,
     num_workers=1,
     shuffle_subjects=False,
     shuffle_patches=False,
     verbose=True
    )
print("DONE")

print("train_patches_queue_ct: ", train_patches_queue_ct)


# Define train and val loader
print("Define train and val loader...")
batch_size = 1
train_loader_ct = DataLoader(train_patches_queue_ct, batch_size=batch_size, num_workers=0)

# Train the neural network for 5 epochs
print("Training...")
MAX_EPOCH = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
for epoch in range(MAX_EPOCH):
    print("Epoch ", epoch)

    #reset iterator
    dataiter_ct = iter(train_loader_ct)
    # dataiter_mr = iter(train_loader_mr)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for batch_ct in dataiter_ct:    # https://torchio.readthedocs.io/patches/patch_training.html
        # batch_mr = next(dataiter_mr)

        # You can obtain the raw volume arrays by accessing the data attribute of the subject
        img_ct = batch_ct["CT"][tio.DATA].to(device)
        mask_ct = batch_ct["Label"][tio.DATA][:,0].to(device)  # Remove single channel as CrossEntropyLoss expects NxHxW
        mask_ct = mask_ct.long()

        #reset gradients
        optimizer.zero_grad()

        #forward propagation through the network
        pred_ct = model.forward(img_ct)
        
        #calculate the loss
        loss = model.loss_fn(pred_ct, mask_ct)
        
        #backpropagation
        loss.backward()
        
        #update the parameters
        optimizer.step()

print("Training done")

# Saving model 
PATH = "unet_5_epoch.pt"
torch.save({
            'epoch': MAX_EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)