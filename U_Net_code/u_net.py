# Imports
from torch.nn.modules import Module
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import UNet

# Create the Segmentation model.
class Segmenter(Module):  #pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = UNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, data_ct, data_mr):
        # pred_ct, pred_mr = self.model(data_ct, data_mr)
        pred_ct = self.model(data_ct, data_mr)
        return pred_ct#, pred_mr
    
    # def training_step(self, batch, batch_idx):
    #     print("Training step")
    #     # You can obtain the raw volume arrays by accessing the data attribute of the subject
    #     img_ct = batch["loader_a"]["CT"]["data"]
    #     img_mr = batch["loader_b"]["MR"]["data"]
    #     mask_ct = batch["loader_a"]["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
    #     mask_mr = batch["loader_b"]["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
    #     mask_ct = mask_ct.long()
    #     mask_mr = mask_mr.long()
        
    #     pred_ct, pred_mr = self(img_ct, img_mr)
    #     loss_ct = self.loss_fn(pred_ct, mask_ct)
    #     loss_mr = self.loss_fn(pred_mr, mask_mr)
        
    #     # Logs
    #     self.log("Train Loss CT", loss_ct)
    #     self.log("Train Loss MR", loss_mr)
    #     print("Train Loss CT", loss_ct)
    #     print("Train Loss MR", loss_mr)
    #     if batch_idx[0] % 50 == 0:
    #         self.log_images(img_ct.cpu(), pred_ct.cpu(), mask_ct.cpu(), "Train CT")
    #     if batch_idx[1] % 50 == 0:
    #         self.log_images(img_mr.cpu(), pred_mr.cpu(), mask_mr.cpu(), "Train MR")
    #     return loss_ct  #, loss_mr
    
        
    # def validation_step(self, batch, batch_idx):
    #     print("Validation step")
    #     # You can obtain the raw volume arrays by accessing the data attribute of the subject
    #     img_ct = batch["loader_a"]["CT"]["data"]
    #     img_mr = batch["loader_b"]["MR"]["data"]
    #     mask_ct = batch["loader_a"]["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
    #     mask_mr = batch["loader_b"]["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
    #     mask_ct = mask_ct.long()
    #     mask_mr = mask_mr.long()
        
    #     pred_ct, pred_mr = self(img_ct, img_mr)
    #     loss_ct = self.loss_fn(pred_ct, mask_ct)
    #     loss_mr = self.loss_fn(pred_mr, mask_mr)
        
    #     # Logs
    #     self.log("Train Loss CT", loss_ct)
    #     self.log("Train Loss MR", loss_mr)
    #     print("Train Loss CT", loss_ct)
    #     print("Train Loss MR", loss_mr)
    #     if batch_idx[0] % 50 == 0:
    #         self.log_images(img_ct.cpu(), pred_ct.cpu(), mask_ct.cpu(), "Train CT")
    #     if batch_idx[1] % 50 == 0:
    #         self.log_images(img_mr.cpu(), pred_mr.cpu(), mask_mr.cpu(), "Train MR")
    #     return loss_ct #, loss_mr

    
    # def log_images(self, img, pred, mask, name):
        
    #     results = []
    #     pred = torch.argmax(pred, 1) # Take the output with the highest value
    #     axial_slice = 50  # Always plot slice 50 of the 96 slices
        
    #     fig, axis = plt.subplots(1, 2)
    #     axis[0].imshow(img[0][0][:,:,axial_slice], cmap="bone")
    #     mask_ = np.ma.masked_where(mask[0][:,:,axial_slice]==0, mask[0][:,:,axial_slice])
    #     axis[0].imshow(mask_, alpha=0.6)
    #     axis[0].set_title("Ground Truth")
        
    #     axis[1].imshow(img[0][0][:,:,axial_slice], cmap="bone")
    #     mask_ = np.ma.masked_where(pred[0][:,:,axial_slice]==0, pred[0][:,:,axial_slice])
    #     axis[1].imshow(mask_, alpha=0.6, cmap="autumn")
    #     axis[1].set_title("Pred")

    #     self.logger.experiment.add_figure(f"{name} Prediction vs Label", fig, self.global_step)

    # def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    #     """Save checkpoint if a new best is achieved"""
    #     if is_best:
    #         print("=> Saving a new best")
    #         torch.save(state, filename)  # save checkpoint
    #     else:
    #         print("=> Validation Accuracy did not improve")
    
    # def configure_optimizers(self):
    #     #Caution! You always need to return a list here (just pack your optimizer into one :))
    #     return [self.optimizer] 