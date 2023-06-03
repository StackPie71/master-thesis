import torch

class DoubleConv(torch.nn.Module):
    """
    Helper Class which implements the intermediate Convolutions
    """
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())
        
    def forward(self, X):
        return self.step(X)

    
    

class UNet(torch.nn.Module):
    """
    This class implements a UNet for the Segmentation
    We use 3 down- and 3 UpConvolutions and two Convolutions in each step
    """

    def __init__(self):
        """Sets up the U-Net Structure
        """
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # CT ------------------------------------------------
        ############# DOWN #####################
        self.layer1_ct = DoubleConv(1, 32)
        self.layer2_ct = DoubleConv(32, 64)
        self.layer3_ct = DoubleConv(64, 128)
        #########################################
        self.layer4 = DoubleConv(128, 256)
        ############## UP #######################
        self.layer5_ct = DoubleConv(256 + 128, 128)
        self.layer6_ct = DoubleConv(128 + 64, 64)
        self.layer7_ct = DoubleConv(64 + 32, 32)
        self.layer8_ct = torch.nn.Conv3d(32, 2, 1)  # Output: 2 values -> background, GTV
        #########################################

        self.maxpool = torch.nn.MaxPool3d(2)

    def forward(self, x_ct):
        
        ####### DownConv 1#########
        x1_ct = self.layer1_ct(x_ct)
        x1m_ct = self.maxpool(x1_ct)
        ###########################
        
        ####### DownConv 2#########        
        x2_ct = self.layer2_ct(x1m_ct)
        x2m_ct = self.maxpool(x2_ct)
        ###########################

        ####### DownConv 3#########        
        x3_ct = self.layer3_ct(x2m_ct)
        x3m_ct = self.maxpool(x3_ct)
        ###########################
        
        ##### Intermediate Layer ## 
        x4 = self.layer4(x3m_ct)
        ###########################

        ####### UpCONV 1#########        
        x5_ct = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x4)  # Upsample with a factor of 2
        x5_ct = torch.cat([x5_ct, x3_ct], dim=1)  # Skip-Connection
        x5_ct = self.layer5_ct(x5_ct)
        ###########################

        ####### UpCONV 2#########        
        x6_ct = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x5_ct)        
        x6_ct = torch.cat([x6_ct, x2_ct], dim=1)  # Skip-Connection    
        x6_ct = self.layer6_ct(x6_ct)
        ###########################
        
        ####### UpCONV 3#########        
        x7_ct = torch.nn.Upsample(scale_factor=2, mode="trilinear")(x6_ct)
        x7_ct = torch.cat([x7_ct, x1_ct], dim=1)       
        x7_ct = self.layer7_ct(x7_ct)
        ###########################
        
        ####### Predicted segmentation#########        
        ret_ct = self.layer8_ct(x7_ct)
        
        return ret_ct