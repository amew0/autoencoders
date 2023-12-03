import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from typing import Union, Optional
import yaml

"""
TASK: DATA LOADING
Available Classes:
- DiffImg
"""
"""
TASK: IMAGE RECONSTRUCTION
Available Classes:
- AutoencoderEIT_config
    - where ./models/0014.pt resides 
- AutoencoderEIT
    Build model from THAT LR 216
- Autoencoder_linear
    Build all linear model
- Reconstructor
"""
"""
TASK: V2LR
Available Classes:
- V2ImgLR
"""
"""
TASK: INVERSE PROBLEM
Available Classes:
- AutoencoderEIT_v
    LR is 216
- VoltageAE
    Big LR and Residual
- VoltageAE_base
    Big LR and Residual (from scratch)
- vAE
    (not recommended)
    v16toi24 + (0014.pt)
"""

"""
LAYER exp
"""
def ksp(input_size, output_size):
    possible_params = []
    
    for k in range(1, input_size):
        for s in range(1, input_size):
            for p in range(0, input_size):
                if output_size == (input_size - k + 2*p)//s + 1:
                    possible_params.append((k, s, p))
    return possible_params

def kspo(input_size, output_size):
    possible_params = []
    for k in range(1, output_size//2+1):
        for s in range(1, output_size//2+1):
            for p in range(0, output_size//2+1):
                for op in range(0, output_size//2+1):
                    if output_size == (input_size - 1) * s + k - 2*p + op :
                        possible_params.append((k, s, p, op))
    return possible_params

def optimizer_build(optimizer_config,model):
    if 'Adam' in optimizer_config:
        adam = optimizer_config['Adam']
        return torch.optim.Adam(model.parameters(),
            lr=adam['learning_rate'],
            weight_decay=adam['weight_decay'])

"""
DATA LOADING
"""

class DiffImg(Dataset):
    def __init__(
            self, 
            csv_file: str, 
            diff_transform: Optional[Union[None, transforms.Compose]] = None,  
            img_transform: Optional[Union[None, transforms.Compose]] = None
        ) -> None:
        self.data = pd.read_csv(csv_file,header=None)
        
        self.diff_transform = diff_transform
        self.img_transform = img_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        diff_image_data = np.array(self.data.iloc[idx].values,dtype=np.float32)
        
        diff = diff_image_data[:256]
        image = diff_image_data[256:]
        
        diff = diff.reshape(16,16)
        image = image.reshape(24,24)
        
        # diff = torch.from_numpy(diff)
        # image = torch.from_numpy(image)

        # PUT all transformations here
        if self.diff_transform:
            diff = self.diff_transform(diff)
        
        if self.img_transform:
            image = self.img_transform(image)

        return (diff ,image)

"""
IMAGE RECONSTRUCTION
"""
class AutoencoderEIT_config(nn.Module):
    def __init__(self, config:Union[dict,None]=None):
        super().__init__()
        if config:
            self.encoder = self.encoder_build(config['encoder'])
            self.decoder = self.decoder_build(config['decoder'])
        else:
            print("No config provided, using default")

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded
    
    def encoder_build(self,block_config):
        layers = nn.Sequential()
        for config in block_config:
            if 'layer' in config:
                attrs = config['layer']
                layers.append(
                    nn.Conv2d(attrs[0], attrs[1], attrs[2],
                            stride=attrs[3], padding=attrs[4]))
            elif 'act' in config:
                if config['act'] == 'ReLU':
                    layers.append(nn.ReLU())
                else:
                    print("'act' found but is not ReLU")
        return layers
    
    def decoder_build(block_config):
        layers = nn.Sequential()
        for config in block_config:
            if 'layer' in config:
                attrs = config['layer']
                layers.append(
                    nn.ConvTranspose2d(attrs[0], attrs[1], attrs[2],
                            stride=attrs[3], padding=attrs[4],
                            output_padding=attrs[5]))
            elif 'act' in config:
                if config['act'] == 'ReLU':
                    layers.append(nn.ReLU())
                elif config['act'] == 'Tanh':
                    layers.append(nn.Tanh())
                else:
                    print("'act' found but is not in ['ReLU','Tanh']")
        return layers

class AutoencoderEIT(nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.load("./models/0014.pt")
        # make all prev model params untrainable
        for param in model.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(
            *model.encoder,
            nn.Conv2d(192, 24, 3, 2, 1), # N,24,3,3
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(216,216),
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (24, 3, 3)),
            nn.ConvTranspose2d(24, 192, 3, 2, 1, 1), # N,192,6,6
            nn.ReLU(),
            *model.decoder[:-1] # removed tanh
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

class AutoencoderEIT142(nn.Module):
    def __init__(self):
        super().__init__() 
        model = torch.load("./models/img/14.2.20231110000321.pt")
        # make all prev model params untrainable
        for param in model.parameters():
            param.requires_grad = True
        self.encoder = nn.Sequential(
            *model.encoder[1:],
        )
        self.decoder = nn.Sequential(
            *model.decoder[:2],
            nn.ReLU(),
            *model.decoder[2:-1]
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

class Reconstructor(nn.Module): # ibrar's
    def __init__(self):
        super().__init__()
        # from ibrar
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 576),
            nn.Unflatten(1, (1, 24, 24)),
            
            nn.Conv2d(1,32,3,2,1), # 24x24 -> 12x12
            nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), # 12x12 -> 6x6
            nn.ReLU(),
            nn.Conv2d(64,128,3,2,1), # 6x6 -> 3x3
            nn.ReLU(),
            nn.Conv2d(128,256,2,2,0), # 3x3 -> 1x1
            nn.ReLU(),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256,1,1)), # 1x1 -> 256x1x1
            nn.ConvTranspose2d(256,128,2,2,0,1), # 1x1 -> 3x3
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,3,2,1,1), # 3x3 -> 6x6
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,2,1,1), # 6x6 -> 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(32,1,3,2,1,1) # 12x12 -> 24x24
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded
    
class Autoencoder_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 576),
            nn.Tanh()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

"""
V2LR
"""
class V2ImgLR (nn.Module):
    def __init__(self):
        super().__init__()
        # model = torch.load("./models/v2lr/2_1.2.1.1.20231117023937_v2lr.pt")
        # for param in model.parameters():
        #     param.requires_grad = False
        self.v2lr = None
        # 1.3 with hidden linear
        # self.v2lr = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 216)
        # )

    def forward(self,x):
        mapped = self.v2lr(x)
        return mapped

"""
INVERSE PROBLEM
"""
class VoltageReconsturctor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_v = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.decoder_v = nn.Sequential(
            nn.Unflatten(1, (24, 3, 3)),
            nn.ConvTranspose2d(24, 16, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, 2, 1, 1)
        )
    def forward(self,x):
        encoded_v = self.encoder_v(x)
        decoded_v = self.decoder_v(encoded_v)
        return encoded_v, decoded_v

class Diff2Image (nn.Module):
    def __init__(self,reconstructor_v:VoltageReconsturctor, reconstructor:AutoencoderEIT):
        super().__init__()
        # pretrained ones (frozen outside)
        self.reconstructor_v = reconstructor_v
        self.reconstructor = reconstructor

        self.encoder_v = self.reconstructor_v.encoder_v # put the model here
        self.vlr2ilr = nn.Sequential(
            nn.Linear(216,108),
            nn.ReLU(),
            nn.Linear(108,216)
        )
        self.decoder = self.reconstructor.decoder
    def forward(self,x):
        encoded_v = self.encoder_v(x)
        mapped = self.vlr2ilr(encoded_v)
        decoded = self.decoder(mapped)
        return encoded_v, mapped, decoded

class AutoencoderEIT_v (nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.load("./models/img/14.1.20231109121105.pt")
        # make all model params untrainable
        for param in model.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), # 32 16
            nn.ReLU(),
            nn.Conv2d(32,48,5,1,0), # 48 12
            nn.ReLU(),
            nn.Conv2d(48,96,2,2,2), # 96 8
            nn.ReLU(),
            nn.Conv2d(96,192,3,1,0), # 192 6 
            nn.ReLU(),
            nn.Conv2d(192, 24, 3, 2, 1), # N,12,3,3
            nn.ReLU(),
            nn.Flatten()
        ) 
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (24, 3, 3)),
            nn.ConvTranspose2d(12, 192, 3, 2, 1, 1), # N,192,6,6
            *model.decoder
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded
    
class MinMax(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x_:torch.Tensor):
        min_vals, _ = x_.min(dim=2, keepdim=True)
        max_vals, _ = x_.max(dim=2, keepdim=True)
        normalized = (x_ - min_vals) / (max_vals - min_vals)
        return normalized

class VoltageAE(nn.Module): # from 0014.pt
    def __init__(self):
        super().__init__()
        model_AE = torch.load("models/0014.pt")
        for param in model_AE.parameters():
            param.requires_grad = False
        self.encoder_v = nn.Sequential(
            ResidualBlock(1, 32, 3, 1, 1), # 32 16
            ResidualBlock(32, 48, 5, 1, 0), # 48 12
            *model_AE.encoder[2:]
        )
        self.decoder = nn.Sequential(
            *model_AE.decoder[:-1],
            MinMax()
        )
    def forward(self, x):
        encoded_v = self.encoder_v(x)
        decoded = self.decoder(encoded_v)
        return encoded_v,decoded
    
class ResidualBlock(nn.Module): # reset model
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU()
        self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)  # Add a 1x1 convolution for the skip connection
    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        skip = self.skip(x)
        out += skip[:, :, :out.size(2), :out.size(3)]
        out = self.relu(out)
        return out
    
class VoltageAE_base(nn.Module): # from scratch
    def __init__(self):
        super().__init__()
        self.encoder_v = nn.Sequential(
            ResidualBlock(1, 32, 3, 1, 1), # 32 16
            ResidualBlock(32, 48, 5, 1, 0), # 48 12
            ResidualBlock(48, 96, 2, 2, 2), # 96 8
            ResidualBlock(96, 192, 3, 1, 0) # 192 6 
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=(2, 2), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            MinMax()
        )
    def forward(self, x):
        encoded_v = self.encoder_v(x)
        decoded = self.decoder(encoded_v)
        return encoded_v,decoded

class vAE(nn.Module): # not recommended
    def __init__(self):
        super().__init__()
        self.v16toi24 = nn.Sequential(
            nn.ConvTranspose2d(1, 4, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 8, kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,kernel_size=1),
            nn.ReLU(),
        )
        model = torch.load("./models/0014.pt")
        # make all model params untrainable
        for param in model.parameters():
            param.requires_grad = False
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, x):
        transformed = self.v16toi24(x)
        encoded = self.encoder(transformed)
        decoded = self.decoder(encoded)
        return transformed,encoded,decoded
