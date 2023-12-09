'''
This script is only for image reconstruction

Note the following:
Reconstructor (at least the first architecture) will be from Ibrar's presentation
21 (input with minmaxscaler and tanh) (final layer tanh input )
    21.1 (input with tanh final with tanh)

Autoencoder (will be modified Autoencoder_config (where 0014.pt is))
14.1 (first mod to 0014.pt LR to 108)

Autoencoder complete linear
'''
import sys
import time
import csv
from datetime import datetime
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torchmetrics.image import StructuralSimilarityIndexMeasure

from utils.classes import *
print("Importing finished!!")

start = time.time()
seed,batch_size,epochs = 64,4,150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is going to be used!!")

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ID = sys.argv[1]
TASK = "img"
CONDUCTANCE_VALUES = ""
DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"
recon_path = "./models/img/14.2.1.retraining.2.20231130014311_img.pt"

diff_transform = transforms.Compose([transforms.ToTensor()])
img_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = DiffImg(csv_file=DIFFS_IMGS_TRAIN_PATH, diff_transform=diff_transform, img_transform=img_transform)
test_dataset = DiffImg(csv_file=DIFFS_IMGS_TEST_PATH, diff_transform=diff_transform, img_transform=img_transform)

generator = torch.Generator().manual_seed(seed)
train_size = int(0.8 * len(train_dataset)) 
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],generator=generator)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Dataset loaded!! Length (train dataset) - {len(train_dataset)}")

def play(job="Training",dataloader:DataLoader=train_dataloader,recon:nn.Module=None,mse=None,ssim=None,optimizer=None,scale_to_input = False):
    recon.train() if job=="Training" else recon.eval()
    epoch_loss = 0.0
    for i, (_, batch_img) in enumerate(dataloader):
        batch_img = batch_img.to(device)
        _,batch_decoded = recon(batch_img)
        
        if scale_to_input:
            min_value = batch_decoded.min()
            max_value = batch_decoded.max()
            batch_decoded = (batch_decoded - min_value) / (max_value - min_value) \
                * (batch_img.max() - batch_img.min()) + batch_img.min()
            
        mse_loss = F.mse_loss(batch_img, batch_decoded).requires_grad_()
        ssim_value = 1 - ssim(batch_img, batch_decoded).requires_grad_()

        loss = alpha*mse_loss + beta*ssim_value

        if job == "Training": # batch
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        epoch_loss += loss.item()


    epoch_loss /= len(dataloader)
    if job == "Training": # epoch

        optimizer.zero_grad()
        epoch_loss = torch.tensor(epoch_loss,device=device,requires_grad=True)
        optimizer.step()

    return epoch_loss, mse_loss, ssim_value, loss

def ksp_helper (w):
    if w == 4: return (3,1,0),(3,1,0,0)
    elif w == 3: return (3,2,1),(3,2,1,1)
    elif w == 2: return (3,2,0),(3,2,0,1)
    else: raise ValueError("w should be 2,3,4")

# configs = [
#     (12,4,*ksp_helper(4)),#192
#     (18,3,*ksp_helper(3)),#162
#     (16,3,*ksp_helper(3)),#144
#     (8,4,*ksp_helper(4)),#128
#     (6,4,*ksp_helper(4)),#96
#     (8,3,*ksp_helper(3)),#72
#     (5,3,*ksp_helper(3)),#45
#     (6,2,*ksp_helper(2)),#24
# ]
configs=[216]
for i,config in enumerate(configs):
    print(f"Config {config}")
    nownow = datetime.now()
    id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"
    LOSS_TRACKER_PATH = f'./results/loss_tracker_{TASK}.csv'
    MODEL_STATEDICT_SAVE_PATH = f"./models/{TASK}/{id_config}_{TASK}.pth"
    MODEL_SAVE_PATH = MODEL_STATEDICT_SAVE_PATH[:-1] # pt instead of pth

    # f,w,ksp_,kspo_ = config
    recon = torch.load(recon_path)
    # recon.encoder = nn.Sequential(
    #     *recon.encoder[:6],
    #     nn.Conv2d(192, f, *ksp_),
    #     nn.BatchNorm2d(f),
    #     nn.ReLU(),
    #     nn.Flatten()
    # )
    # recon.decoder = nn.Sequential(
    #     nn.Unflatten(1, (f, w, w)),
    #     nn.ConvTranspose2d(f, 192, *kspo_),
    #     nn.BatchNorm2d(192),
    #     nn.ReLU(),
    #     *recon.decoder[2:]
    # )
    recon = recon.to(device)
    print(recon)

    mse = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)
    
    optimizer_config = {'Adam': {'learning_rate':1e-3,'weight_decay':1e-5}}
    optimizer = optimizer_build(optimizer_config,recon)

    best_recon = deepcopy(recon)
    min_loss = np.inf
    best_epoch = -1

    alpha = 0.5  # Weight for MSE Loss
    beta = 0.05 # Weight for SSIM

    train_losses = []
    for epoch in range(epochs):
        epoch_loss, mse_loss, ssim_value, loss = play("Training",train_dataloader,recon,mse,ssim,optimizer)
        train_losses.append(loss.item())
        
        with torch.no_grad():
            val_epoch_loss,val_mse_loss,_,_=play("Validation",val_dataloader,recon,mse,ssim)

        if (epoch % 10 == 0) or (epoch == epochs - 1):
            print(f'Task: Reconstruction, Epoch:{epoch}, Epoch Loss: {epoch_loss}, Loss:{loss.item():.6f}, MSELoss:{mse_loss:.6f}, SSIM: {ssim_value}\
                \tVal Epoch Loss: {val_epoch_loss:.6f},Val MSE: {val_mse_loss:.6f}')

        if val_epoch_loss < min_loss:
            min_loss = val_epoch_loss
            best_recon = deepcopy(recon)
            best_epoch = epoch
            print(f"Best state dict yet, with Val Epoch MSE: {min_loss:.6f} Val MSE {mse_loss:.6f}, found @ {epoch}...")
        
    recon = deepcopy(best_recon)
    with torch.no_grad():
        test_epoch_loss,test_mse_loss,_,_=play("Testing",test_dataloader,recon,mse,ssim)
        
        train_losses.append(test_epoch_loss)
        print(f"Avg Epoch Test Loss: {test_epoch_loss}, Last MSE Test Loss: {test_mse_loss}")

    torch.save(recon.state_dict(), MODEL_STATEDICT_SAVE_PATH)
    print(f"written to: {MODEL_STATEDICT_SAVE_PATH}")
    torch.save(recon, MODEL_SAVE_PATH) # this saves the model as-is
    print(f"written to: {MODEL_SAVE_PATH}")

    with open(LOSS_TRACKER_PATH, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(train_losses)

    print(f"written to: {LOSS_TRACKER_PATH}")
    print(f"Elapsed time: {time.time() - start} seconds.")