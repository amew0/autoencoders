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
seed,batch_size,epochs = 64,4,200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is going to be used!!")

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ID = sys.argv[1]
TASK = "v2lr"
CONDUCTANCE_VALUES = ""
DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"
recon_path = "./models/img/14.2.1.retraining.2.20231130014311_img.pt" # "./models/img/14.2.1.20231116190651_img.pt" #"./models/img/14.2.20231110000321.pt"

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

def play(job="Training",dataloader:DataLoader=train_dataloader,v2lr:nn.Module=None,mse=None,ssim=None,optimizer=None,scale_to_input = False):
    v2lr.train() if job=="Training" else v2lr.eval()
    epoch_mse_loss = 0.0
    epoch_ssim_value = 0.0
    epoch_loss = 0.0
    for i, (batch_diff, batch_img) in enumerate(dataloader):
        batch_diff = batch_diff.to(device)
        batch_img = batch_img.to(device)
        batch_mapped, batch_lr, batch_recon_v = v2lr(batch_diff,batch_img)

        if scale_to_input:
            min_value = batch_recon_v.min()
            max_value = batch_recon_v.max()

            batch_recon_v = (batch_recon_v - min_value) / (1e-8+max_value - min_value) \
                * (batch_img.max() - batch_img.min()) + batch_img.min()
        
        mse_loss = F.mse_loss(batch_img, batch_recon_v).requires_grad_()
        ssim_value = 1 - ssim(batch_img, batch_recon_v).requires_grad_()

        loss = alpha*mse_loss + beta*ssim_value
        mse_lr = mse(batch_lr, batch_mapped)

        if job == "Training": # batch
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        epoch_mse_loss += mse_loss.item()
        epoch_ssim_value += ssim_value.item()
        epoch_loss += loss.item()

    epoch_mse_loss /= len(dataloader)
    epoch_ssim_value /= len(dataloader)
    epoch_loss /= len(dataloader)

    if job == "Training": # epoch

        optimizer.zero_grad()
        epoch_loss = torch.tensor(epoch_loss,device=device,requires_grad=True)
        optimizer.step()

    return epoch_loss, mse_lr, mse_loss, ssim_value, loss, epoch_mse_loss, epoch_ssim_value

alphas = [i/100 for i in range(9,39)][::2]

for i,alpha in enumerate(alphas):
    print(alpha)
    print(f"Config {alpha}")
    nownow = datetime.now()
    id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"
    LOSS_TRACKER_PATH = f'./results/loss_tracker_{TASK}.csv'
    MODEL_STATEDICT_SAVE_PATH = f"./models/{TASK}/{id_config}_{TASK}.pth"
    MODEL_SAVE_PATH = MODEL_STATEDICT_SAVE_PATH[:-1] # pt instead of pth

    v2lr = V2ImgLR(recon_path)
    v2lr = v2lr.to(device)
    print(v2lr.v2lr)
    # summary(v2lr,(1,16,16),(1,24,24))
    
    mse = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)
    
    optimizer_config = {'Adam': {'learning_rate':1e-3,'weight_decay':1e-5}}
    optimizer = optimizer_build(optimizer_config,v2lr)

    best_v2lr = deepcopy(v2lr)
    min_loss = np.inf
    best_epoch = -1

    alpha = alpha  # Weight for MSE Loss
    beta = 0.1*(1-alpha) # Weight for SSIM

    train_losses = []
    for epoch in range(epochs):
        epoch_loss,mse_lr, mse_loss, ssim_value, loss, epoch_mse, epoch_ssim = play("Training",train_dataloader,v2lr,mse,ssim,optimizer)
        train_losses.append(loss.item())
        
        with torch.no_grad():
            val_epoch_loss,_,val_mse_loss,_,_,val_epoch_mse_loss, val_epoch_ssim =play("Validation",val_dataloader,v2lr,mse,ssim)

        if (epoch % 10 == 0) or (epoch == epochs - 1):
            print(f'Task: InvProb, Epoch:{epoch}, Epoch Loss: {epoch_loss}, Epoch MSE: {epoch_mse}, Epoch SSIM: {epoch_ssim}, \tLoss:{loss.item():.6f}, MSELoss:{mse_loss:.6f}, SSIM: {ssim_value}\
                    \tVal Epoch Loss: {val_epoch_loss:.6f},Val MSE: {val_mse_loss:.6f}\
                    \tTask: V2LR, MSELoss:{mse_lr.item():.6f}')

        if val_epoch_loss < min_loss:
            min_loss = val_epoch_loss
            best_v2lr = deepcopy(v2lr)
            best_epoch = epoch
            print(f"Best state dict yet, with Val Epoch MSE: {min_loss:.6f} Val MSE {mse_loss:.6f}, found @ {epoch}...")
        
    v2lr = deepcopy(best_v2lr)
    with torch.no_grad():
        test_epoch_loss,_,test_mse_loss,_,_,test_epoch_mse, test_epoch_ssim=play("Testing",test_dataloader,v2lr,mse,ssim)
        
        train_losses.append(test_epoch_loss)
        print(f"Avg Test Loss: {test_epoch_loss} Avg Test MSE: {test_epoch_mse} Avg Test SSIM {test_epoch_ssim}, Last MSE Test Loss: {test_mse_loss}")

    torch.save(v2lr.state_dict(), MODEL_STATEDICT_SAVE_PATH)
    print(f"written to: {MODEL_STATEDICT_SAVE_PATH}")
    torch.save(v2lr, MODEL_SAVE_PATH) # this saves the model as-is
    print(f"written to: {MODEL_SAVE_PATH}")

    with open(LOSS_TRACKER_PATH, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(train_losses)

    print(f"written to: {LOSS_TRACKER_PATH}")
    print(f"Elapsed time: {time.time() - start} seconds.")