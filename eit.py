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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import time
from datetime import datetime
import sys
from utils.classes import *

print("Importing finished!!")
start = time.time()
seed = 64
batch_size = 16
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is going to be used!!")

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

nownow = datetime.now()
ID = sys.argv[1]

TASK = "img"
CONDUCTANCE_VALUES = "2_"
id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"


# DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
# DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"

DIFFS_IMGS_TRAIN_PATH = "./data/eit2/diffs_imgs_train_2.csv"
DIFFS_IMGS_TEST_PATH = "./data/eit2/diffs_imgs_test_2.csv"

# DIFFS_IMGS_TRAIN_PATH = "./data/eit4/diffs_imgs_train_4.csv"
# DIFFS_IMGS_TEST_PATH = "./data/eit4/diffs_imgs_test_4.csv"

# DIFFS_IMGS_TRAIN_PATH = "./data/eit6/diffs_imgs_train_6.csv"
# DIFFS_IMGS_TEST_PATH = "./data/eit6/diffs_imgs_test_6.csv"

LOSS_TRACKER_PATH = f'./results/loss_tracker_{TASK}.csv'
MODEL_STATEDICT_SAVE_PATH = f"./models/{TASK}/{id_config}_{TASK}.pth"
MODEL_SAVE_PATH = MODEL_STATEDICT_SAVE_PATH[:-1] # pt instead of pth

diff_transform = transforms.Compose([
    transforms.ToTensor(),
])

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = DiffImg(csv_file=DIFFS_IMGS_TRAIN_PATH, diff_transform=diff_transform, img_transform=img_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = DiffImg(csv_file=DIFFS_IMGS_TEST_PATH, diff_transform=diff_transform, img_transform=img_transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Dataset loaded!! Length (train dataset) - {len(train_dataset)}")

recon = AutoencoderEIT().to(device)
criterion = nn.MSELoss()
optimizer_config = {'Adam': {'learning_rate':1e-3,'weight_decay':1e-5}}
optimizer = optimizer_build(optimizer_config,recon)

print(summary(recon, (1,24,24)))

print("Training started!! - Reconstruction")
train_losses = []
for epoch in range(epochs):
    for i, (_, batch_img) in enumerate(train_dataloader):
        # _ is batch_diff 
        batch_img = batch_img.to(device)
        _,batch_recon = recon(batch_img)
        loss = criterion(batch_recon, batch_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0 or (epoch == epochs - 1):
        print(f'Epoch:{epoch}, Loss:{loss.item():.6f}')
    train_losses.append(loss.item())

recon.eval()
test_losses = [] 
with torch.no_grad():
    for i, (_, batch_img) in enumerate(train_dataloader):
        # _ is batch_diff 
        batch_img = batch_img.to(device)
        _,batch_recon = recon(batch_img)
        loss = criterion(batch_recon, batch_img)
        test_losses.append(loss.item())

avg_test_loss = sum(test_losses) / len(test_losses)

print(f"Avg Test Loss: {avg_test_loss}")
print(f"Max Test Loss: {max(test_losses)}")
print(f"Min Test Loss: {min(test_losses)}")
train_losses.append(avg_test_loss)

# try:
loss_tracker = pd.read_csv(LOSS_TRACKER_PATH,header=None,index_col=0)
# except:
    # print("File not found, creating new file")
    # loss_tracker = pd.DataFrame()
loss_tracker.loc[id_config] = train_losses
loss_tracker.to_csv(LOSS_TRACKER_PATH,header=None)
print(f"written to: {LOSS_TRACKER_PATH}")
torch.save(recon.state_dict(), MODEL_STATEDICT_SAVE_PATH)
print(f"written to: {MODEL_STATEDICT_SAVE_PATH}")
torch.save(recon, MODEL_SAVE_PATH) # this saves the model as-is
print(f"written to: {MODEL_SAVE_PATH}")

print(f"Elapsed time: {time.time() - start} seconds.")