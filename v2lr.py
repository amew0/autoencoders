import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
import time
from datetime import datetime
from utils.classes import *

print("Importing finished!!")
start = time.time()
seed = 64
batch_size = 4
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is going to be used!!")

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

nownow = datetime.now()
ID = sys.argv[1]
TASK = "v2lr"
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

# recon_path = "./models/img/14.2.1.20231116190651_img.pt"
recon_path = "./models/img/2_14.2.1.20231116193059_img.pt" #2
# recon_path = "./models/img/4_14.2.1.20231116190154_img.pt" #4
# recon_path = "./models/img/6_14.2.1.20231116190136_img.pt" #6

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

v2lr = V2ImgLR().to(device)
criterion = nn.MSELoss()
optimizer_config = {'Adam': {'learning_rate':1e-3,'weight_decay':1e-5}}
optimizer = optimizer_build(optimizer_config,v2lr)
# l1_lambda = 0.001

print(summary(v2lr,(1,16,16)))
recon = torch.load(recon_path)

train_losses = []
for epoch in range(epochs):
    for i, (batch_diff, batch_img) in enumerate(train_dataloader):
        batch_diff = batch_diff.to(device)
        batch_img = batch_img.to(device)
        batch_mapped = v2lr(batch_diff)

        batch_encoded = recon.encoder(batch_img)
        batch_recon_v = recon.decoder(batch_mapped)
        loss = criterion(batch_mapped, batch_encoded)

        # # Add L1 regularization term to the loss
        # l1_reg = torch.tensor(0., requires_grad=True).to(device)
        # for param in v2lr.parameters():
        #     l1_reg = l1_reg + torch.norm(param, 1)

        # loss = loss + l1_lambda * l1_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch % (10) == 0) or (epoch == epochs - 1):
        recon_v_loss = criterion(batch_img, batch_recon_v)

        print(f'Task: v2lr, Epoch:{epoch+1}, Loss:{loss.item():.6f}\
              \t\tTask: InvProb, Epoch:{epoch+1}, Loss:{recon_v_loss.item():.6f}')
    train_losses.append(loss.item())

v2lr.eval()
test_losses = [] 
with torch.no_grad():
    for i, (batch_diff, batch_img) in enumerate(test_dataloader):
        batch_diff = batch_diff.to(device)
        batch_img = batch_img.to(device)
        batch_mapped = v2lr(batch_diff)

        batch_encoded = recon.encoder(batch_img)
        loss = criterion(batch_mapped, batch_encoded)

        test_losses.append(loss.item())

avg_test_loss = sum(test_losses) / len(test_losses)
print(f"Avg Test Loss: {avg_test_loss}")
print(f"Max Test Loss: {max(test_losses)}")
print(f"Min Test Loss: {min(test_losses)}")

train_losses.append(avg_test_loss)

torch.save(v2lr.state_dict(), MODEL_STATEDICT_SAVE_PATH)
print(f"written to: {MODEL_STATEDICT_SAVE_PATH}")
torch.save(v2lr, MODEL_SAVE_PATH) # this saves the model as-is
print(f"written to: {MODEL_SAVE_PATH}")

loss_tracker = pd.read_csv(LOSS_TRACKER_PATH,header=None,index_col=0)
loss_tracker.loc[id_config] = train_losses
loss_tracker.to_csv(LOSS_TRACKER_PATH,header=None)
print(f"written to: {LOSS_TRACKER_PATH}")

print(f"Elapsed time: {time.time() - start} seconds.")