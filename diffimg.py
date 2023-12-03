import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import sys
import time
from datetime import datetime
from utils.classes import *

print("Importing finished!!")
start = time.time()
seed = 64
batch_size = 4
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is going to be used!!")

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

nownow = datetime.now()
ID = sys.argv[1]
TASK = "diffimg"
CONDUCTANCE_VALUES = "2_"
id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"

LOSS_TRACKER_PATH = f'./results/loss_tracker_{TASK}.csv'
MODEL_STATEDICT_SAVE_PATH = f"./models/{TASK}/{id_config}_{TASK}.pth"
MODEL_SAVE_PATH = MODEL_STATEDICT_SAVE_PATH[:-1] # pt instead of pth

reconstructor_v_path = "./models/diffimg/VR_2_0.20231117042725_diffimg.pt"
recon_path = "./models/img/2_14.2.1.20231116193059_img.pt"

DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"

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

# diffrecon = VoltageReconsturctor().to(device)
# diff2img = Reconstructor().to(device)

reconstructor_v = torch.load(reconstructor_v_path)
for param in reconstructor_v.parameters():
    param.requires_grad = False

reconstructor = torch.load(recon_path)
for param in reconstructor.parameters():
    param.requires_grad = False

diff2img = Diff2Image(reconstructor_v,reconstructor).to(device)

criterion = nn.MSELoss()
optimizer_config = {'Adam': {'learning_rate':1e-3,'weight_decay':1e-5}}
optimizer = optimizer_build(optimizer_config,diff2img)
# l1_lambda = 0.001

print(summary(diff2img,(1,16,16)))

train_losses = []
for epoch in range(epochs):
    for i, (batch_diff, batch_img) in enumerate(train_dataloader):
        batch_diff = batch_diff.to(device)
        batch_img = batch_img.to(device)
        _,batch_mapped,batch_recon = diff2img(batch_diff)

        loss = criterion(batch_recon, batch_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch % (10) == 0) or (epoch == epochs - 1):
        mapping_loss = criterion(diff2img.reconstructor.encoder(batch_img),batch_mapped)
        print(f'Task: VR_{TASK}, Epoch:{epoch+1}, Loss (Total):{loss.item():.6f}\
              \tLoss (Mapping): {mapping_loss.item():.6f}')
    train_losses.append(loss.item())

diff2img.eval()
test_losses = [] 
with torch.no_grad():
    for i, (batch_diff, batch_img) in enumerate(test_dataloader):
        batch_diff = batch_diff.to(device)
        batch_img = batch_img.to(device)
        _,batch_mapped,batch_recon = diff2img(batch_diff)

        loss = criterion(batch_recon, batch_img)

        test_losses.append(loss.item())

avg_test_loss = sum(test_losses) / len(test_losses)
print(f"Avg Test Loss: {avg_test_loss}")
print(f"Max Test Loss: {max(test_losses)}")
print(f"Min Test Loss: {min(test_losses)}")

train_losses.append(avg_test_loss)

torch.save(diffrecon.state_dict(), MODEL_STATEDICT_SAVE_PATH)
print(f"written to: {MODEL_STATEDICT_SAVE_PATH}")
torch.save(diffrecon, MODEL_SAVE_PATH) # this saves the model as-is
print(f"written to: {MODEL_SAVE_PATH}")

loss_tracker = pd.read_csv(LOSS_TRACKER_PATH,header=None,index_col=0)
loss_tracker.loc[id_config] = train_losses
loss_tracker.to_csv(LOSS_TRACKER_PATH,header=None)
print(f"written to: {LOSS_TRACKER_PATH}")

print(f"Elapsed time: {time.time() - start} seconds.")