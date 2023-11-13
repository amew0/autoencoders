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
from sklearn.preprocessing import MinMaxScaler

print("Importing finished!!")
start = time.time()
seed = 64
batch_size = 8
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is going to be used!!")

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

nownow = datetime.now()
ID = sys.argv[1]
id_config = f"{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"

# DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
# DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"
LOSS_TRACKER_PATH = './results/loss_tracker_diffimg.csv'
MODEL_STATEDICT_SAVE_PATH = f"./models/img/{id_config}.pth"
MODEL_SAVE_PATH = f"./models/img/{id_config}.pt"

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.array(self.data.iloc[idx].values,dtype=np.float32)
        # image = image.reshape(24, 24)
        image = np.tanh(image)  # Reshape to 24x24 and convert to PIL Image
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image

# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
transform = None
TRAIN_PATH = "./data/eit/imgs_train.csv"
TEST_PATH = "./data/eit/imgs_test.csv"
train_dataset = CustomDataset(csv_file=TRAIN_PATH, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = CustomDataset(csv_file=TEST_PATH, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Dataset loaded!!")
    
class Reconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        # from ibrar
        self.encoder = nn.Sequential(
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
            nn.ConvTranspose2d(32,1,3,2,1,1), # 12x12 -> 24x24
            nn.Tanh(),
        )
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded  # encoder is returned to check latent representaion
class AutoencoderEIT_config(nn.Module):
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded
class AutoencoderEIT(nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.load("./models/0014.pt")
        # make all model params untrainable
        for param in model.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, 24, 24)),
            *model.encoder,
            nn.Conv2d(192, 24, 3, 2, 1), # N,24,3,3
            nn.ReLU(),
            nn.Flatten()
        ) 
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (24, 3, 3)),
            nn.ConvTranspose2d(24, 192, 3, 2, 1, 1), # N,192,6,6
            *model.decoder,
            nn.Flatten()
            )
    def forward(self, x):
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

def loss_build(loss_config):
    # if loss_config == 'MSELoss':
        return nn.MSELoss()

def optimizer_build(optimizer_config,model):

    if 'Adam' in optimizer_config:
        adam = optimizer_config['Adam']
        return torch.optim.Adam(model.parameters(),
                            lr=adam['learning_rate'],
                            weight_decay=adam['weight_decay'])

model_v = AutoencoderEIT().to(device)
criterion = loss_build('MSELoss')
optimizer_config = {'Adam': {'learning_rate':1e-3,'weight_decay':1e-5}}
optimizer = optimizer_build(optimizer_config,model_v)

print(summary(model_v, (576,)))

print("Training started!! - Reconstruction")
train_losses = []
for epoch in range(epochs):
    for i, batch_img in enumerate(train_dataloader):
        batch_img = batch_img.to(device)
        _,batch_recon = model_v(batch_img)
        loss = criterion(batch_recon, batch_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % (epochs // 10) == 0:
        print(f'Epoch:{epoch}, Loss:{loss.item():.6f}')
    train_losses.append(loss.item())
    # outputs.append((epoch, batch_img, batch_recon))
print(f'Last Epoch:{epoch}, Loss:{loss.item():.6f}')

model_v.eval()
test_losses = [] 
with torch.no_grad():
    for i, batch_img in enumerate(test_dataloader):
        batch_img = batch_img.to(device)
        _,batch_recon = model_v(batch_img)
        loss = criterion(batch_recon, batch_img)
        test_losses.append(loss.item())

avg_test_loss = sum(test_losses) / len(test_losses)
print(f"Avg Test Loss: {avg_test_loss}")
train_losses.append(avg_test_loss)

# try:
loss_tracker = pd.read_csv(LOSS_TRACKER_PATH,header=None,index_col=0)
# except:
    # print("File not found, creating new file")
    # loss_tracker = pd.DataFrame()
loss_tracker.loc[id_config] = train_losses
loss_tracker.to_csv(LOSS_TRACKER_PATH,header=None)
print(f"written to: {LOSS_TRACKER_PATH}")
torch.save(model_v.state_dict(), MODEL_STATEDICT_SAVE_PATH)
print(f"written to: {MODEL_STATEDICT_SAVE_PATH}")
torch.save(model_v, MODEL_SAVE_PATH) # this saves the model as-is
print(f"written to: {MODEL_SAVE_PATH}")

print(f"Elapsed time: {time.time() - start} seconds.")