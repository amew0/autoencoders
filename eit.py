import sys
CONFIG_YML = sys.argv[1]
print(CONFIG_YML)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import pandas as pd
import time

nownow = datetime.now()
start = time.time()
print(nownow.strftime("%Y%m%d%H%M%S%f"))
device="cpu" if not torch.cuda.is_available() else "cuda:0"
print(f"Using: {device}")

DATA_PATH = "./data/eit/24x24_Images_11Cond_30k_2022-02-23.csv"
TRAIN_PATH = "./data/eit/train_images.csv"
TEST_PATH = "./data/eit/test_images.csv"
# RESULTS_PATH = "./results"
# TRAIN_LOSS_TRACKER_PATH = './results/loss_tracker.csv'
TEST_LOSS_TRACKER_PATH = './results/loss_tracker_test.csv'
YML_ID = CONFIG_YML[-8:-4] # /../../xxxx.yml => xxxx
MODEL_SAVE_PATH = f"./models/{YML_ID}.pth"

def load_yml(yml_path):
    with open(yml_path,'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

config = load_yml(CONFIG_YML)

encoder_config = config['encoder']
decoder_config = config['decoder']
loss_config = config['criterion']['loss']
epochs = config['epochs']
batch_size = config['batch_size']
seed = config['seed']
optimizer_config = config['optimizer']

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = np.array(self.data.iloc[idx].values,dtype=np.float32)
        image = np.tanh(image_data.reshape(24, 24))  # Reshape to 24x24 and convert to PIL Image
        if self.transform:
            image = self.transform(image)

        return image.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = CustomDataset(csv_file=TRAIN_PATH, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = CustomDataset(csv_file=TEST_PATH, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def encoder_build(block_config):
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

class AutoencoderEIT_config(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder_build(encoder_config)
        self.decoder = decoder_build(decoder_config)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def loss_build(loss_config):
    if loss_config == 'MSELoss':
        return nn.MSELoss()

def optimizer_build(optimizer_config,model:AutoencoderEIT_config):

    if 'Adam' in optimizer_config:
        adam = optimizer_config['Adam']
        return torch.optim.Adam(model.parameters(),
                             lr=adam['learning_rate'],
                             weight_decay=adam['weight_decay'])

model = AutoencoderEIT_config().to(device)
criterion = loss_build(loss_config)
optimizer = optimizer_build(optimizer_config,model)

id_config = f"{YML_ID}.{nownow.strftime('%Y%m%d%H%M%S%f')}"
train_losses = []
for epoch in range(epochs):
    for i, (img) in enumerate(train_dataloader):
        img = img.to(device )
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % (epochs//10) == 0:
        print(f'ID_config:{id_config} Epoch:{epoch+1}, Loss:{loss.item():.6f}')
    train_losses.append(loss.item())

model.eval()
test_losses = [] 
with torch.no_grad():
    for img in test_dataloader:
        img = img.to(device)  # Move batch of testing images to the GPU
        recon = model(img)
        loss = criterion(recon, img)
        test_losses.append(loss.item())

avg_test_loss = sum(test_losses) / len(test_losses)
train_losses.append(avg_test_loss)

loss_tracker = pd.read_csv(TEST_LOSS_TRACKER_PATH,header=None,index_col=0)
loss_tracker.loc[id_config] = train_losses
loss_tracker.to_csv(TEST_LOSS_TRACKER_PATH,header=None)

torch.save(model.state_dict(), MODEL_SAVE_PATH)

# with open(f"{RESULTS_PATH}/{YML_ID}.yml", 'w') as f:
#     yaml.dump(config, f)
print(f"Elapsed time: {time.time() - start} seconds.")