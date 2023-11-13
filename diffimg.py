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

DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"
LOSS_TRACKER_PATH = './results/loss_tracker_diffimg.csv'
MODEL_STATEDICT_SAVE_PATH = f"./models/{id_config}_diffimg.pth"
MODEL_SAVE_PATH = f"./models/{id_config}_diffimg.pt"

class DiffImg(Dataset):
    def __init__(self, csv_file, diff_transform=None, img_transform=None):
        self.data = pd.read_csv(csv_file,header=None)
        self.diff_transform = diff_transform
        self.img_transform = img_transform
        self.scaler = MinMaxScaler()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        diff_image_data = np.array(self.data.iloc[idx].values,dtype=np.float32)
        diff = diff_image_data[:256]
        image = diff_image_data[256:]
        
        if self.diff_transform:
            diff = diff.reshape(16, 16)
            diff = self.diff_transform(diff)
        
        if self.img_transform:
            # use minmaxscaler to transform image_data        
            # image = self.scaler.fit_transform(image.reshape(-1,1))
            image = image.reshape(24, 24)
            image = np.tanh(image)  # Reshape to 16x16
            image = self.img_transform(image)

        return diff,image

diff_transform = transforms.Compose([
    transforms.ToTensor(),
])

img_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = DiffImg(csv_file=DIFFS_IMGS_TRAIN_PATH, diff_transform=diff_transform, img_transform=img_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = DiffImg(csv_file=DIFFS_IMGS_TEST_PATH, diff_transform=diff_transform, img_transform=img_transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Dataset loaded!!")

# just put it there for the forward func
class AutoencoderEIT_config(nn.Module):
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU()
        self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)  # Add a 1x1 convolution for the skip connection
    def forward(self, x):
        out = self.conv2(self.relu(self.conv1( x)))
        skip = self.skip(x)
        out += skip[:, :, :out.size(2), :out.size(3)]
        out = self.relu(out)
        return out
class MinMax(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x_:torch.Tensor):
        min_vals, _ = x_.min(dim=2, keepdim=True)
        max_vals, _ = x_.max(dim=2, keepdim=True)
        max_vals = torch.where(min_vals == max_vals, min_vals + 1e-8, max_vals)
        normalized = (x_ - min_vals) / (max_vals - min_vals)
        # Clip values to the range [0, 1]
        # normalized = torch.clamp(normalized, 0, 1)
        return normalized
class AutoencoderEIT(nn.Module):
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded
class VoltageAE(nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.load("./models/img/14.2.20231110000321.pt")
        # make all model params untrainable
        for param in model.parameters():
            param.requires_grad = False
        self.encoder_v = nn.Sequential(
            # nn.Conv2d(1,32,3,1,1),   # ResidualBlock(1, 32, 3, 1, 1), # 32 16
            # nn.ReLU(),
            nn.Conv2d(1,48,5,1,0), # ResidualBlock(32, 48, 5, 1, 0), # 48 12
            nn.ReLU(),
            # nn.Conv2d(48,96,2,2,2), # ResidualBlock(48, 96, 2, 2, 2), # 96 8
            # nn.ReLU(),
            # nn.Conv2d(96,192,3,1,0), # ResidualBlock(96, 192, 3, 1, 0) # 192 6 +
            # nn.ReLU(),
            # nn.Conv2d(192, 24, 3, 2, 1), # N,24,3,3
            # nn.ReLU(),
            # nn.Flatten()
            nn.Flatten(),
            nn.Linear(48*12*12,48*9*9),
            nn.ReLU(),
            nn.Linear(48*9*9,48*3*3),
            nn.ReLU(),
            nn.Linear(48*3*3,24*3*3),
            nn.ReLU()
        ) 
        self.decoder = nn.Sequential(
            *model.decoder[:-1] # last is flatten so ignored
        )
        # for layer in self.encoder_v:
        #     if isinstance(layer, nn.Conv2d):
        #         mean_value = 1e-5
        #         std_value = 1e-5
        #         nn.init.normal_(layer.weight, mean=mean_value, std=std_value)
        #         nn.init.constant_(layer.bias, 0)
        # for layer in self.decoder:
        #     if isinstance(layer, nn.ConvTranspose2d):
        #         # You can change the mean and std values as needed
        #         mean_value = 1e-4
        #         std_value = 1e-1
        #         nn.init.normal_(layer.weight, mean=mean_value, std=std_value)
        #         nn.init.constant_(layer.bias, 0)
    def forward(self, x):
        encoded_v = self.encoder_v(x)
        decoded = self.decoder(encoded_v)
        return encoded_v,decoded
   
# class AutoencoderEIT_v (nn.Module):


def loss_build(loss_config):
    if loss_config == 'MSELoss':
        return nn.MSELoss()

def optimizer_build(optimizer_config,model):

    if 'Adam' in optimizer_config:
        adam = optimizer_config['Adam']
        return torch.optim.Adam(model.parameters(),
                            lr=adam['learning_rate'],
                            weight_decay=adam['weight_decay'])

model_v = VoltageAE().to(device)
criterion = loss_build('MSELoss')
optimizer_config = {'Adam': {'learning_rate':1e-3,'weight_decay':0}}
optimizer = optimizer_build(optimizer_config,model_v)

print(summary(model_v, (1,16,16)))

# outputs = []
print("Training started!!")
train_losses = []
for epoch in range(epochs):
    for i, (batch_diff, batch_img) in enumerate(train_dataloader):
        batch_diff = batch_diff.to(device)
        batch_img = batch_img.to(device)
        _,batch_recon = model_v(batch_diff)
        loss = criterion(batch_recon, batch_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if epoch % (epochs // 10) == 0:
    print(f'Epoch:{epoch}, Loss:{loss.item():.6f}')
    train_losses.append(loss.item())
    # outputs.append((epoch, batch_img, batch_recon))

model_v.eval()
test_losses = [] 
with torch.no_grad():
    for i, (batch_diff, batch_img) in enumerate(test_dataloader):
        batch_diff = batch_diff.to(device)
        batch_img = batch_img.to(device)
        _,batch_recon = model_v(batch_diff)
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