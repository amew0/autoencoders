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
print("Importing finished!!")
start = time.time()
seed = 646
batch_size = 64
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is going to be used!!")

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"
LOSS_TRACKER_PATH = './results/loss_tracker_diffimg.csv'
MODEL_STATEDICT_SAVE_PATH = "./models/0001_diffimg.pt"
MODEL_SAVE_PATH = "./models/0001_diffimg.pt"

class DiffImg(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        diff_image_data = np.array(self.data.iloc[idx].values,dtype=np.float32)
        diff_data = diff_image_data[:256]
        image_data = diff_image_data[256:]
        diff = diff_data.reshape(16, 16)
        image = np.tanh(image_data.reshape(24, 24))  # Reshape to 16x16
        if self.transform:
            diff = self.transform(diff)
            image = self.transform(image)

        return diff,image

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = DiffImg(csv_file=DIFFS_IMGS_TRAIN_PATH, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = DiffImg(csv_file=DIFFS_IMGS_TEST_PATH, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Dataset loaded!!")

# just put it there for the forward func
class AutoencoderEIT_config(nn.Module):
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

class VoltageAE(nn.Module):
    def __init__(self):
        super().__init__()
        model_AE = torch.load("models/0014.pt")
        model_AE.eval()
        self.encoder_v = nn.Sequential(
            # N,1,16,16
            nn.Conv2d(1, 32, 5, stride=1, padding=0), # N, 32, 12, 12
            nn.ReLU(),
            nn.Conv2d(32, 96, 5, stride=1, padding=0), # N, 96, 8, 8
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, stride=1, padding=0), # N, 192, 6, 6
            nn.Tanh(),
        )
        for param in model_AE.decoder.parameters():
            param.requires_grad = False
        self.decoder = model_AE.decoder 
    def forward(self, x):
        encoded_v = self.encoder_v(x)
        decoded = self.decoder(encoded_v)
        return encoded_v,decoded
    
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
optimizer_config = {'Adam': {'learning_rate':5e-2,'weight_decay':1e-3}}
optimizer = optimizer_build(optimizer_config,model_v)
model_v.eval()

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

    if epoch % (10) == 0:
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.6f}')
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

nownow = datetime.now()
id_config = f"{nownow.strftime('%Y%m%d%H%M%S%f')}"


# try:
loss_tracker = pd.read_csv(LOSS_TRACKER_PATH,header=None,index_col=0)
# except:
    # print("File not found, creating new file")
    # loss_tracker = pd.DataFrame()
loss_tracker.loc[id_config] = train_losses
loss_tracker.to_csv(LOSS_TRACKER_PATH,header=None)

torch.save(model_v.state_dict(), MODEL_STATEDICT_SAVE_PATH)
torch.save(model_v, MODEL_SAVE_PATH) # this saves the model as-is

print(f"Elapsed time: {time.time() - start} seconds.")