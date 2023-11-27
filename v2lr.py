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
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import csv

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

ID = sys.argv[1]
TASK = "v2lr"
CONDUCTANCE_VALUES = ""

DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"

# DIFFS_IMGS_TRAIN_PATH = "./data/eit2/diffs_imgs_train_2.csv"
# DIFFS_IMGS_TEST_PATH = "./data/eit2/diffs_imgs_test_2.csv"

# DIFFS_IMGS_TRAIN_PATH = "./data/eit4/diffs_imgs_train_4.csv"
# DIFFS_IMGS_TEST_PATH = "./data/eit4/diffs_imgs_test_4.csv"

# DIFFS_IMGS_TRAIN_PATH = "./data/eit6/diffs_imgs_train_6.csv"
# DIFFS_IMGS_TEST_PATH = "./data/eit6/diffs_imgs_test_6.csv"

recon_path = "./models/img/14.2.20231110000321.pt" # "./models/img/14.2.1.20231116190651_img.pt"
# recon_path = "./models/img/2_14.2.1.20231116193059_img.pt" #2
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

factors = [128,256,512,108,324,432]
# for 2
config_no = 3#int(sys.argv[2])
if config_no == 2:
    configs = [
    [
        nn.Linear(256, a),
        nn.ReLU(),
        nn.Linear(a,216),
        nn.ReLU()
    ] for a in factors]
elif config_no == 3:
    # configs = [
    # [
    #     nn.Linear(256, 128),
    #     nn.ReLU(),
    #     nn.Linear(128,128),
    #     nn.ReLU(),
    #     nn.Linear(128,216),
    #     nn.ReLU()
    # ]]

    configs = [[
        nn.Conv2d(1, 8, 3, 2, 1),
        nn.BatchNorm2d(8), 
        nn.Conv2d(8, 8, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1, 0),
        nn.BatchNorm2d(16),  
        nn.Conv2d(16, 16, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(16, 24, 3, 2, 1),
        nn.BatchNorm2d(24),  
        nn.Conv2d(24, 24, 3, 1, 1),
        nn.ReLU(),
        nn.Flatten()
    ]]
elif config_no == 4:
    configs = [
    [
        nn.Linear(256, 324),
        nn.ReLU(),
        nn.Linear(324,256),
        nn.ReLU(),
        nn.Linear(256,324),
        nn.ReLU(),
        nn.Linear(324, 216),
        nn.ReLU()
    ] ]

    

for i,config in enumerate(configs[::-1]):
    print(f"Iteration config: {i}")
    print(config)
    print("\n")

    nownow = datetime.now()
    id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"
    LOSS_TRACKER_PATH = f'./results/loss_tracker_{TASK}.csv'
    MODEL_STATEDICT_SAVE_PATH = f"./models/{TASK}/{id_config}_{TASK}.pth"
    MODEL_SAVE_PATH = MODEL_STATEDICT_SAVE_PATH[:-1] # pt instead of pth

    v2lr = V2ImgLR()
    v2lr.v2lr = nn.Sequential(
            # nn.Flatten(),
            *config
        )
    v2lr = v2lr.to(device)
    criterion = nn.MSELoss()
    optimizer_config = {'Adam': {'learning_rate':1e-3,'weight_decay':1e-5}}
    optimizer = optimizer_build(optimizer_config,v2lr)

    print(summary(v2lr,(1,16,16)))
    recon = torch.load(recon_path)
    print(f"Reconstructor from: {recon_path}")

    alpha = 0.7  # Weight for MSE Loss
    beta = 0.3 # Weight for SSIM
    train_losses = []
    early_stop = False
    for epoch in range(epochs):
        for i, (batch_diff, batch_img) in enumerate(train_dataloader):
            batch_diff = batch_diff.to(device)
            batch_img = batch_img.to(device)
            batch_mapped = v2lr(batch_diff)

            batch_encoded = recon.encoder[1:](batch_img)
            batch_recon_v = recon.decoder[:-2](batch_mapped)
            
            mse_loss = F.mse_loss(batch_mapped, batch_encoded) # mse between v2lr

            batch_recon_v_np = batch_recon_v.squeeze().cpu().detach().numpy()
            batch_img_np = batch_img.squeeze().cpu().detach().numpy()

            ssim_value = 1 - ssim(batch_recon_v_np, batch_img_np,
                    data_range=max(batch_recon_v_np.max(), batch_img_np.max()) - min(batch_img_np.min(), batch_recon_v_np.min()),
                    win_size=3) # general

            loss = alpha*mse_loss + beta*ssim_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % (10) == 0) or (epoch == epochs - 1):
            recon_v_loss = F.mse_loss(batch_img, batch_recon_v)

            print(f'Task: v2lr, Epoch:{epoch+1}, Loss:{loss.item():.6f}\
                \t\tTask: InvProb, Epoch:{epoch+1}, MSELoss:{recon_v_loss.item():.6f}, SSIM:{ssim_value}')

        # if epoch > 10 and epoch < epochs - 40:
        #     if (train_losses[epoch-10] - loss.item()) / train_losses[epoch-10] < 0.5:
        #         early_stop = True
        #         break
        
        train_losses.append(loss.item())

    if not early_stop:
        v2lr.eval()
        test_losses = [] 
        with torch.no_grad():
            for i, (batch_diff, batch_img) in enumerate(test_dataloader):
                batch_diff = batch_diff.to(device)
                batch_img = batch_img.to(device)
                batch_mapped = v2lr(batch_diff)

                batch_encoded = recon.encoder[1:](batch_img)
                batch_recon_v = recon.decoder[:-2](batch_mapped)
                
                mse_loss = F.mse_loss(batch_mapped, batch_encoded) # mse between v2lr

                batch_recon_v = batch_recon_v.squeeze().cpu().detach().numpy()
                batch_img = batch_img.squeeze().cpu().detach().numpy()

                ssim_value = 1 - ssim(batch_recon_v, batch_img,
                        data_range=max(batch_recon_v.max(), batch_img.max()) - min(batch_img.min(), batch_recon_v.min()),
                        win_size=3) # general

                loss = alpha*mse_loss + beta*ssim_value

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

        # loss_tracker = pd.read_csv(LOSS_TRACKER_PATH,header=None,index_col=0)
        # loss_tracker.loc[id_config] = train_losses
        # loss_tracker.to_csv(LOSS_TRACKER_PATH,header=None)

        with open(LOSS_TRACKER_PATH, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(train_losses)

        print(f"written to: {LOSS_TRACKER_PATH}")
    else:
        print(f"Early stopped at epoch {epoch} and model is not saved")

    print(f"Elapsed time: {time.time() - start} seconds.")