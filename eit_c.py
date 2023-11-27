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
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

print("Importing finished!!")
start = time.time()
seed = 64
batch_size = 4
epochs = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} is going to be used!!")

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

factors = [108,144,54,324,432]
configs = [
[
    nn.Linear(216, a),
    nn.ReLU(),
    nn.Linear(a,b),
    nn.ReLU(),
    nn.Linear(b,c),
    nn.ReLU(),
    nn.Linear(c, 216),
    nn.ReLU()
] for a in factors for b in factors  for c in factors]
configs = [
    [
        nn.Linear(216, 108),
        nn.ReLU(),
        nn.Linear(108,324),
        nn.ReLU(),
        nn.Linear(324,324),
        nn.ReLU(),
        nn.Linear(324, 216),
        nn.ReLU()
    ]
]
# configurations to be tested
for i,config in enumerate(configs):
    print(f"Iteration config: {i}")
    print(config)
    print("\n")
    ID = sys.argv[1]
    nownow = datetime.now()

    TASK = "img"
    CONDUCTANCE_VALUES = ""
    id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"


    DIFFS_IMGS_TRAIN_PATH = "./data/eit/diffs_imgs_train.csv"
    DIFFS_IMGS_TEST_PATH = "./data/eit/diffs_imgs_test.csv"

    # DIFFS_IMGS_TRAIN_PATH = "./data/eit2/diffs_imgs_train_2.csv"
    # DIFFS_IMGS_TEST_PATH = "./data/eit2/diffs_imgs_test_2.csv"

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

    recon = AutoencoderEIT142()
    recon.encoder = recon.encoder + nn.Sequential(*config)
    recon = recon.to(device)
    # criterion = nn.MSELoss()
    alpha = 0.3  # Weight for MSE Loss
    beta = 0.7 # Weight for SSIM
    optimizer_config = {'Adam': {'learning_rate':1e-3,'weight_decay':1e-5}}
    optimizer = optimizer_build(optimizer_config,recon)
    print(summary(recon, (1,24,24)))

    print("Training started!! - Reconstruction")
    train_losses = []
    train_losses_ssim = []
    for epoch in range(epochs):
        for i, (_, batch_img) in enumerate(train_dataloader):
            batch_img = batch_img.to(device)
            _,batch_recon = recon(batch_img)

            mse_loss = F.mse_loss(batch_recon, batch_img)
            batch_recon = batch_recon.squeeze().cpu().detach().numpy()
            batch_img = batch_img.squeeze().cpu().detach().numpy()
            ssim_value = 1 - ssim(batch_recon, batch_img,
                                data_range=max(batch_recon.max(), batch_img.max()) - min(batch_img.min(), batch_recon.min()),
                                win_size=3)

            # Combined weighted loss
            loss = alpha * mse_loss + beta * ssim_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or (epoch == epochs - 1):
            print(f'Epoch:{epoch}, Loss:{loss.item():.6f}, MSE Loss: {mse_loss.item():.6f}, SSIM: {ssim_value:.6f}')
        # train_losses.append(loss.item())
        train_losses.append(mse_loss.item())
        train_losses_ssim.append(ssim_value)

    recon.eval()
    test_losses = [] 
    test_losses_ssim = [] 
    with torch.no_grad():
        for i, (_, batch_img) in enumerate(train_dataloader):
            batch_img = batch_img.to(device)
            _,batch_recon = recon(batch_img)

            mse_loss = F.mse_loss(batch_recon, batch_img)
            batch_recon = batch_recon.squeeze().cpu().detach().numpy()
            batch_img = batch_img.squeeze().cpu().detach().numpy()
            ssim_value = 1 - ssim(batch_recon, batch_img,
                                data_range=max(batch_recon.max(), batch_img.max()) - min(batch_img.min(), batch_recon.min()),
                                win_size=3)

            # Combined weighted loss
            loss = alpha * mse_loss + beta * ssim_value

            test_losses.append(mse_loss.item())
            test_losses_ssim.append(ssim_value.item())

    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_loss_ssim = sum(test_losses_ssim) / len(test_losses_ssim)

    print(f"Avg Test Loss: MSE: {avg_test_loss}, SSIM: {avg_test_loss_ssim}")
    print(f"Max Test Loss: MSE: {max(test_losses)} SSIM: {max(test_losses_ssim)}")
    print(f"Min Test Loss: MSE: {min(test_losses)} SSIM: {min(test_losses_ssim)}")
    train_losses.append(avg_test_loss)
    train_losses_ssim.append(avg_test_loss_ssim)

    # try:
    loss_tracker = pd.read_csv(LOSS_TRACKER_PATH,header=None,index_col=0)
    # except:
        # print("File not found, creating new file")
        # loss_tracker = pd.DataFrame()
    loss_tracker.loc[id_config] = train_losses
    loss_tracker.loc[f"ssim-{id_config}"] = train_losses_ssim
    loss_tracker.to_csv(LOSS_TRACKER_PATH,header=None)
    print(f"written to: {LOSS_TRACKER_PATH}")
    torch.save(recon.state_dict(), MODEL_STATEDICT_SAVE_PATH)
    print(f"written to: {MODEL_STATEDICT_SAVE_PATH}")
    torch.save(recon, MODEL_SAVE_PATH) # this saves the model as-is
    print(f"written to: {MODEL_SAVE_PATH}")

    print(f"Elapsed time: {time.time() - start} seconds.")