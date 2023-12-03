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
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import time
from datetime import datetime
import sys
from utils.classes import *
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from copy import deepcopy

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

lrs = [432,408,384,360,336,312,288,264,240,216,192]
# lrs = [-1]
for i,lr in enumerate(lrs[::-1]):
    print(f"LR - {lr}")

    ID = sys.argv[1]
    nownow = datetime.now()

    TASK = "img"
    CONDUCTANCE_VALUES = ""
    id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"

    LOSS_TRACKER_PATH = f'./results/loss_tracker_{TASK}.csv'
    MODEL_STATEDICT_SAVE_PATH = f"./models/{TASK}/{id_config}_{TASK}.pth"
    MODEL_SAVE_PATH = MODEL_STATEDICT_SAVE_PATH[:-1] # pt instead of pth
    
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

    # recon_path = "./models/img/14.2.20231110000321.pt"
    # recon = AutoencoderEIT142()
    recon = torch.load("./models/img/14.2.1.retraining.2.20231130014311_img.pt") #torch.load("./models/img/14.2.20231110000321.pt")
    recon.encoder = nn.Sequential(
        *recon.encoder,
        nn.Linear(216,lr)
    )
    recon.decoder = nn.Sequential(
        nn.Linear(lr,216),
        nn.ReLU(),
        *recon.decoder
    )
    '''
    # recon.encoder = nn.Sequential(
    #     *recon.encoder[:6],
    #     nn.Conv2d(192, 48, 3, 2, 1),
    #     nn.ReLU(),
    #     nn.Flatten(),
    #     nn.Linear(432, lr),
    #     nn.ReLU()
    # )
    # recon.decoder = nn.Sequential(
    #     nn.Linear(lr, 432),
    #     nn.ReLU(),
    #     nn.Unflatten(1, (48, 3, 3)),
    #     nn.ConvTranspose2d(48, 192, 3, 2, 1, 1),
    #     nn.ReLU(),
    #     *recon.decoder[2:] # no tanh
    # )
    '''
    for param in recon.parameters():
        param.requires_grad = True
    recon = recon.to(device)
    if i == 0:
        print("Only gonna be shown once!")
        summary(recon, (1,24,24))
        print("Locals")
        print(locals())

    alpha = 0.5  # Weight for MSE Loss
    beta = 0.05 # Weight for SSIM
    optimizer_config = {'Adam': {'learning_rate':1e-4,'weight_decay':1e-5}}
    optimizer = optimizer_build(optimizer_config,recon)
    best_recon = deepcopy(recon)
    # min_loss = np.inf
    min_val_loss = np.inf
    best_epoch = -1

    print("Training started!! - Reconstruction")
    train_losses = []
    train_losses_ssim = []
    for epoch in range(epochs):
        epoch_loss = 0.0 
        epoch_ssim = 0.0
        for i, (_, batch_img) in enumerate(train_dataloader):
            batch_img = batch_img.to(device)
            _,batch_recon = recon(batch_img)

            min_value = batch_recon.min()
            max_value = batch_recon.max()
            batch_recon = (batch_recon - min_value) / (max_value - min_value) \
                * (batch_img.max() - batch_img.min()) + batch_img.min()

            mse_loss = F.mse_loss(batch_recon, batch_img)
            batch_recon = batch_recon.squeeze().cpu().detach().numpy()
            batch_img = batch_img.squeeze().cpu().detach().numpy()
            ssim_value = 1 - ssim(batch_recon, batch_img,
                                data_range=max(batch_recon.max(), batch_img.max()) - min(batch_img.min(), batch_recon.min()))

            # Combined weighted loss
            loss = alpha * mse_loss + beta * ssim_value

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss.item()
        
        epoch_loss /= len(train_dataloader)
        optimizer.zero_grad()
        epoch_loss = torch.tensor(epoch_loss,device=device,requires_grad=True)
        optimizer.step()

        # train_losses.append(loss.item())
        train_losses.append(mse_loss.item())
        train_losses_ssim.append(ssim_value)

        # VALIDATION
        with torch.no_grad():
            val_loss = 0.0
            for _, val_batch_img in enumerate(val_dataloader):
                val_batch_img = val_batch_img.to(device)
                _, val_batch_recon = recon(val_batch_img)

                # Calculate validation loss
                val_loss += F.mse_loss(val_batch_recon, val_batch_img).item()

            val_loss /= len(val_dataloader)

            # Check if validation loss is lower than the current best
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_recon_state_dict = deepcopy(recon)  # Save the state_dict instead of the whole model
                best_epoch = epoch
                print(f"Best state dict, with Epoch validation MSELoss {min_val_loss:.6f}, found @ {epoch}...")


        if epoch % 10 == 0 or (epoch == epochs - 1):
            print(f'Epoch:{epoch}, Epoch Loss: {epoch_loss:.6f}, Loss:{loss.item():.6f}, MSE Loss: {mse_loss.item():.6f}, SSIM: {ssim_value:.6f}, Val_MSELoss: {min_val_loss:.6f}')
        
        # if epoch > 20 and epoch < epochs - epochs // 2:
        #     impr = 1  - loss / train_losses[epoch-20]
        #     if  impr < 0.05:
        #         early_stop = True
        #         print(f"Early stop at epoch: {epoch}, loss: {loss}, improvement: {impr}")
        #         break

    recon = deepcopy(best_recon)
    recon.eval()
    test_losses = [] 
    test_losses_ssim = [] 
    with torch.no_grad():
        for i, (_, batch_img) in enumerate(test_dataloader):
            batch_img = batch_img.to(device)
            _,batch_recon = recon(batch_img)

            mse_loss = F.mse_loss(batch_recon, batch_img)
            batch_recon = batch_recon.squeeze().cpu().detach().numpy()
            batch_img = batch_img.squeeze().cpu().detach().numpy()
            ssim_value = 1 - ssim(batch_recon, batch_img,
                                data_range=max(batch_recon.max(), batch_img.max()) - min(batch_img.min(), batch_recon.min()))

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
    # loss_tracker = pd.read_csv(LOSS_TRACKER_PATH,header=None,index_col=0)
    # except:
        # print("File not found, creating new file")
        # loss_tracker = pd.DataFrame()
    # loss_tracker.loc[id_config] = train_losses
    # loss_tracker.loc[f"ssim-{id_config}"] = train_losses_ssim
    # loss_tracker.to_csv(LOSS_TRACKER_PATH,header=None)

    import csv
    with open(LOSS_TRACKER_PATH, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([id_config,*train_losses])
        writer.writerow([f"ssim-{id_config}",*train_losses_ssim])

    print(f"written to: {LOSS_TRACKER_PATH}")
    torch.save(recon.state_dict(), MODEL_STATEDICT_SAVE_PATH)
    print(f"written to: {MODEL_STATEDICT_SAVE_PATH}")
    torch.save(recon, MODEL_SAVE_PATH) # this saves the model as-is
    print(f"written to: {MODEL_SAVE_PATH}")

    print(f"Elapsed time: {time.time() - start} seconds.")