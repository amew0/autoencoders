import sys
import time
import csv
from datetime import datetime
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torchmetrics.image import StructuralSimilarityIndexMeasure
from utils.classes import *

print("Importing finished!!")

start = time.time()
seed,batch_size,epochs = 64,4,200
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
recon_path = "./models/img/14.2.1.retraining.2.20231130014311_img.pt" # "./models/img/14.2.1.20231116190651_img.pt" #"./models/img/14.2.20231110000321.pt"

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

trainer = LossTracker("Training")
validator = LossTracker("Validation")
tester = LossTracker("Testing")

def play(dataloader:DataLoader=None,
         tracker:LossTracker=None,
         v2lr:nn.Module=None,
         mse=None,ssim=None,optimizer=None,
         scale_to_input = False):
    
    v2lr.train() if tracker.job=="Training" else v2lr.eval()

    for i, (batch_diff, batch_img) in enumerate(dataloader):
        batch_diff = batch_diff.to(device)
        batch_img = batch_img.to(device)
        batch_mapped, batch_lr, batch_recon_v = v2lr(batch_diff,batch_img)

        if scale_to_input: # tempting but never was it :)
            min_value = batch_recon_v.min()
            max_value = batch_recon_v.max()

            batch_recon_v = (batch_recon_v - min_value) / (1e-8+max_value - min_value) \
                * (batch_img.max() - batch_img.min()) + batch_img.min()
        
        mse_loss = F.mse_loss(batch_img, batch_recon_v).requires_grad_()
        ssim_value = 1 - ssim(batch_img, batch_recon_v).requires_grad_() 
        loss = alpha*mse_loss + beta*ssim_value
     
        if tracker.job == "Training": # batch
            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

        tracker.mse_loss = mse_loss.item()
        tracker.ssim = ssim_value.item()
        tracker.loss = loss.item()
        tracker.mse_loss_lr = mse(batch_lr, batch_mapped)

        tracker.epoch_mse_loss += tracker.mse_loss
        tracker.epoch_ssim += tracker.ssim
        tracker.epoch_loss += tracker.loss
        tracker.epoch_loss_lr += tracker.mse_loss_lr.item()

    tracker.epoch_mse_loss /= len(dataloader)
    tracker.epoch_ssim /= len(dataloader)
    tracker.epoch_loss /= len(dataloader)
    tracker.epoch_loss_lr /= len(dataloader)

    # if tracker.job == "Training": # epoch

    #     optimizer.zero_grad()
    #     epoch_loss = torch.tensor(tracker.epoch_loss,device=device,requires_grad=True)
    #     epoch_loss.backward()
    #     optimizer.step()
    return tracker

# '''
# configs = [nn.Sequential(
    nn.Conv2d(1,6,5,1,0),
    nn.BatchNorm2d(6),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Conv2d(6,18,5,1,0),
    nn.BatchNorm2d(18),
    nn.ReLU(),
    nn.Dropout(0.3),
    
    nn.Conv2d(18,54,5,1,0),
    nn.BatchNorm2d(54),
    nn.ReLU(),
    nn.Dropout(0.3),
    
    nn.Conv2d(54,108,3,1,0),
    nn.BatchNorm2d(108),   
    nn.ReLU(),
    nn.Dropout(0.3),
    
    nn.Conv2d(108,216,2,1,0),
    nn.BatchNorm2d(216),
    nn.Flatten()
# )]
# '''
'''
factors = [6,8,9,12,18,27,36,54,72,108]
configs = [
    nn.Sequential(
    nn.Conv2d(1,a,5,1,0),
    nn.BatchNorm2d(a),
    nn.ReLU(),
    nn.Conv2d(a,b,5,1,0),
    nn.BatchNorm2d(b),
    nn.ReLU(),
    nn.Conv2d(b,c,5,1,0),
    nn.BatchNorm2d(c),
    nn.ReLU(),
    nn.Conv2d(c,d,3,1,0),
    nn.BatchNorm2d(d),   
    nn.ReLU(),
    nn.Conv2d(d,216,2,1,0),
    nn.BatchNorm2d(216),
    nn.Flatten()
) 
for a in factors\
for b in factors\
for c in factors\
for d in factors\
if a < b and b < c and c < d
]
'''
# alphas = [0.1]

# alphas = [i/100 for i in range(9,39)][::2]
configs = [-1]
learning_rate = 5e-3
momentum = 0.9
weight_decay = 1e-5
optimizers = [
    # {'LBFGS': {}},
    {'SGD': {'learning_rate':learning_rate, 'weight_decay':weight_decay,'momentum':momentum}},
    {'RMSprop': {'learning_rate':learning_rate, 'weight_decay':weight_decay}},
    {'Adagrad': {'learning_rate':learning_rate, 'weight_decay':weight_decay}},
    {'Adam': {'learning_rate':learning_rate, 'weight_decay':weight_decay}}
]
for i,optimizer_config in enumerate(optimizers):
    alpha=0.01
    print(f"optimzer {optimizer_config}")
    nownow = datetime.now()
    id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"
    LOSS_TRACKER_PATH = f'./results/loss_tracker_{TASK}.csv'
    MODEL_STATEDICT_SAVE_PATH = f"./models/{TASK}/{id_config}_{TASK}.pth"
    MODEL_SAVE_PATH = MODEL_STATEDICT_SAVE_PATH[:-1] # pt instead of pth

    # v2lr = V2ImgLR(recon_path)
    # v2lr.v2lr = config
    v2lr = torch.load("./models/v2lr/0.alpha.01.20231206000636_v2lr.pt")
    v2lr = v2lr.to(device)
    print(v2lr.v2lr)
    # summary(v2lr,(1,16,16))
    
    mse = nn.MSELoss()
    ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)
    
    # optimizer_config = {'Adam': {'learning_rate':3e-5,}}
    optimizer = optimizer_build(optimizer_config,v2lr)

    best_v2lr = deepcopy(v2lr)
    min_loss = np.inf
    best_epoch = 0

    alpha = alpha  # Weight for MSE Loss
    beta = 0.1584*(1-alpha) # Weight for SSIM

    train_losses = []
    last_printed = 0
    for epoch in range(epochs):
        trainer = play(train_dataloader,trainer,v2lr,mse,ssim,optimizer)
        train_losses.append(trainer.loss)
        
        with torch.no_grad():
            validator =play(val_dataloader,validator,v2lr,mse,ssim)

        if validator.epoch_loss < min_loss:

            min_loss = validator.epoch_loss
            del best_v2lr
            best_v2lr = deepcopy(v2lr)
            best_epoch = epoch

            trainer.best_epoch = best_epoch
            validator.best_epoch = best_epoch

            print(f"{trainer} !==! {validator}")
            last_printed = epoch
        
        if epoch - last_printed > 20:
            print(f"Digging!! {trainer} !==! {validator}")
            last_printed = epoch

            # note that this is not the best epoch
            trainer.best_epoch = epoch
            validator.best_epoch = epoch

    del v2lr
    v2lr = deepcopy(best_v2lr)
    with torch.no_grad():
        tester = play(test_dataloader,tester,v2lr,mse,ssim)
        
        train_losses.append(tester.epoch_loss)
        # print(f"Avg Test Loss: {test_epoch_loss} Avg Test MSE: {test_epoch_mse} Avg Test SSIM {test_epoch_ssim}, Last MSE Test Loss: {test_mse_loss}")
        print(tester)

    # torch.save(v2lr.state_dict(), MODEL_STATEDICT_SAVE_PATH)
    # print(f"written to: {MODEL_STATEDICT_SAVE_PATH}")
    torch.save(v2lr, MODEL_SAVE_PATH) # this saves the model as-is
    print(f"written to: {MODEL_SAVE_PATH}")

    with open(LOSS_TRACKER_PATH, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(train_losses)

    print(f"written to: {LOSS_TRACKER_PATH}")
    end = time.time()
    print(f"Elapsed time: {end - start} seconds.")
    start = end