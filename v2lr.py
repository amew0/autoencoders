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
seed,batch_size,epochs = 64,8,200
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

def compute_mse_aoi(recons,preds):
    mses_aoi = torch.tensor(0.0, device=device)
    for recon,pred in zip(recons,preds):
        vals,counts = recon.unique(return_counts=True)
        
        foreground_value = vals[counts.argmin()]
        aoi = (recon == foreground_value).nonzero()
        xmin=aoi[:,1].min()
        xmax=aoi[:,1].max()
        ymin=aoi[:,2].min()
        ymax=aoi[:,2].max()
        recon_aoi = recon[:, xmin:xmax + 1, ymin:ymax + 1]
        pred_aoi = pred[:, xmin:xmax + 1, ymin:ymax + 1]

        mse_aoi=F.mse_loss(recon_aoi, pred_aoi)
        mses_aoi += mse_aoi

    mse_aoi = mses_aoi/len(recons)
    return mse_aoi

def stretch_diff(batch_diff):
    batch_capsule = torch.zeros((batch_diff.shape[0],batch_diff.shape[1],batch_diff.shape[2]*2,batch_diff.shape[3]*2))
    for i,diff in enumerate(batch_diff): # batch_diff.shape = (32,1,16,16)
        diff = diff.squeeze()
        mask = torch.triu(torch.ones_like(diff), diagonal=1)

        # Apply the mask to select the upper triangle
        upper_triangle = mask*3
        lower_triangle = (1 - mask)

        capsule = torch.zeros((diff.shape[0]*2,diff.shape[1]*2))
        capsule[:16,:16] = diff
        capsule[:16,16:] = lower_triangle
        capsule[16:,:16] = upper_triangle

        batch_capsule[i] = capsule
    
    return batch_capsule

def adjusted_mse(batch_img, batch_recon):
    mask = (torch.Tensor(batch_img) != 0.00312).float()
    num_valid_pixels = torch.sum(mask)
    squared_diff = (batch_img - batch_recon).to(device)**2 * mask.to(device)
    mse = torch.sum(squared_diff) / (num_valid_pixels)

    return mse 

criterion = VGGPerceptualLoss().to(device)

def play(dataloader:DataLoader=None,
         tracker:LossTracker=None,
         v2lr:nn.Module=None,
         ssim=None,optimizer=None):
    
    v2lr.train() if tracker.job=="Training" else v2lr.eval()
    if tracker.best_epoch == -1 and tracker.job == "Training":
        print("Ready to TRAIN!!")
    for i, (batch_diff, batch_img) in enumerate(dataloader):
        # batch_diff = stretch_diff(batch_diff)
        batch_diff = batch_diff.to(device)
        batch_img = batch_img.to(device)
        batch_mapped, batch_lr, batch_recon_v = v2lr(batch_diff,batch_img)

        # scale_to_input removed from here
        mse_loss = F.mse_loss(batch_img, batch_recon_v)
        # ssim_value = (ssim(batch_img, batch_recon_v) + 1) / 2
        ssim_value = 1 - ssim(batch_img, batch_recon_v) 
        # mse_aoi = compute_mse_aoi(batch_img,batch_recon_v)
        # loss = 0.5*mse_aoi + 10*mse_loss + 2*ssim_value
        # loss = (mse_loss / (ssim_value+1e-6))
        mse_loss_lr = F.mse_loss(batch_lr, batch_mapped)
        
        # l1_reg = torch.tensor(0., requires_grad=True)
        # for param in v2lr.parameters():
        #     l1_reg = l1_reg + torch.norm(param, 1)

        # Add L1 regularization to the loss
        # loss = mse_aoi + mse_loss_lr + 1e-2*l1_reg
        # pcc=torch.cat([batch_img.reshape(-1), batch_recon_v.reshape(-1)]).reshape(-1,576*batch_size)
        # pcc=torch.corrcoef(pcc)[0,1]

        # loss = ((1-pcc) + ssim_value)/2
        vgg_loss = criterion(batch_img,batch_recon_v)
        # adj_mse = adjusted_mse(batch_img,batch_recon_v)
        
        # max_img=torch.max(batch_img.view(-1,576), dim=1)[0]
        # max_recon=torch.max(batch_recon_v.view(-1,576), dim=1)[0]
        # min_img=torch.min(batch_img.view(-1,576), dim=1)[0]
        # min_recon=torch.min(batch_recon_v.view(-1,576), dim=1)[0]

        # max_loss = torch.mean(torch.square(max_img - max_recon))
        # min_loss = torch.mean(torch.square(min_img - min_recon))
        
        # loss = vgg_loss+10*mse_loss  + max_loss
        loss = vgg_loss+mse_loss
    
        
        if tracker.job == "Training": # batch
            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

        # tracker.mse_loss = mse_loss.item()
        # tracker.ssim = ssim_value.item()
        # tracker.loss = loss.item()
        # tracker.mse_loss_lr = mse_loss_lr.item()

        tracker.epoch_mse_loss += mse_loss.item()
        tracker.epoch_ssim += ssim_value.item()
        tracker.epoch_loss += loss.item()
        tracker.epoch_loss_lr += mse_loss_lr.item()

    tracker.epoch_mse_loss /= (i+1)
    tracker.epoch_ssim /= (i+1)
    tracker.epoch_loss /= (i+1)
    tracker.epoch_loss_lr /= (i+1)

    # epoch rebackprog removed from here
    return tracker

idk=[-1]
for i in enumerate(idk):

    nownow = datetime.now()
    id_config = f"{CONDUCTANCE_VALUES}{ID}.{nownow.strftime('%Y%m%d%H%M%S%f')[:14]}"
    LOSS_TRACKER_PATH = f'./results/loss_tracker_{TASK}.csv'
    MODEL_STATEDICT_SAVE_PATH = f"./models/{TASK}/{id_config}_{TASK}.pth"
    MODEL_SAVE_PATH = MODEL_STATEDICT_SAVE_PATH[:-1] # pt instead of pth

    v2lr = V2ImgLR(recon_path,train_recon=True)
    # v2lr.v2lr = config
    # v2lr = torch.load("./models/v2lr/0.alpha.01.20231206000636_v2lr.pt")
    # v2lr = torch.load("./models/v2lr/1.5.vgg.b.1.20231221023841_v2lr.pt",map_location=device)
    v2lr = v2lr.to(device)
    # for d in v2lr.recon.decoder.parameters():
    #     d.requires_grad = False

    # for v in v2lr.v2lr.parameters():
    #     v.requires_grad = True
    summary(v2lr.v2lr,(1,16,16))
    # print(v2lr.recon.decoder)
    summary(v2lr.recon.decoder,(216))
    # summary(v2lr,(1,16,16))
    
    ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)
    
    optimizer_config = {'Adam': {'learning_rate':1e-3, 'weight_decay':0}}    
    optimizer = optimizer_build(optimizer_config,v2lr)

    best_v2lr = deepcopy(v2lr)
    min_loss = np.inf
    best_epoch = 0

    train_losses = []
    last_printed = 0
    tolerance = 3
    for epoch in range(epochs):
        trainer = play(train_dataloader,trainer,v2lr,ssim,optimizer)
        train_losses.append(trainer.epoch_mse_loss)
        
        with torch.no_grad():
            validator =play(val_dataloader,validator,v2lr,ssim)

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
            last_printed = epoch

            # note that this is not the best epoch
            trainer.best_epoch = epoch
            validator.best_epoch = epoch
            print(f"Tolerance: {tolerance}!! {trainer} !==! {validator}")
            tolerance -= 1
            
        if tolerance == 0:
            break

    del v2lr
    v2lr = deepcopy(best_v2lr)
    with torch.no_grad():
        tester = play(test_dataloader,tester,v2lr,ssim)
        
        train_losses.append(tester.epoch_loss)
        print(tester)

    torch.save(v2lr, MODEL_SAVE_PATH) # this saves the model as-is
    print(f"written to: {MODEL_SAVE_PATH}")

    with open(LOSS_TRACKER_PATH, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(train_losses)

    print(f"written to: {LOSS_TRACKER_PATH}")
    end = time.time()
    print(f"Elapsed time: {end - start} seconds.")
    start = end