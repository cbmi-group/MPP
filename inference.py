import argparse
import numpy as np
import skimage
from skimage import io
import torch
import pytorch_lightning as pl
from _augment import pad
from model.unet import UNet


parser = argparse.ArgumentParser(description='Inference(Prediction) Script')

parser.add_argument('--stage', default='semi-supervised',
                    help='training stage (supervised or semi-supervised)')
parser.add_argument('--checkpoint', default="",
                    help='checkpoint path')
# model configs:
parser.add_argument('--channels', default=15, type=int,
                    help='input channels for SCM network')


class MODEL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(1,1)
            
    def forward(self, x):
        return self.model(x)

class MODEL2(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(15,1)
            
    def forward(self, x):
        return self.model(x)

args = parser.parse_args()

if args.stage == 'supervised':
    net = MODEL().load_from_checkpoint(args.checkpoint)
if args.stage == 'semi-supervised':
    net = MODEL2().load_from_checkpoint(args.checkpoint)

net.eval()
net.cuda()

if args.stage == 'supervised':
    img_path = './data/train_data/img/training.tif'
    label_path = './data/train_data/label/training_groundtruth.tif'

    img_stack, label_stack = io.imread(img_path)[:,:,:,np.newaxis], io.imread(label_path)[:,:,:,np.newaxis]
    img_stack = np.pad(img_stack/255,((0,0),(128,128),(0,0),(0,0))).transpose(0,3,1,2) #1024*1024,TCHW 

    mask_stack = ((label_stack)//255).astype(np.float32).transpose(0,3,1,2)  #TCHW
    mask_stack = np.pad(mask_stack,((0,0),(0,0),(128,128),(0,0))) #1024*1024
    label_stack = (label_stack//255).astype(np.int32)  

if args.stage == 'semi-supervised':
    img_path = "./data/SEG_result/train_img/space_img_32.tif"
    label_path = "./data/SEG_result/train_label/space_label_32.tif"

    img_stack = io.imread(img_path).transpose(0,2,3,1) #THWC
    label_stack = (io.imread(label_path).transpose(0,2,3,1)/255).astype(np.float32) #THWC
    
    img_stack,label_stack = pad(img_stack,label_stack,args)
    print(img_stack.shape)
    img_stack = np.pad(img_stack,((0,0),(128,128),(0,0),(0,0))).transpose(3,0,1,2) #1024*1024,CTHW
    mask_stack = (label_stack).astype(np.float32).transpose(3,0,1,2)  #CTHW
    mask_stack = np.pad(mask_stack,((0,0),(0,0),(128,128),(0,0))) #1024*1024


frames_aug = []
frames = torch.from_numpy(img_stack).to(torch.float32).cuda()

# flip/rotation augmentation
for k in range(8):
    temp_frames = torch.rot90(frames,k,[2,3])
    if k >3:
        temp_frames = torch.flip(temp_frames,[2,3])
    frames_aug.append(temp_frames)

mask = torch.from_numpy(mask_stack).to(torch.float32).cuda()
batch_out_aug = []

with torch.no_grad():
    for k in range(8):
        batch_out = []
        for i in range(0,165):
            if args.stage == 'supervised':
                logits = net.model(frames_aug[k][i:i+1])
            if args.stage == 'semi-supervised':
                logits = net.model(frames_aug[k][0:1,i:i+args.channels])
            batch_out.append(logits)

        batch_out = torch.cat(batch_out, dim=0) # T, K, H, W
        batch_out = torch.rot90(batch_out,k,[3,2])
        if k >3:
            batch_out = torch.flip(batch_out,[3,2])
        batch_out = batch_out[:,:,128:896,:]
        batch_out = batch_out.detach().cpu().numpy()
        batch_out_aug.append(batch_out)

#Averaging and Binarization
sum_batch = 0
for i in range(8):
    sum_batch = batch_out_aug[i]+sum_batch
batch_out = (sum_batch/8).astype(np.float32)
batch_out[batch_out>0.2] = 1
batch_out[batch_out<=0.2] = 0
batch_out = skimage.img_as_bool(batch_out,)
batch_out = batch_out.astype(np.float32)
if args.stage == 'supervised':
    predict_path = "./data/SEG_result/train_img/space_img_32.tif"
if args.stage == 'semi-supervised':
    predict_path = "./data/SCM_result/train_img/space_img_32.tif"
io.imsave(predict_path,batch_out)