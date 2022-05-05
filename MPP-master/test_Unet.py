import numpy as np
import torch
from skimage import io
import pytorch_lightning as pl
from model.unet import *
import skimage
from utils import calculate_IoU

class MODEL(pl.LightningModule):
    def __init__(self):
        super(MODEL, self).__init__()
        self.model = UNet(1,1)

    def forward(self, x):
        return self.model(x)
parser.add_argument('--checkpoint', default="",
                    help='checkpoint path')
args = parser.parse_args()

torch.cuda.set_device(0)
net = MODEL().load_from_checkpoint(args.checkpoint)

net.eval()
net.cuda()

#input and mask
img_path = "./data/test_data/img/testing.tif"
label_path = "./data/test_data/label/testing_groundtruth.tif"

img_stack, label_stack = io.imread(img_path)[:,:,:,np.newaxis], io.imread(label_path)[:,:,:,np.newaxis]
img_stack = np.pad(img_stack/255,((0,0),(128,128),(0,0),(0,0))).transpose(0,3,1,2) #1024*1024,TCHW 

mask_stack = ((label_stack)//255).astype(np.float32).transpose(0,3,1,2)  #TCHW
mask_stack = np.pad(mask_stack,((0,0),(0,0),(128,128),(0,0))) #1024*1024
label_stack = (label_stack//255).astype(np.int32)  
label_stack = (label_stack//255).astype(np.int32)
frames_aug = []
frames = torch.from_numpy(img_stack).to(torch.float32).cuda()

#eight rotten
for k in range(8):
    temp_frames = torch.rot90(frames,k,[2,3])
    if k >3:
        temp_frames = torch.flip(temp_frames,[2,3])
    frames_aug.append(temp_frames)

mask = torch.from_numpy(mask_stack).to(torch.float32).cuda()
batch_out_aug = []

#Test
with torch.no_grad():
    for k in range(8):
        batch_out = []
        for i in range(0,165):
            logits = net.model(frames_aug[k][i:i+1])
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
predict_path = './data/SEG_result/test_label/stack.tif'
io.imsave(predict_path,batch_out)

#Calculate IoU
calculate_IoU(batch_out,label_stack)

