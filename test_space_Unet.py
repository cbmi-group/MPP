import numpy as np
import skimage
from skimage import io
import torch
import pytorch_lightning as pl

from model.unet import UNet
from utils import calculate_IoU

class MODEL(pl.LightningModule):
    def __init__(self):
        super(MODEL, self).__init__()
        self.model = UNet(15,1)

    def forward(self, x):
        return self.model(x)

img_path = './data/SEG_result/test_label/stack.tif'
label_path = "./data/test_data/label/testing_groundtruth.tif"
predict_path = './data/SCM_result/test_label/stack.tif'

img_stack = io.imread(img_path).astype(np.int32) #TCHW
label_stack = io.imread(label_path)[:,:,:,np.newaxis] #TCHW
img_stack = np.pad(img_stack,((0,0),(0,0),(128,128),(0,0)))

mask_stack = (label_stack//255).astype(np.float32).transpose(0,3,1,2)
mask_stack = np.pad(mask_stack,((0,0),(0,0),(128,128),(0,0)))

label_stack = (label_stack//255).astype(np.int32)

batch_out = frames = torch.from_numpy(img_stack).to(torch.float32).cuda()
mask = torch.from_numpy(mask_stack).to(torch.float32).cuda()

#Padding
new_batch_out = torch.zeros((179,1,1024,1024)).cuda()
new_batch_out[7:172] = batch_out
for i in range(7):
    new_batch_out[i:i+1] = batch_out[0:1]
    new_batch_out[172+i:172+i+1] = batch_out[164:165]

parser.add_argument('--checkpoint', default=".",
                    help='checkpoint path')
args = parser.parse_args()

net = MODEL().load_from_checkpoint(args.checkpoint)
net.eval()
net.cuda()
new_batch_out.cuda()
new_batch_out = new_batch_out.permute(1,0,2,3) #CTHW

#Multiple flip/rotation
new_batch_out_aug = []
new_label_all = []
for k in range(8):
    temp_frames = torch.rot90(new_batch_out,k,[2,3])
    if k >3:
        temp_frames = torch.flip(temp_frames,[2,3])
    new_batch_out_aug.append(temp_frames)

#Test with augmentation
with torch.no_grad():
    for k in range(8):
        new_label = []
        for i in range(0,165):
            lab = net(new_batch_out_aug[k][:,i:i+15])
            new_label.append(lab)
        new_label = torch.cat(new_label,dim = 0)
        new_label = torch.rot90(new_label,k,[3,2])
        if k >3:
            new_label = torch.flip(new_label,[3,2])
        new_label_all.append(new_label)


#Averaging and Binarization
sum_label = 0
for i in range(8):
    sum_label = sum_label+new_label_all[k]
new_label = sum_label/8
batch_out = new_label[:,:,128:896,:]
batch_out = batch_out.detach().cpu().numpy()
batch_out[batch_out>0.2] = 1
batch_out[batch_out<=0.2] = 0
batch_out = skimage.img_as_bool(batch_out,)
batch_out = batch_out.astype(np.uint32)
io.imsave(predict_path,batch_out)

#Calculate IoU
calculate_IoU(batch_out,label_stack)