import imgaug.augmenters as iaa
import numpy as np
from skimage import io
import argparse

def rotate_and_flip(cou_list,imgset,labelset,count):

    img_list = []
    label_list = []

    for k in range(len(cou_list)):
        img = imgset[cou_list[k]:cou_list[k]+count,:,:,:]
        label = labelset[cou_list[k]:cou_list[k]+count,:,:,:]
        nwimg = np.pad(img,((0,0),(128,128),(0,0),(0,0)))
        nwlabel = np.pad(label,((0,0),(128,128),(0,0),(0,0)))
        
        img_list.append(nwimg)
        label_list.append(nwlabel)
    
    img_aug = []
    label_aug = []

    for k in range(len(img_list)):
        img = img_list[k]
        label = label_list[k]
        
        for aug_count in range(8):
            nwimg = np.rot90(img,aug_count,(1,2))
            nwlabel = np.rot90(label,aug_count,(1,2))
            
            if aug_count > 3:
                nwimg = np.flip(nwimg,1)
                nwlabel = np.flip(nwlabel,1)
            
            img_aug.append(nwimg)
            label_aug.append(nwlabel)
            img_aug.append(np.flip(nwimg,0))
            label_aug.append(np.flip(nwlabel,0))
    return img_aug, label_aug

def piece_affine_argument(frames,labels):

    aug_seq = iaa.Sequential([iaa.Sometimes(0.3,iaa.Affine(rotate =[-45,-30,-22.5,22.5,30,45])),
                              iaa.Sometimes(0.8,iaa.PiecewiseAffine(scale = (0.01,0.05)))])
    
    cou = len(labels)
    for i in range(cou):
        print(i)
        IAA = aug_seq.to_deterministic()
        ter_frame = frames[i].copy()
        ter_label = labels[i].copy()
        
        ter_frame2 = ter_frame.copy()
        ter_label2 = ter_label.copy()
    
        ter_frame[0] = IAA(image = ter_frame[0])
        ter_label[0] = IAA(image = ter_label[0])
        
        ter_frame2[0] = IAA(image = ter_frame2[0])
        ter_label2[0] = IAA(image = ter_label2[0])

        frames.append(ter_frame)
        labels.append(ter_label)
        
        frames.append(ter_frame2)
        labels.append(ter_label2)
    frame_aug = (np.stack(frames,axis = 0)).transpose(0,1,4,2,3).astype(np.float32)
    label_aug = (np.stack(labels,axis = 0)).transpose(0,1,4,2,3).astype(np.float32)
    print(len(frame_aug))

    return frame_aug,label_aug

def pad(img_stack,label_stack,args):
    img_padding = np.zeros((165+args.channels-1,768,1024,1))
    label_padding = np.zeros((165+args.channels-1,768,1024,1))

    img_padding[args.channels//2:(args.channels//2)+165] = img_stack
    label_padding[args.channels//2:(args.channels//2)+165] = label_stack

    for i in range(args.channels//2):
        img_padding[i:i+1] = img_stack[0:1]
        img_padding[165+(args.channels//2)+i:166+i+(args.channels//2)] = img_stack[164:165]

        label_padding[i:i+1] = label_stack[0:1]
        label_padding[165+(args.channels//2)+i:166+i+(args.channels//2)] = label_stack[164:165]
    
    return img_padding, label_padding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default= 'surpevised',help='Train Mode')
    parser.add_argument('--slices', type=int, default= 2,help='surpevised training slices')
    parser.add_argument('--channels', type=int, default= 15,help='input channels')
    args = parser.parse_args()

    if args.stage == 'surpevised':

        img_path = './data/train_data/img/training.tif'
        label_path = './data/train_data/label/training_groundtruth.tif'

        img_stack = (io.imread(img_path)[:,:,:,np.newaxis]/255).astype(np.float32)
        label_stack = (io.imread(label_path)[:,:,:,np.newaxis]/255).astype(np.float32)
        cou_list = []

        for k in range(0,160,160//args.slices):
            cou_list.append(k)

        img, label = rotate_and_flip(cou_list,img_stack,label_stack,1)
        frame_aug, label_aug = piece_affine_argument(img,label)

        for i in range(label_aug.shape[0]):
            st = str(i).zfill(4)
            io.imsave('./dataset/aug/img/'+st+'.tif',frame_aug[i])
            io.imsave('./dataset/aug/label/'+st+'.tif',label_aug[i])


    if args.stage == 'semi-surpevised':
        img_path = "./data/SEG_result/train_img/space_img_32.tif"  
        label_path = "./data/SEG_result/train_label/space_label_32.tif"

        img_stack = io.imread(img_path).transpose(0,2,3,1) #THWC
        label_stack = (io.imread(label_path).transpose(0,2,3,1)/255).astype(np.float32) #THWC
        
        img_stack,label_stack = pad(img_stack,label_stack,args)

        cou_list = []

        for k in range(0,165,40):
            cou_list.append(k)
        
        img, label = rotate_and_flip(cou_list,img_stack,label_stack,args.channels)
        frame_aug = (np.stack(img,axis = 0)).transpose(0,1,4,2,3).astype(np.float32)
        label_aug = (np.stack(label,axis = 0)).transpose(0,1,4,2,3).astype(np.float32)

        for i in range(label_aug.shape[0]):
            st = str(i).zfill(4)
            io.imsave('./dataset/SCM_aug/img/'+st+'.tif',frame_aug[i])
            io.imsave('./dataset/SCM_aug/label/'+st+'.tif',label_aug[i])
