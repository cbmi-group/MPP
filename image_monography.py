from skimage import data,exposure
from skimage import io
from skimage import morphology
import skimage
import numpy as np

#This method is for the EPFL dataset and takes a groundtruth every five slices

predict_path = "./data/SEG_result/train_img/space_img_32.tif"
label_path = './data/train_data/label/training_groundtruth.tif'
save_path = "./data/SEG_result/train_label/space_label_32.tif"
predict_stack = io.imread(predict_path)[:,0]
label_stack = (io.imread(label_path)//255)

lis_frames = []
batch_out = []

predict_stack = predict_stack.astype(np.bool_)
label_stack = label_stack.astype(np.bool_)


for i in range(0,160,5):
    lis_frames.append(i)
    
for i in range(0,165):
    predict_stack[i] = morphology.remove_small_objects(predict_stack[i],min_size=300)

for i in range(0,165):
    if i in lis_frames:
        logits2 = label_stack[i]
    else:
        logits2 = predict_stack[i]
    batch_out.append(logits2.astype(np.bool))
batch_out = np.stack(batch_out,0)
batch_out = batch_out[:,np.newaxis,:,:]
batch_out.astype(np.bool_)

def erosion_fn(batch_out):
    for i in range(155,0,-1):
        if i%5 == 0 and i>0:
            logits = batch_out[i,0]
        else:
            logits = np.pad(logits, ((20,20),(20,20)), mode='edge')
            for j in range(3):
                logits = morphology.erosion(logits)
            logits = logits[20:788,20:1044]
            logits = skimage.img_as_bool(logits)
            logits = morphology.remove_small_objects(logits,min_size=450)
            batch_out[i,0,:,:] = np.logical_or(logits,batch_out[i,0,:,:])
        
    for i in range(0,160):
        if i%5 == 0:
            logits = batch_out[i,0]
        else:
            logits = np.pad(logits, ((20,20),(20,20)), mode='edge')
            for j in range(3):
                logits = morphology.erosion(logits)
            logits = logits[20:788,20:1044]
            logits = skimage.img_as_bool(logits)
            logits = morphology.remove_small_objects(logits,min_size=450)
            batch_out[i,0,:,:] = np.logical_or(logits,batch_out[i,0,:,:])

    return batch_out

batch_out = erosion_fn(batch_out)


def spatial_context_fn(batch_out):
    for i in range(155):
        if i %5 !=0:
            front_slice = batch_out[5*(i//5),0,:,:]
            back_slice = batch_out[5*((i//5)+1),0,:,:]
            
            AND = np.logical_or(front_slice,back_slice)
            batch_out[i,0,:,:] = np.logical_and(AND,batch_out[i,0,:,:])
            batch_out[i,0,:,:] = morphology.remove_small_objects(batch_out[i,0,:,:],min_size=300)

    for i in range(155):
        if i %5 !=0:
            front_slice = batch_out[5*(i//5),0,:,:]
            back_slice = batch_out[5*((i//5)+1),0,:,:]

            OR = np.logical_and(front_slice,back_slice)
            batch_out[i,0,:,:] = np.logical_or(OR,batch_out[i,0,:,:])
            batch_out[i,0,:,:] = morphology.remove_small_objects(batch_out[i,0,:,:],min_size=300)

    for i in range(165):
        if i not in lis_frames:
            if i>=1 and i<164:
                AND = np.logical_or(batch_out[i-1,0,:,:],batch_out[i+1,0,:,:])
                batch_out[i,0,:,:] = np.logical_and(AND,batch_out[i,0,:,:])
            batch_out[i,0,:,:] = morphology.remove_small_objects(batch_out[i,0,:,:],min_size=300)
            
    for i in range(165):
        if i not in lis_frames:
            if i>=1 and i<164:
                OR = np.logical_and(batch_out[i-1,0,:,:],batch_out[i+1,0,:,:])
                batch_out[i,0,:,:] = np.logical_or(OR,batch_out[i,0,:,:])
            batch_out[i,0,:,:] = morphology.remove_small_objects(batch_out[i,0,:,:],min_size=300)
    for i in range(165):
        if i not in lis_frames:
            if i>=2 and i<163:
                AND = np.logical_or(batch_out[i-2,0,:,:],batch_out[i+2,0,:,:])
                batch_out[i,0,:,:] = np.logical_and(AND,batch_out[i,0,:,:])
            batch_out[i,0,:,:] = morphology.remove_small_objects(batch_out[i,0,:,:],min_size=300)
            
    for i in range(165):
        if i not in lis_frames:
            if i>=2 and i<134:
                OR = np.logical_and(batch_out[i-2,0,:,:],batch_out[i+2,0,:,:])
                batch_out[i,0,:,:] = np.logical_or(OR,batch_out[i,0,:,:])
            batch_out[i,0,:,:] = morphology.remove_small_objects(batch_out[i,0,:,:],min_size=300)

    return batch_out
            
batch_out = erosion_fn(batch_out)
batch_out = spatial_context_fn(batch_out)

#Save
batch_out_x = 255*(batch_out).astype(np.uint8)
io.imsave(save_path,batch_out_x)