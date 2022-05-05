# Semi-Supervised Segmentation of Cell Image Stacks for Electron Microscopy
This repository contains source code of Semi-supervised Segmentation for EM images ([ISBI 2022](https://ieeexplore.ieee.org/document/9761519)).

![image](https://github.com/cbmi-group/MPP/blob/main/img.jpg)

## Getting Started
### Install Dependencies
```
pip install -r requirements.txt
```
### Prepare Dataset
Please place the 3D training image stack and labels in ./data/train_data/ and test image stack and labels in ./dataset/test_data/

### Training
Due to memory constraints, we use offline augmentation, as follows:
```
python _augment.py --stage surpevised
```
Augmented images and labels are placed in ./dataset/aug/. The number of image slices can be adjusted as needed.



Then, use augmented images to train segmentation network.
```
python train.py --stage surpevised
```


Segment the entire training image stack using the trained network.
```
python inference.py --stage surpevised
```
The segmentation result is placed in ./data/SEG_result/train_img/


Then, Use MPP:

```
python image_monography.py
```
Result is placed in ./data/SEG_result/train_label/



Augment images and labels in ./data/SEG_result/train_img/ and ./data/SEG_result/train_label/
```
python _augment.py --stage semi-surpevised
```
The number of Z-axis slices can be adjusted as needed, augmented images and labels are placed in ./dataset/SCM_aug/img and ./dataset/SCM_aug/label

Train SCM:

```
python train.py --stage semi-surpevised
```
The scm training is exactly the same as the segmentation network, the only difference is the number of input channels.

### Testing
```
python test_Unet.py
```
The coarse segmentation result is placed in ./dataset/SEG_result/test_label/stack.tif

Then use:
```
python test_space_Unet.py
```

The segmentation result is placed in './data/SCM_result/test_label/stack.tif'
