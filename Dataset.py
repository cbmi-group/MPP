import os
from torch.utils.data.dataset import Dataset
from skimage import io


# Standard segmentation network dataset
class MyDataset(Dataset):
    def __init__(self):
        self.img_path = './dataset/aug/img/'
        self.label_path = './dataset/aug/label/'
        self.img_list = os.listdir(self.img_path)
        self.label_list = os.listdir(self.img_path)

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.img_list[index])
        lbl_path = os.path.join(self.label_path, self.label_list[index])
        img = io.imread(img_path)[0]
        label = io.imread(lbl_path)[0]
        return img, label

# Spatial continuity network dataset 
class SCM_Dataset(Dataset):
    def __init__(self):
        self.space_img_path = "./dataset/SCM_aug/img/"
        self.label_path = "./dataset/SCM_aug/label/"
        self.space_img_list = os.listdir(self.space_img_path)
        self.label_list = os.listdir(self.label_path)

    def __len__(self):
        return len(self.space_img_list)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.space_img_path, self.space_img_list[index])
        lbl_path = os.path.join(self.label_path, self.label_list[index])
        img = io.imread(img_path)[:,0]
        label = io.imread(lbl_path)[7:8,0]
        return img, label
    
if __name__ == '__main__':
    disDataset = MyDataset()