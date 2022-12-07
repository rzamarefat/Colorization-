import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from glob import glob
import cv2

class ColorizationDataset():
    def __init__(self, images, input_size=(256, 256), is_train=True):
        self.images = images
        self.input_size = input_size

        self.is_train = is_train

        if self.is_train:
            self.train_transforms = T.Compose([
                T.ToTensor(),
                T.Resize(self.input_size), 
            ])
        else:
            self.test_transforms = T.Compose([
                T.ToTensor(),
                T.Resize(self.input_size), 
            ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #grayscale_img = cv2.merge([grayscale_img, grayscale_img, grayscale_img])
        
        if self.is_train:
            img = self.train_transforms(img)
            grayscale_img = self.train_transforms(grayscale_img)
            
        else:
            img = self.test_transforms(img)
            grayscale_img = self.test_transforms(grayscale_img)

        return img, grayscale_img

if __name__ == "__main__":
    root_path_to_images = "/home/marefat/projects/NSFW_zip_files_dataset/INHOUSE/neutral/*"
    images = [f for f in sorted(glob(root_path_to_images))]
    ds = ColorizationDataset(images)

    

        