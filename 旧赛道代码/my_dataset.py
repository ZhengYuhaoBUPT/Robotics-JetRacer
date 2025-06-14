import torch
import os
import glob
import uuid
import PIL.Image
import torch.utils.data
import subprocess
import cv2
import numpy as np


class MY_Dataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, random_hflip=False):
        super(MY_Dataset, self).__init__()
        self.directory = directory
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.refresh()
        self.random_hflip = random_hflip
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        
        x = 2.0 * (ann['x'] / width - 0.5) # -1 left, +1 right
        y = 2.0 * (ann['y'] / height - 0.5) # -1 top, +1 bottom
        speed_Up = ann['speed_Up']
        over = ann['over']
        stop = ann['stop']
        straight = ann['straight']
        
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            x = -x
            
        return image, torch.Tensor([x, y,speed_Up,over,stop,straight])
    
    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split('_')
        x = items[0]
        y = items[1]
        speed_Up = items[2]
        over = items[3]
        stop = items[4]
        straight = items[5]
        return int(x), int(y),int(speed_Up),int(over),int(stop),int(straight)
        
    def refresh(self):
        self.annotations = []
        for image_path in self.image_paths:
            x, y, speed_Up, over, stop, straight = self._parse(image_path)
            self.annotations += [{
                'image_path': image_path,
                'x': x,
                'y': y,
                'speed_Up':speed_Up,
                'over':over,
                'stop':stop,
                'straight':straight
            }]
        
    def save_entry(self, category, image, x, y, speed_Up, over, stop, straight):
        category_dir = os.path.join(self.directory, category)
        if not os.path.exists(category_dir):
            subprocess.call(['mkdir', '-p', category_dir])
            
        filename = '%d_%d_%s.jpg' % (x, y, str(uuid.uuid1()))
        
        image_path = os.path.join(category_dir, filename)
        cv2.imwrite(image_path, image)
        self.refresh()
        


class HeatmapGenerator():
    def __init__(self, shape, std):
        self.shape = shape
        self.std = std
        self.idx0 = torch.linspace(-1.0, 1.0, self.shape[0]).reshape(self.shape[0], 1)
        self.idx1 = torch.linspace(-1.0, 1.0, self.shape[1]).reshape(1, self.shape[1])
        self.std = std
        
    def generate_heatmap(self, xy):
        x = xy[0]
        y = xy[1]
        heatmap = torch.zeros(self.shape)
        heatmap -= (self.idx0 - y)**2 / (self.std**2)
        heatmap -= (self.idx1 - x)**2 / (self.std**2)
        heatmap = torch.exp(heatmap)
        return heatmap





class roadlineDataset(torch.utils.data.Dataset):
    def __init__(self, directory, categories, transform=None, random_hflip=False):
        super(roadlineDataset,self).__init__()
        self.directory = directory
        self.categories = categories
        self.transform = transform
        self.refresh()
        self.random_hflip = random_hflip
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        
        x = 2.0 * (ann['x'] / width - 0.5) # -1 left, +1 right
        y = 2.0 * (ann['y'] / height - 0.5) # -1 top, +1 bottom
        
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            x = -x
            
        return image, ann['category_index'], torch.Tensor([x, y])