import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from tqdm import tqdm

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class StatsDataset(Dataset):
    def __init__(self, list_path, size=416):
        self.transform = transforms.Compose([
            # TODO circumvent stretching of image
            transforms.Resize(size=(size,size)),
            transforms.ToTensor()
                ])
        
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
            
        self.img_size = size
        self.batch_count = 0
    
    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
#         print(img.shape)
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        return img

    def __len__(self):
        return len(self.img_files)
    
    
class MeanStd():
    def __init__(self, train_path, size=416, batch_size=750):
        self.dataset = StatsDataset(train_path)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=7)
        self.pop_mean = []
        self.pop_std = []
        
    def _get_means(self):
        for inputs in tqdm(self.loader):
            numpy_image = inputs.numpy()
            batch_mean = np.mean(numpy_image, axis=(0,2,3))
            self.pop_mean.append(batch_mean)
        self.pop_mean = np.array(self.pop_mean).mean(axis=0)
        
    def _get_stds(self):
        # TODO use self.mean to calc std
        for inputs in tqdm(self.loader):
            numpy_image = inputs.numpy()
            batch_std = np.std(numpy_image, axis=(0,2,3))
            self.pop_std.append(batch_std)
        self.pop_std = np.array(self.pop_std).mean(axis=0)
        
    def get_means_stds(self):
        self._get_means()
        self._get_stds()
        return self.pop_mean, self.pop_std


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416, means=[.5,.5,.5], stds=[.5,.5,.5]):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        # TODO why is there no convert('RGB') in ImageFolder
        img = Image.open(img_path)
        img = transform(img).convert('RGB')
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
    
        # TRANSFORM
        
        return img_path, img

    def __len__(self):
        return len(self.files)



        
class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True, means=[.5,.5,.5], stds=[.5,.5,.5]):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("image_yolo","label_yolo").replace("images", "labels").replace("data", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transforms.Compose([
#             transforms.ToPILImage(),
            transforms.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
                                            ])

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        # my_trans = transforms.Compose([
        #     transforms.CenterCrop(10),
        #     transforms.ToTensor(),
        #     ])

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        
        # Extract image as PyTorch tensor
        # TODO why is there no convert('RGB') in ImageFolder
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        # TRANSFORMS
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        # img = my_trans(img)
        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
