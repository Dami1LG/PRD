from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging

#import PIL.Image
#from PIL import Image
import PIL
from torchvision.transforms import  Resize
import torchvision
import torchvision.transforms
from PIL import Image
from torch.autograd import Variable
import pandas


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
    

    
    
     # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    
    def convertMask(self, image):
        (l, h) = image.size
        matrice = np.zeros((l, h))
        for y in range(h):
            for x in range(l):
                (r,g,b) = image.getpixel((x, y))
                
                if (r,g,b)==(0,0,255):
                    matrice[x, y] = 1
                elif (r,g,b)==(255,127,0):
                    matrice[x, y] = 2
                elif (r,g,b)==(255,0,0):
                    matrice[x, y] = 3  
                elif (r,g,b)==(255,255,0):
                    matrice[x, y] = 2 
                elif (r,g,b)==(0,127,0):
                    matrice[x, y] = 2 
                
        return matrice
    
    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = PIL.Image.open(mask_file[0])
        img = PIL.Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

     

        img = self.preprocess(img, self.scale)
        #mask = self.preprocess(mask, self.scale)
        mask = self.convertMask(mask)
        
        cropped_image = img[0:128, 0:128]
        cropped_mask = mask[0:128, 0:128]
        
        #rdm = RandomCrop((128,128))
        nimg, nmask = rdm({'image': img, 'landmarks': mask})
        print("rdm", nimg.size)
        
        #cropped_image = torchvision.transforms.CenterCrop(128)
        #target = Resize((128,128), Image.NEAREST)
        
        
        
        #cropped_image.unsqueeze(0) 
        
        
        print("\nmask", mask.size)
        print("img", img.size)
        
        #torchvision.transforms.FiveCrop(img.size)
        
        return {'image': torch.from_numpy(cropped_image), 'mask': torch.from_numpy(cropped_mask)}
        
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        print("test")
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        print("h",h)
        print("w",w)
        new_h, new_w = self.output_size
        
        

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

