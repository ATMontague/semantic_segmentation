from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import random
import glob
import os
import torch
import numpy as np


class FreiburgForestDataset(Dataset):
    """ Freiburg Forest dataset """

    def __init__(self, image_path, mask_path, transform_images=False, encode=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_path, '*')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_path, '*')))
        self.transform_images = transform_images
        self.encode = encode
        self.num_classes = 6

    def transform(self, image, mask):
        """
        Transform the given image and mask with the exact same set of transformations.
        """

        if self.transform_images:

            # randomly perform horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # randomly shear
            if random.random() > 0.5:
                # todo: figure out how to handle new pixels after shear. utilize 'fill'
                image = TF.affine(image, angle=0, translate=[0, 0], scale=1.0, shear=45.0)
                mask = TF.affine(mask, angle=0, translate=[0, 0], scale=1.0, shear=45.0)

            # random rotation
            if random.random() > 0.5:
                rand_rotation = random.randint(-13, 14)
                image = TF.affine(image, angle=rand_rotation, translate=[0, 0], scale=1.0, shear=0.0,
                                  fillcolor=0)
                mask = TF.affine(mask, angle=rand_rotation, translate=[0, 0], scale=1.0, shear=0.0,
                                 fillcolor=0)

            # random scaling. (scaling < 1.0 causes unnecessary border)
            if random.random() > 0.5:
                rand_scale = random.uniform(0.5, 2.0)
                image = TF.affine(image, angle=0, translate=[0, 0], scale=rand_scale, shear=0.0, fillcolor=0)
                mask = TF.affine(mask, angle=0, translate=[0, 0], scale=rand_scale, shear=0.0, fillcolor=0)

            # random vignetting (reduce pixel brightness on outside portion of image)
            if random.random() > 0.5:
                pass  # todo: implement vignetting

            # random cropping
            if random.random() > 0.5:
                # todo: set parameters according to paper
                i_loc = random.randint(0, 100)
                j_loc = random.randint(0, 100)
                w_size = random.randint(500, 600)
                h_size = random.randint(200, 400)
                image = TF.resized_crop(image, i=i_loc, j=j_loc, h=h_size, w=w_size, size=(384, 768))
                mask = TF.resized_crop(mask, i=i_loc, j=j_loc, h=h_size, w=w_size, size=(384, 768))

            # random brightness
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.6, 1.4))

            # random contrast
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.5, 1.5))

        if self.encode:
            mask = self.rgb_to_class(mask)

        resize = transforms.Resize(size=(250, 250))
        image = resize(image)
        mask = resize(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def rgb_to_class(self, mask):
        """
        Convert RGB mask image to a single channel ID image, ie map pixel color to class ID
        """

        # [170, 170, 170] -> 170  (1) Road
        # [0, 255, 0]     -> 150  (2) Grass
        # [102, 102, 51]  -> 96   (3) Vegetation
        # [0, 60, 0]      -> 35   (3) Tree
        # [0, 120, 255]   -> 100  (4) Sky
        # [0, 0, 0]       -> 0    (5) Obstacle

        grayscale = mask.convert('L')
        grayscale = np.array(grayscale)

        grayscale[grayscale == 170] = 1
        grayscale[grayscale == 150] = 2
        grayscale[grayscale == 96] = 3
        grayscale[grayscale == 35] = 3
        grayscale[grayscale == 100] = 4
        grayscale[grayscale == 0] = 5

        gray_mask = Image.fromarray(grayscale)

        return gray_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Read the given image based off of the index passed to this function.
        """

        image = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])
        image, mask = self.transform(image, mask)
        return image, mask
