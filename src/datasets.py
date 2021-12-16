from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import albumentations as A
import glob
import os
import torch
import cv2
import numpy as np


# todo: add ability to stack depth as additional channel, so (r, g, b, d)
# todo: add ability to stack edge image as additional channel, so (r, g, b, e) where e is single channel edge image
# todo: make sure images are resized to appropriate shape when evaluating model


class SunRGBD(Dataset):
    """
    kv1:
        b3dodata:         555
        NYUdata:          1450
    kv2:
        align_kv2:        300
        kinect2data:      3486

    realsense:
        lg:               548
        sa:               440
        sh:               21
        shr:              151
    xtion:
        sun3ddata:        207
        xtion_align_data: 300

    total = 7458 images
    """

    def __init__(self, image_path, mask_path, height=384, width=768, num_classes=5, traversable=True,
                 transform_images=False):
        self.image_paths = image_path
        self.mask_paths = mask_path
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.traversable = traversable
        self.transform_images = transform_images

    def transform(self, image, mask):
        raise NotImplementedError

    def rgb_to_class(self, mask):
        """
        Convert RGB mask image to a single channel ID image, ie map pixel color to class ID
        :return:
        """

        grayscale = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        raise NotImplementedError

    def update_mask_labels(self, mask):
        # temporary fix to correct class labels
        mask[mask == 5] = 3
        mask[mask == 8] = 4

        # the resulting list of class is now [0, 1, 2, 3, 4, 5]
        # instead of [0, 1, 3, 5, 8]
        return mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.load(self.mask_paths[index])

        # temp fix
        mask = self.update_mask_labels(mask)

        if self.traversable:
            # reduce classes
            mask[mask > 2] = 0

        if self.transform_images:
            transformed = self.transform(image, mask)
            image = transformed['image']
            mask = transformed['mask']
        else:  # still need to make images correct size
            transformations = A.Compose(
                [
                    A.Resize(self.height, self.width),
                ]
            )
            transformed = transformations(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask


class TerrainData(Dataset):
    """
    A combination of Freiburg Forest and TAS500 for terrain classification.
    """

    def __init__(self, freiburg_imgs, freiburg_masks, tas_images, tas_masks, height, width, traversable=True):
        self.freiburg_imgs = sorted(glob.glob(os.path.join(freiburg_imgs, '*')))
        self.freiburg_masks = sorted(glob.glob(os.path.join(freiburg_masks, '*')))
        self.tas_images = sorted(glob.glob(os.path.join(tas_images, '*')))
        self.tas_masks = sorted(glob.glob(os.path.join(tas_masks, '*')))
        self.height = height
        self.width = width
        self.traversable = traversable
        self.combine()

    def combine(self):
        # combine both datasets and shuffle them
        self.image_path = self.freiburg_imgs + self.tas_images
        self.mask_path = self.freiburg_masks + self.tas_masks



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Read the given image based off of the index passed to this function.
        """

        # todo: handle class difference based on which dataset image is from
        pass


class TAS500Dataset(Dataset):
    """
    TAS500 dataset, a novel semantic segmentation dataset for autonomous driving in unstructured environments.

    Train: 440 images
    Val:   100 images
    Test:  100 images

    Key Classes for Terrain Classification:
        0: Asphalt,          RGB=(192, 192, 192)
        1: Gravel,           RGB=(105, 105, 105)
        2: Soil,             RGB=(160, 82, 45)
        3: Sand,             RGB=(244, 164, 96)
        4: Bush,             RGB=(60, 179, 113)
        5: Forest,           RGB=(34, 139, 34)
        6: Low Grass,        RGB=(154, 205, 0)
        7: High Grass,       RGB=(0, 128, 0)
        8: Misc. Vegetation, RGB=(0, 100, 0)
        9: Tree Crown,       RGB=(0, 250, 154)
        10: Tree Trunk,      RGB=(139, 69, 19)
    """

    def __init__(self, image_path, mask_path_id, mask_path_rgb, height=384, width=768, traversable=False, terrain=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_path, '*')))
        self.mask_paths_id = sorted(glob.glob(os.path.join(mask_path_id, '*')))
        self.mask_path_rgb = sorted(glob.glob(os.path.join(mask_path_rgb, '*')))
        self.height = height
        self.width = width
        self.traversable = traversable
        self.terrain = terrain

    def transform(self, image, mask):
        raise NotImplementedError

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Read the given image based off of the index passed to this function.
        """

        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask already has class labels, need to convert to single channel
        mask = cv2.imread(self.mask_paths_id[index])
        mask = mask[:, :, 0]

        # remove non-traversable classes
        # new set of classes = [0, 1, 2, 3, 4, 5] -> [asphalt, gravel, sand, soil, obstacle, null]
        if self.traversable:
            mask[mask == 255] = 5
            mask[mask >= 4] = 4

        # keep only classes similar to Freiburg Forest (make them 'obstacles')
        # new set of classes = [0, 1, 2, 3, 4, 5, 6, 7]
        # [asphalt, gravel, soil, sand, bush, grass, obstacle, null]
        elif self.terrain:
            mask[mask == 5] = 4  # combine forest and bush
            mask[mask == 7] = 6  # combine high grass and low grass
            mask[mask == 6] = 5  # shift class for grass from 6 to 5
            mask[mask == 8] = 4   # misc vegetation to bush
            mask[mask == 9] = 4  # tree crown to bush
            mask[mask == 10] = 4  # tree trunk to bush
            mask[mask == 255] = 7  # change null/undefined class from 255 to 7
            mask[mask >= 11] = 6  # the rest become 'obstacle'

        if not self.traversable and not self.terrain:
            mask[mask == 255] = 23

        transformations = A.Compose(
            [
                A.Resize(self.height, self.width),
            ]
        )
        transformed = transformations(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask


class FreiburgForestDataset(Dataset):
    """
    Freiburg Forest dataset
    Train has 230 images.
    Test has 136 images
    Class Distribution:
        Obstacle:         87/230 ~ 37.8%
        Road:            222/230 ~ 96.5%
        Grass:           230/230 = 100%
        Vegetation/Tree: 230/230 = 100%
        Sky:             230/230 = 100%
    """

    def __init__(self, image_path, mask_path, height=384, width=768, transform=None, encode=False, traversable=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_path, '*')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_path, '*')))
        self.transform = transform
        self.encode = encode
        self.num_classes = 6  # original number of classes
        self.height = height
        self.width = width
        self.traversable = traversable

    def transform(self, image, mask):
        """
        Transform the given image and mask with the exact same set of transformations.
        :param image:
        :param mask:
        :return: image, mask
        """
        # todo: add salt & pepper noise?
        # todo: add luminance?
        # todo: implement vignetting
        # todo: experiment with silhouettes
        # todo: randomly 'copy -> paste' obstacles in different parts of the image (if img doesnt have obstacle, add it)
        # todo: generate images that have less emphasis on texture...
        # todo: determine appropriate values for each transformation
        # todo: determine if probability should be same for all transformations
        # todo: look into how/why IAAAffine breaks mask
        # todo: add weather augmentations such as shadows and sun flair, fog
        # todo: shift images to the left/right more, so network doesn't think road always in the middle
        # todo: normalize?

        prob = 0.5

        transformations = A.Compose(
            [
                A.RandomCrop(height=300, width=700, p=prob),
                A.Resize(self.height, self.width),
                A.HorizontalFlip(p=prob),
                A.ColorJitter(brightness=0.05, contrast=0.0, saturation=0.0, hue=0.0, p=prob),
                A.ColorJitter(brightness=0.0, contrast=0.01, saturation=0.0, hue=0., p=prob),
                A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.01, hue=0.0, p=prob),
                A.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.01, p=prob),
                A.CoarseDropout(max_holes=10, max_height=50, max_width=50, min_holes=1, min_height=10,
                                min_width=10, mask_fill_value=255, p=prob),
                A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=0, p=prob),
                A.Blur(blur_limit=7, p=prob),
            ]
        )

        return transformations(image=image, mask=mask)

    def rgb_to_class(self, mask):
        """
        Convert RGB mask image to a single channel ID image, ie map pixel color to class ID
        """

        # [0.6667, 0.6667, 0.6667] -> [170, 170, 170] -> 170  (1) Road
        # [0.0000, 1.0000, 0.0000] -> [0, 255, 0]     -> 150  (2) Grass
        # [0.4000, 0.4000, 0.2000] -> [102, 102, 51]  -> 96   (3) Vegetation
        # []                       -> [0, 60, 0]      -> 35   (3) Tree
        # [0.0000, 0.4706, 1.0000] -> [0, 120, 255]   -> 100  (4) Sky
        # []                       -> [0, 0, 0]       -> 0    (5) Obstacle

        grayscale = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        grayscale[grayscale == 170] = 1
        grayscale[grayscale == 150] = 2
        grayscale[grayscale == 96] = 3
        grayscale[grayscale == 35] = 3
        grayscale[grayscale == 100] = 4
        grayscale[grayscale == 0] = 5
        grayscale[grayscale == 255] = 0

        # classes would be only: road, grass, null
        if self.traversable:
            grayscale[grayscale == 3] = 0
            grayscale[grayscale == 4] = 0
            grayscale[grayscale == 5] = 0

        return grayscale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Read the given image based off of the index passed to this function.
        """

        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # necessary since mask is RGB image

        if self.encode:
            mask = self.rgb_to_class(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask
