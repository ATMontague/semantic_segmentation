import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import cv2
import glob
import os
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import copy
import matplotlib.pyplot as plt
import json


class KvasirSegDataset(Dataset):

    def __init__(self, image_path, gt_path, height=256, width=256, num_classes=2, transform=None,
                 multiclass=False):
        self.image_path = sorted(glob.glob(os.path.join(image_path, '*')))
        self.gt_path = sorted(glob.glob(os.path.join(gt_path, '*')))
        self.height = height
        self.width = width
        self.transform = transform
        self.multiclass = multiclass
        if self.multiclass:
            self.num_classes = 3
        else:
            self.num_classes = num_classes

    def rgb_to_class(self, image, mask):
        """
        Convert RGB mask image to 2D tensor containing class labels
        Map each pixel to its corresponding class label.
        :param mask: 3D tensor with shape (H, W, C)
        :return:
        """

        # adding an additional class
        # we now have [0, 1, 2] -> [skin, polyp, null]
        if self.multiclass:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # get location of every black pixel in image
            thresh = 5  # hyperparameter

            # coords = np.column_stack(np.where(gray < thresh))
            # make mask for these pixels
            m = gray < thresh

            # [0, 0, 0], [255, 255, 255], and [0, 0, 255]
            grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # convert the mask to class labels
            grayscale[grayscale > 0] = 1
            grayscale[m] = 2
        else:
            thresh, class_image = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            # image now contains pixel value of either 0 or 255
            # divide by 255 to convert to class labels (255 -> 1)
            class_image = class_image / 255
        return class_image

    def class_to_rgb(self, prediction):
        """
        Convert a segmentation map to an rgb image for visualization purposes.
        :param prediction: torch.tensor of shape [1, h, w] or [h, w]
        :return: rgb, np.ndarray of shape [h, w, 3]
        """

        prediction = torch.squeeze(prediction).cpu().detach().numpy()
        if self.multiclass:
            label_colors = np.array([(0, 0, 0), (255, 0, 0), (0, 0, 255)])
        else:
            label_colors = np.array([(0, 0, 0), (255, 0, 0)])
        r = np.zeros_like(prediction).astype(np.uint8)
        b = np.zeros_like(prediction).astype(np.uint8)
        g = np.zeros_like(prediction).astype(np.uint8)

        for i in range(0, self.num_classes):
            idx = prediction == i
            r[idx] = label_colors[i, 0]
            g[idx] = label_colors[i, 1]
            b[idx] = label_colors[i, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):

        image = cv2.imread(self.image_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # [h, w, 3] uint-8
        mask = cv2.imread(self.gt_path[idx], cv2.IMREAD_GRAYSCALE)  # [h, w] uint-8

        # encode mask: convert grayscale image to binary. 1 is polyp, 0 is ~polyp
        mask = self.rgb_to_class(image, mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask


class KvasirSegDataset2(Dataset):

    def __init__(self, image_path, gt_path, bboxes_path, height=256, width=256, num_classes=2,
                 transform_images=False,
                 multiclass=False):
        self.image_path = sorted(glob.glob(os.path.join(image_path, '*')))
        self.gt_path = sorted(glob.glob(os.path.join(gt_path, '*')))
        self.bbox_data = self.process_bboxes(bboxes_path)
        self.height = height
        self.width = width
        self.transform_images = transform_images
        self.multiclass = multiclass
        if self.multiclass:
            self.num_classes = 3
        else:
            self.num_classes = num_classes

    def process_bboxes(self, path):

        # process json and load bbox data for each image in a list
        f = open(path)
        data = json.load(f)
        bboxes = []
        for name in self.image_path:
            n = name[26: -4]  # stripping image name from path
            info = data[n]['bbox']  # get data for specific image bboxes
            current_img_bboxes = []  # there can be more than one polyp per image
            for i in range(0, len(info)):
                xmin = info[i]['xmin']
                ymin = info[i]['ymin']
                xmax = info[i]['xmax']
                ymax = info[i]['ymax']
                current_img_bboxes.append([xmin, ymin, xmax, ymax])
            bboxes.append(current_img_bboxes)
        return bboxes

    def transform(self, image, mask):

        prob = 0.5

        transformations = A.Compose(
            [
                A.Resize(self.height, self.width),
                A.HorizontalFlip(p=prob),
                A.RandomRotate90(p=prob),
                A.Blur(blur_limit=3, p=prob)
            ], bbox_params=A.BboxParams(format='pascal_voc')
        )

        return transformations(image=image, mask=mask)

    def rgb_to_class(self, image, mask):
        """
        Convert RGB mask image to 2D tensor containing class labels
        Map each pixel to its corresponding class label.
        :param mask: 3D tensor with shape (H, W, C)
        :return:
        """

        # adding an additional class
        # we now have [0, 1, 2] -> [skin, polyp, null]
        if self.multiclass:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # get location of every black pixel in image
            thresh = 5  # hyperparameter

            # coords = np.column_stack(np.where(gray < thresh))
            # make mask for these pixels
            m = gray < thresh

            # [0, 0, 0], [255, 255, 255], and [0, 0, 255]
            grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # convert the mask to class labels
            grayscale[grayscale > 0] = 1
            grayscale[m] = 2
        else:
            grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            grayscale[grayscale > 0] = 1
        return grayscale

    def class_to_rgb(self, mask):
        """
        Convert a segmentation map to an rgb image for visualization purposes.
        :param mask: torch.tensor of shape [1, h, w] or [h, w]
        :return: mask_img, np.ndarray of shape [h, w, 3]
        """

        mask = torch.squeeze(mask).cpu().detach().numpy()
        if self.multiclass:
            label_colors = np.array([(0, 0, 0), (255, 255, 255), (0, 0, 255)])
        else:
            label_colors = np.array([(0, 0, 0), (255, 0, 0)])
        r = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)

        for i in range(0, self.num_classes):
            idx = mask == i
            r[idx] = label_colors[i, 0]
            g[idx] = label_colors[i, 1]
            b[idx] = label_colors[i, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):

        image = cv2.imread(self.image_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.gt_path[idx])
        bbox = self.bbox_data[idx]

        # encode mask: [H, W, C] -> [H, W] and each 'pixel' in mask is 0 or 1
        mask = self.rgb_to_class(image, mask)

        if self.transform_images:
            transformed = self.transform(image, mask)
            image = transformed['image']
            mask = transformed['mask']

        else:
            # probably a simpler way...
            transformations = A.Compose(
                [
                    A.Resize(self.height, self.width, interpolation=cv2.INTER_AREA),
                ]
            )
            transformed = transformations(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)*255
        return image, mask, bbox
