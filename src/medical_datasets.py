import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import cv2
import glob
import os
from albumentations.pytorch import ToTensorV2
import json


class KvasirSegDataset(Dataset):

    def __init__(self, image_path, gt_path, bboxes_path, transform=None):
        self.image_path = sorted(glob.glob(os.path.join(image_path, '*')))
        self.gt_path = sorted(glob.glob(os.path.join(gt_path, '*')))
        self.bbox_data = self.process_bboxes(bboxes_path)
        self.transform = transform

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
                current_img_bboxes.append([xmin, ymin, xmax, ymax, 'polyp'])
            bboxes.append(current_img_bboxes)
        return bboxes

    def rgb_to_class(self, mask):
        """
        Convert RGB mask image to 2D tensor containing class labels
        Map each pixel to its corresponding class label.
        :param mask: 3D tensor with shape (H, W, C)
        :return:
        """

        thresh, class_image = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        class_image = class_image / 255
        return class_image

    def class_to_rgb(self, prediction):
        """
        Convert a segmentation map to an rgb image for visualization purposes.
        :param prediction: torch.tensor of shape [1, h, w] or [h, w]
        :return: rgb, np.ndarray of shape [h, w, 3]
        """

        prediction = torch.squeeze(prediction).cpu().detach().numpy()
        label_colors = np.array([(0, 0, 0), (255, 255, 255)])
        r = np.zeros_like(prediction).astype(np.uint8)
        b = np.zeros_like(prediction).astype(np.uint8)
        g = np.zeros_like(prediction).astype(np.uint8)

        for i in range(0, 2):
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.gt_path[idx], cv2.IMREAD_GRAYSCALE)
        bboxes = self.bbox_data[idx]

        # encode mask: [H, W, C] -> [H, W] and each 'pixel' in mask is 0 or 1
        mask = self.rgb_to_class(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask, bboxes=bboxes)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask, bboxes


