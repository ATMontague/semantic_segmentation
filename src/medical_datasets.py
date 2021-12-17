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


# from albumentations example
# https://albumentations.ai/docs/examples/example_bboxes/
BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


class KvasirSegDataset(Dataset):

    def __init__(self, image_path, gt_path, bboxes_path, height=256, width=256, transform=None):
        self.image_path = sorted(glob.glob(os.path.join(image_path, '*')))
        self.gt_path = sorted(glob.glob(os.path.join(gt_path, '*')))
        self.bbox_data = self.process_bboxes(bboxes_path)
        self.height = height
        self.width = width
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

    def rgb_to_class(self, image, mask):
        """
        Convert RGB mask image to 2D tensor containing class labels
        Map each pixel to its corresponding class label.
        :param mask: 3D tensor with shape (H, W, C)
        :return:
        """

        grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        grayscale[grayscale > 0] = 1
        return grayscale

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):

        image = cv2.imread(self.image_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.gt_path[idx])
        bboxes = self.bbox_data[idx]

        # encode mask: [H, W, C] -> [H, W] and each 'pixel' in mask is 0 or 1
        mask = self.rgb_to_class(image, mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask, bboxes=bboxes)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask, bboxes


def main():

    img_path = '../data/Kvasir-SEG/images'
    gt_path = '../data/Kvasir-SEG/masks'
    bbox_path = '../data/Kvasir-SEG/kavsir_bboxes.json'

    valid_transform = A.Compose(
            [
                A.Resize(512, 512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc')
        )

    dataset = KvasirSegDataset(img_path, gt_path, bbox_path, transform=valid_transform, height=512, width=512)


if __name__ == '__main__':
    main()
