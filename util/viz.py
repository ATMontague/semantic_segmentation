import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.medical_datasets import KvasirSegDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# from albumentations example
# https://albumentations.ai/docs/examples/example_bboxes/
BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min,  x_max, y_max, c = bbox

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
    plt.savefig('bbox_viz.png')


def main():

    img_path = '../data/Kvasir-SEG/images'
    gt_path = '../data/Kvasir-SEG/masks'
    bbox_path = '../data/Kvasir-SEG/kavsir_bboxesv2.json'

    valid_transform = A.Compose(
            [
                A.Resize(height=528, width=617),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc')
        )

    dataset = KvasirSegDataset(img_path, gt_path, bbox_path, transform=None)

    category_ids = [0, 0]
    category_id_to_name = {0: 'polyp'}
    idx = 7
    img, mask, bbox = dataset[idx]

    visualize(img, bbox, category_ids, category_id_to_name)


if __name__ == '__main__':
    main()
