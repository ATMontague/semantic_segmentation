import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.medical_datasets import KvasirSegDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy


BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


# from albumentations example
# https://albumentations.ai/docs/examples/example_bboxes/
def visualize_bbox(img, bboxes, class_name='polyp', color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""

    for bbox in bboxes:
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


def visualize(dataset, idx, samples):

    fig, ax = plt.subplots(nrows=samples, ncols=2, figsize=(5, 10))
    for i in range(0, samples):
        img, mask, bboxes = dataset[idx]
        mask = mask * 255
        img = img.cpu().detach().numpy()
        img = np.moveaxis(img, 0, -1)
        img = visualize_bbox(img, bboxes)
        ax[i, 0].imshow(img)
        ax[i, 1].imshow(mask)
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.savefig('bbox_viz.png')


def main():

    img_path = '../data/Kvasir-SEG/images'
    gt_path = '../data/Kvasir-SEG/masks'
    bbox_path = '../data/Kvasir-SEG/kavsir_bboxes.json'

    trans = A.Compose(
            [
                A.Resize(height=512, width=512),
                A.HorizontalFlip(),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc'))

    idx = 645
    samps = 4
    dataset = KvasirSegDataset(img_path, gt_path, bbox_path, transform=trans)

    visualize(dataset, idx=idx, samples=samps)


if __name__ == '__main__':
    main()
