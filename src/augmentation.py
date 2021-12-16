import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
import copy
import argparse
import os
from datasets import FreiburgForestDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Freiburg', help='Which dataset to use.')
    parser.add_argument('--height', default=384, help='Height the images')
    parser.add_argument('--width', default=768, help='Width of the image.')
    parser.add_argument('--path', default='../data/freiburg_forest_annotated')
    parser.add_argument('--num_aug', default=10, help='The number of augmented images to add.')

    return parser.parse_args()


# from albumentations tutorial
# https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
def visualize_augmentations(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(5, 10))
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.savefig('aug_results.png')


def main():

    args = get_args()

    transformations = A.Compose(
        [
            A.Resize(args.height, args.width),
            # A.HorizontalFlip(p=0.5),
            # A.IAAAffine(translate_px=(0, 100), mode='constant', cval=0),
            # A.RandomSunFlare(),
            A.RandomShadow(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    )

    if args.data == 'Freiburg' or args.data == 'freiburg':
        img_path = args.path + '/train/rgb'
        mask_path = args.path + '/train/GT_color'
        dataset = FreiburgForestDataset(img_path, mask_path, transform=transformations)

    idx = 2
    visualize_augmentations(dataset, idx, 5)
    actual = cv2.imread(dataset.image_paths[idx])
    # actual = cv2.cvtColor(actual, cv2.COLOR_GRAY2RGB)
    cv2.imwrite('actual.png', actual)


if __name__ == '__main__':
    main()
