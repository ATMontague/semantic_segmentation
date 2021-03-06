import argparse
import os
from src.datasets import FreiburgForestDataset
from torchvision.utils import save_image
from distutils.dir_util import copy_tree
import shutil
import random

# todo: add function to append depth data to rgb and save each image as numpy array


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='../data/freiburg_forest_annotated/train/rgb',
                        help='The path to the training data folder.')
    parser.add_argument('--masks', default='../data/freiburg_forest_annotated/train/GT_color',
                        help='The path to the test data folder.')
    parser.add_argument('--train', default='True', help='Train or test.')
    parser.add_argument('--path', default='..n/data/freiburg_augmented', help='Path where new data will be saved.')
    parser.add_argument('--dataset', default='Freiburg Forest', help='Dataset to be used.')
    parser.add_argument('--transform', default=True, help='Performing data augmentation.')
    parser.add_argument('--amount', default=1000, help='How many new images to create.')
    return parser.parse_args()


def create_file(image_path, mask_path, new_path, train, images_path, masks_path):

    images = []
    masks = []
    counter = 0
    for filename in os.listdir(image_path):
        filename = '{}.png'.format(counter)
        images.append(images_path+'/'+filename)
        counter += 1
    counter = 0
    for filename in os.listdir(mask_path):
        filename = '{}.png'.format(counter)
        masks.append(masks_path+'/'+filename)
        counter += 1

    # combine the two
    if len(images) != len(masks):
        print('Each image must have corresponding mask')
        print('Images: {} != Masks: {}'.format(len(images), len(masks)))
        return
    else:
        if train:
            name = os.path.join(new_path, 'train/data.txt')
        else:
            name = os.path.join(new_path, 'test/data.txt')
        f = open(name, 'w')
        for i in range(0, len(images)):
            f.write('{} {}\n'.format(images[i], masks[i]))
        f.close()
        print('file generated')


def transform_images(image_path, mask_path, output_path, train, images_path, masks_path, transform, amount):

    count = 0
    # first get original data, then do augmented data. not the best way but it works
    data = FreiburgForestDataset(image_path, mask_path, transform_images=False, encode=True)
    for i in range(0, len(data)):
        image, mask = data[i]
        image_name = '{}/{}.png'.format(images_path, count)
        save_image(image, image_name)
        mask_name = '{}/{}.png'.format(masks_path, count)
        save_image(mask, mask_name)
        count += 1

    data = FreiburgForestDataset(image_path, mask_path, transform_images=transform, encode=True)

    # need to handle when we're using some number of augmented images larger than the amount of actual data we have
    # create random int between 0 and len(data) and access that image

    for i in range(0, amount):
        image, mask = data[random.randint(0, len(data) - 1)]
        image_name = '{}/{}.png'.format(images_path, count)
        save_image(image, image_name)
        mask_name = '{}/{}.png'.format(masks_path, count)
        save_image(mask, mask_name)
        count += 1


def main():
    args = get_args()

    if args.images is None or not os.path.isdir(args.images):
        print('Train directory missing or not valid')
        return
    if args.masks is None or not os.path.isdir(args.masks):
        print('Mask directory missing or not valid')
        return
    if not os.path.isdir(args.path):
        try:
            os.mkdir(args.path)
        except OSError:
            print('Error creating directory: {}'.format(args.path))
            return

    # building image/mask sub-folders for train/test
    args.train = eval(args.train)
    if args.train:
        path = os.path.join(args.path, 'train')
    else:
        path = os.path.join(args.path, 'test')
    images_path = os.path.join(path, 'images')
    masks_path = os.path.join(path, 'masks')
    if not os.path.isdir(path):
        try:
            os.makedirs(images_path)
            os.makedirs(masks_path)
        except OSError:
            print('Error creating directory')
            return

    # move initial data to new path and then add the augmented images
    # images_path = args.path+'/train/rgb'
    # masks_path = args.path+'/train/masks'
    # shutil.copytree(args.images, images_path)
    # shutil.copytree(args.masks, masks_path)

    # transform and save images
    transform_images(args.images, args.masks, args.path, args.train, images_path, masks_path, args.transform, args.amount)

    # create text file for image locations
    create_file(args.images, args.masks, args.path, args.train, images_path, masks_path)


if __name__ == '__main__':
    main()
