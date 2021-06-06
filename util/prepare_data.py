import argparse
import os
from datasets import FreiburgForestDataset
from torchvision.utils import save_image


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default=None, help='The path to the training data folder.')
    parser.add_argument('--masks', default=None, help='The path to the test data folder.')
    parser.add_argument('--train', default=True, help='Train or test.')
    parser.add_argument('--path', default=None, help='Path where new data will be saved.')
    parser.add_argument('--dataset', default='Freiburg Forest', help='Dataset to be used.')
    return parser.parse_args()


def create_file(image_path, mask_path, new_path, train, images_path, masks_path):

    images = []
    masks = []
    counter = 0
    for filename in os.listdir(image_path):
        #filename = '{}.{}'.format(counter, filename[-3:])
        filename = '{}.png'.format(counter)
        images.append(images_path+'/'+filename)
        counter += 1
    counter = 0
    for filename in os.listdir(mask_path):
        #filename = '{}.{}'.format(counter, filename[-3:])
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


def transform_images(image_path, mask_path, output_path, train, images_path, masks_path):

    data = FreiburgForestDataset(image_path, mask_path, transform_images=False, encode=True)

    # todo, loop 2*len(image_path) and do original images at same time as transformed images

    for i in range(0, len(data)):
        image, mask = data[i]
        image_name = '{}/{}.png'.format(images_path, i)
        save_image(image, image_name)
        mask_name = '{}/{}.png'.format(masks_path, i)
        save_image(mask, mask_name)


def main():
    args = get_args()

    if args.images is None or not os.path.isdir(args.images):
        print('Train directory missing or not valid')
        return
    if args.masks is None or not os.path.isdir(args.masks):
        print('Test directory missing or not valid')
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

    # transform and save images
    transform_images(args.images, args.masks, args.path, args.train, images_path, masks_path)

    # create text file for image locations
    create_file(args.images, args.masks, args.path, args.train, images_path, masks_path)


if __name__ == '__main__':
    main()
