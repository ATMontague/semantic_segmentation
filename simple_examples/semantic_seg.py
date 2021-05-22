import os
import torch
import torchvision
from d2l import torch as d2l


#################################################################################
# from tutorial
# https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html #
#################################################################################

def read_voc_images(voc_dir, amount=10, is_train=True):
    """
    Read all VOC feature and label images
        This reads every image into memory....why would we want to do that?
    :param voc_dir: path to VOC2012
    :param amount: the number of images to read (since this is tutorial we don't need to load everything)
    :param is_train: loading train images or test images
    :return:
    """

    if is_train:
        txt = 'train.txt'
    else:
        txt = 'val.txt'

    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', txt)
    mode = torchvision.io.image.ImageReadMode.RGB

    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features = []
    labels = []
    counter = 0
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png')))
        counter += 1
        if counter == amount:
            break
    return features, labels


def main():
    # download the data
    # d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
    #                            '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')
    #
    # voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

    voc_dir = '../data/VOCdevkit/VOC2012'
    train_features, train_labels = read_voc_images(voc_dir)

    # viz images
    n = 5
    imgs = train_features[0: n] + train_labels[0: n]
    imgs = [img.permute(1, 2, 0) for img in imgs]
    d2l.show_images(imgs, 2, n)


if __name__ == '__main__':
    main()
