import torch
import numpy as np
from torchvision import models
from PIL import Image
import torchvision.transforms as TF
from torchvision.utils import save_image

'''
tutorial: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
input:  [batch_size, num_channels, height, width]
output: [batch_size, num_classes, height, width]

this tutorial uses FCN with Resnet-101 backbone
'''


class SegmentationMap(object):

    def __init__(self):
        self.label_map_colors = np.array([(0, 0, 0),        # 0 = background
                                  (128, 0, 0),      # 1 = aeroplane
                                  (0, 128, 0),      # 2 = bicycle
                                  (128, 128, 0),    # 3 = bird
                                  (0, 0, 128),      # 4 = boat
                                  (128, 0, 128),    # 5 = bottle
                                  (0, 128, 128),    # 6 = bus
                                  (128, 128, 128),  # 7 = car
                                  (64, 0, 0),       # 8 = cat
                                  (192, 0, 0),      # 9 = chair
                                  (64, 128, 0),     # 10 = cow
                                  (192, 128, 0),    # 11 = dining table
                                  (64, 0, 128),     # 12 = dog
                                  (192, 0, 128),    # 13 = horse
                                  (64, 128, 128),   # 14 = motorbike
                                  (192, 128, 128),  # 15 = person
                                  (0, 64, 0),       # 16 = potted plant
                                  (128, 64, 0),     # 17 = sheep
                                  (0, 192, 0),      # 18 = sofa
                                  (128, 192, 0),    # 19 = train
                                  (0, 64, 128)])    # 20 = tv/monitor

        self.label_map_names = {0: 'background',
                                1: 'aeroplane',
                                2: 'bicycle',
                                3: 'bird',
                                4: 'boat',
                                5: 'bottle',
                                6: 'bus',
                                7: 'car',
                                8: 'cat',
                                9: 'chair',
                                10: 'cow',
                                11: 'dining table',
                                12: 'dog',
                                13: 'horse',
                                14: 'motorbike',
                                15: 'person',
                                16: 'potted plant',
                                17: 'sheep',
                                18: 'sofa',
                                19: 'train',
                                20: 'tv/monitor'}


def decode_segmentation_map(image, num_channels):
    """
    Convert a 'class' image (tensor where each pixel corresponds to a given class) to an image
    where each pixel has the appropriate color that corresponds to its correct class

    :param image:
    :param num_channels:
    :return: rgb_image
    """

    label_map = np.array([(0, 0, 0),        # 0 = background
                          (128, 0, 0),      # 1 = aeroplane
                          (0, 128, 0),      # 2 = bicycle
                          (128, 128, 0),    # 3 = bird
                          (0, 0, 128),      # 4 = boat
                          (128, 0, 128),    # 5 = bottle
                          (0, 128, 128),    # 6 = bus
                          (128, 128, 128),  # 7 = car
                          (64, 0, 0),       # 8 = cat
                          (192, 0, 0),      # 9 = chair
                          (64, 128, 0),     # 10 = cow
                          (192, 128, 0),    # 11 = dining table
                          (64, 0, 128),     # 12 = dog
                          (192, 0, 128),    # 13 = horse
                          (64, 128, 128),   # 14 = motorbike
                          (192, 128, 128),  # 15 = person
                          (0, 64, 0),       # 16 = potted plant
                          (128, 64, 0),     # 17 = sheep
                          (0, 192, 0),      # 18 = sofa
                          (128, 192, 0),    # 19 = train
                          (0, 64, 128)])    # 20 = tv/monitor

    # create zero array same size as image provided for each color channel
    r = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)

    # for each class update the color arrays where class matches
    # ex: let image[0, 0] = 4 -> replace that with (0, 0, 128)
    for l in range(0, num_channels):
        # obtain the corresponding indexes in the image for pixels that match current class label
        idx = image == l
        r[idx] = label_map[l, 0]
        g[idx] = label_map[l, 1]
        b[idx] = label_map[l, 2]

    rgb_image = np.stack([r, g, b], axis=2)
    return rgb_image


def objects_in_image(image):

    objects = []
    seg_map = SegmentationMap()
    unique_objects = np.unique(image)

    for obj in unique_objects:
        objects.append(seg_map.label_map_names[obj])
    return objects


def main():
    #########################################################
    # Load a pre-trained FCN model with Resnet-101 backbone #
    # Load image and pre-process it for loading into model  #
    #########################################################

    fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
    img = Image.open('../data/0000063.jpg')
    transforms = TF.Compose([TF.Resize((256, 256)),
                             TF.CenterCrop(224),
                             TF.ToTensor(),
                             TF.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])  # normalize w/ Imagenet specific values
                             ])
    # add extra dimension to image (batch_size) & apply transformations
    input = transforms(img).unsqueeze(0)
    print('new image shape: ', input.shape)
    # save_image(input, 'test.png')

    ####################################
    # forward pass through the network #
    ####################################

    output = fcn(input)['out']
    print('output shape: ', output.shape)
    #print('output looks like...\n', output)
    # output shape is [1, 21, 224, 224] or [batch_size, classes, height, width]
    # it has 21 'channels' where each individual channel pixel represents...
    # todo: understand what the pixel values represent
    #  it isn't probability
    #  it isn't the class label

    # 'condense' the 21 channels to 1 channel and give each pixel the appropriate class label
    out = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    print('squeezing...')
    print('output shape: ', out.shape)
    print('classes in the output image: ', np.unique(out))

    ##################################################
    # decode the output                              #
    # transform the 'class' image so that each pixel #
    # has the appropriate color                      #
    ##################################################

    rgb = decode_segmentation_map(out, 21)
    img = Image.fromarray(rgb)
    img.save('output.png')

    objects = objects_in_image(out)
    print(objects)


if __name__ == '__main__':
    main()
