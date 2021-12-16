import torch
import numpy as np


# todo: add options for different datasets, implement mask color option
def class_to_rgb(self, prediction, dataset, mask_color):
    """
    Convert a segmentation map to an rgb image for visualization purposes.
    :param prediction: torch.tensor of shape [1, h, w] or [h, w]
    :param dataset:
    :param mask_color:
    :return: rgb, np.ndarray of shape [h, w, 3]
    """

    prediction = torch.squeeze(prediction).cpu().detach().numpy()

    if dataset == 'VOC':
        label_map = np.array([(0, 0, 0),  # 0 = background
                              (128, 0, 0),  # 1 = aeroplane
                              (0, 128, 0),  # 2 = bicycle
                              (128, 128, 0),  # 3 = bird
                              (0, 0, 128),  # 4 = boat
                              (128, 0, 128),  # 5 = bottle
                              (0, 128, 128),  # 6 = bus
                              (128, 128, 128),  # 7 = car
                              (64, 0, 0),  # 8 = cat
                              (192, 0, 0),  # 9 = chair
                              (64, 128, 0),  # 10 = cow
                              (192, 128, 0),  # 11 = dining table
                              (64, 0, 128),  # 12 = dog
                              (192, 0, 128),  # 13 = horse
                              (64, 128, 128),  # 14 = motorbike
                              (192, 128, 128),  # 15 = person
                              (0, 64, 0),  # 16 = potted plant
                              (128, 64, 0),  # 17 = sheep
                              (0, 192, 0),  # 18 = sofa
                              (128, 192, 0),  # 19 = train
                              (0, 64, 128)])  # 20 = tv/monitor

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

