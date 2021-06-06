import torch
from torch import nn, optim
from datasets import FreiburgForestDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import jaccard_score as jsc
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image
from ignite.metrics.confusion_matrix import ConfusionMatrix


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def decode_segmentation_map(image, num_channels, dataset):
    """
    Convert a 'class' image (tensor where each pixel corresponds to a given class) to an image
    where each pixel has the appropriate color that corresponds to its correct class

    :param image:
    :param num_channels:
    :param dataset:
    :return: rgb_image
    """

    if dataset == 'freiburg':
        #  todo: fix label_map, since two classes map to same color it doesn't work
        label_map = np.array([(0, 0, 0),        # 0 = void        (black)
                              (170, 170, 170),  # 1 = road        (gray)
                              (0, 255, 0),      # 2 = grass       (light green)
                              (102, 102, 51),   # 3 = vegetation  (brownish)
                              #(0, 60, 0),       # 4 = tree        (dark green)
                              (0, 120, 255),    # 4 = sky         (light blue)
                              (0, 0, 0),        # 5 = obstacle    (black)
                              ])
    elif dataset == 'sun':
        label_map = np.array([(0, 0, 0),        # 0 = void                  (black)
                              (119, 119, 119),  # 1 = wall                  (gray)
                              (244, 243, 132),  # 2 = floor                 (light yellow)
                              (54, 114, 113),   # 3 = chair (originally 5)  (grey-blue)
                              (87, 112, 255)    # 4 = door                  (light blue)
                              ])

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


# inspiration: https://stackoverflow.com/questions/49338166/python-intersection-over-union
def intersection_over_union(prediction, target, num_classes):
    """
    Calculate the intersection over union.
    :param prediction:
    :param target:
    :param num_classes:
    :return:
    """

    prediction = torch.argmax(prediction.squeeze(), dim=0).detach().cpu().numpy()

    target = target.squeeze()
    target = np.array(target)

    prediction = np.array(prediction, dtype=np.float32)
    target = np.array(target, dtype=np.float32)

    ious = []
    for c in range(1, num_classes):  # exclude NULL
        pred_indx = prediction == c
        target_indx = target == c

        intersection = pred_indx * target_indx
        union = pred_indx + target_indx
        iou = (intersection.sum()) / (float(union.sum()))
        ious.append(iou)
    return ious


def calculate_accuracy_single_image(model, loader, num_classes, dataset='sun', visualize=False):
    """
    Calculate accuracy for just a single image.
    :param model:
    :param loader:
    :param num_classes:
    :param dataset:
    :param visualize:
    :return:
    """

    # get image and labels
    img, label = next(iter(loader))
    label = label * 255
    aux1, aux2, output = model(img)

    # calculate iou
    ious = intersection_over_union(output, label, num_classes)

    if visualize:
        # save labels (ground truth) as rgb
        label_rgb = label.squeeze()
        label_rgb = decode_segmentation_map(label_rgb, num_classes, dataset)
        label_rgb = Image.fromarray(label_rgb)
        label_rgb.save('ground_truth.png')

        # save image and prediction
        save_image(img, 'image.png')

        output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        rgb = decode_segmentation_map(output, 5, 'sun')
        img = Image.fromarray(rgb)
        img.save('segmentation.png')

    return ious


def calculate_accuracy(model, loader, num_classes, method='torch'):
    """
    Calculate accuracy for entire test dataset.
    :param model:
    :param loader:
    :param num_classes:
    :param dataset:
    :return:
    """

    print('calculating accuracy...')

    # inspiration: https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial
    if method == 'torch':
        confusion_matrix = torch.zeros(num_classes, num_classes)
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                aux1, aux2, output = model(images)
                _, preds = torch.max(output, 1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        print(confusion_matrix)
        return 0

    else:
        iou_each_image = []
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels * 255
            aux1, aux2, output = model(images)

            # calculate iou for single image
            ious = intersection_over_union(output, labels, num_classes)
            iou_each_image.append([ious])
        accuracy = None  # todo: finish this by using all ious for each image

    return accuracy


def main():

    image_path = '/mnt/d/data/freiburg_forest_annotated/test/rgb'
    mask_path = '/mnt/d/data/freiburg_forest_annotated/test/GT_color'
    dataset = FreiburgForestDataset(image_path, mask_path, transform_images=False, encode=True)

    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # load previously computed model
    #model = AdapNet(C=dataset.num_classes)
    model = model.to(device)
    #m = 'models/sun_best_model.pt'
    m = 'models/best_model_depth.pt'
    model.load_state_dict(torch.load(m, map_location=torch.device('cpu')))
    model.eval()

    #test_single = calculate_accuracy_single_image(model, test_loader, dataset.num_classes, 'freiburg', True)
    test_entire = calculate_accuracy(model, test_loader, dataset.num_classes)



if __name__ == '__main__':
    main()
