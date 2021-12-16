import torch
from datasets import FreiburgForestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import jaccard_score as jsc
from PIL import Image
from torchvision.utils import save_image
from src.models.simple import Net
from inference import decode_segmentation_map
from ignite.metrics import ConfusionMatrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def calculate_accuracy_single_image(model, loader, num_classes, dataset='freiburg', visualize=False):
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
    output = model(img)

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
        rgb = decode_segmentation_map(output, num_classes, dataset)
        img = Image.fromarray(rgb)
        img.save('segmentation.png')

    return ious


# inspiration: https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial
def calculate_accuracy(model, loader, num_classes):
    """
    Calculate accuracy for entire test dataset.
    :param model:
    :param loader:
    :param num_classes:
    :param dataset:
    :return:
    """

    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for images, labels in loader:
            labels = labels*255
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, preds = torch.max(output, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix, confusion_matrix.diag()/confusion_matrix.sum(1)


def main():

    image_path = '../data/freiburg_forest_annotated/test/rgb/'
    mask_path = '../data/freiburg_forest_annotated/test/GT_color/'
    dataset = FreiburgForestDataset(image_path, mask_path, transform_images=False, encode=True)

    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    # load previously computed model
    model = Net()
    model = model.to(device)
    m = '../models/best_model.pt'
    model.load_state_dict(torch.load(m, map_location=torch.device('cpu')))
    model.eval()

    #acc_single_img = calculate_accuracy_single_image(model, test_loader, dataset.num_classes, dataset='freiburg',visualize=True)
    confusion_mat, scores = calculate_accuracy(model, test_loader, dataset.num_classes)
    print(confusion_mat)
    print(scores)


if __name__ == '__main__':
    main()
