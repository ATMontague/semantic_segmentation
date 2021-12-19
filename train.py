import argparse
import yaml
import torch
from torch.optim import Adam
import torch.nn as nn
from src.datasets import FreiburgForestDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
from src.models.unet import Unet
from mlflow import log_metric, log_param
from tqdm import tqdm
from torchmetrics.classification import IoU, Precision, Accuracy
import albumentations as A
from albumentations.pytorch import ToTensorV2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='../config/freiburg_config.yaml',
                        help='The config file to be used to determine hyperparameters.')
    return parser.parse_args()


def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=2):

    # move model to appropriate device
    model = model.to(device)

    # track change in validation loss
    valid_loss_min = np.Inf

    # track additional metrics
    valid_precision = Precision(compute_on_step=False, num_classes=model.num_classes,
                                average='none', mdmc_average='global')
    valid_accuracy = Accuracy(compute_on_step=False, num_classes=model.num_classes,
                              average='none', mdmc_average='global')
    valid_iou = IoU(compute_on_step=False, num_classes=model.num_classes, reduction='none')

    for e in tqdm(range(0, epochs)):

        # track training & validation loss
        train_loss = 0.0
        validation_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for images, masks in train_loader:
            images = images.to(device)  # shape = (batch_size, 3, height, width)
            masks = masks.to(device)  # shape = (batch_size, 1, height, width)

            target = masks.squeeze(dim=1)
            target = target.type(torch.LongTensor)
            target = target.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward
            output = model(images)  # shape = (batch_size, num_classes, height, width)

            # backward
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(device)
                masks = masks.to(device)

                target = masks.squeeze(dim=1)
                target = target.type(torch.LongTensor)
                target = target.to(device)

                # forward
                output = model(images)

                # metric calculation
                prediction = torch.argmax(output, dim=1)
                valid_precision(prediction, target)
                valid_accuracy(prediction, target)
                valid_iou(prediction, target)

                # backward
                loss = criterion(output, target)
                validation_loss += loss.item() * images.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        validation_loss = validation_loss / len(valid_loader.sampler)
        log_metric('training loss', train_loss)
        log_metric('validation loss', validation_loss)

        # calculate and log metrics
        log_metric('IoU', torch.mean(valid_iou.compute()).item())
        log_metric('Precision', torch.mean(valid_precision.compute()).item())
        log_metric('Accuracy', torch.mean(valid_accuracy.compute()).item())

        # save model if validation loss has decreased
        if validation_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, validation_loss))
            save_name = 'checkpoints/{}.pt'.format(model.name)
            torch.save(model.state_dict(), save_name)
            valid_loss_min = validation_loss
            log_metric('validation loss', valid_loss_min)


def main():

    args = get_args()
    try:
        with open(args.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError('No config file found.')

    # todo: simplify this
    # todo: add cut/paste type transform to add additional obstacles to handle class imbalance
    if params['transform'] == 'horizontal flip':
        trans = A.Compose(
            [
                A.Resize(params['height'], params['width']),
                A.HorizontalFlip(),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )
    elif params['transform'] == 'vertical flip':
        trans = A.Compose(
            [
                A.Resize(params['height'], params['width']),
                A.VerticalFlip(),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )
    elif params['transform'] == 'flip':
        trans = A.Compose(
            [
                A.Resize(params['height'], params['width']),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )
    elif params['transform'] == 'rotate':
        trans = A.Compose(
            [
                A.Resize(params['height'], params['width']),
                A.IAAAffine(rotate=(-360, 360)),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )
    elif params['transform'] == 'translate':
        trans = A.Compose(
            [
                A.Resize(params['height'], params['width']),
                A.IAAAffine(translate_px=(0, 100), mode='constant', cval=0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )
        # todo: make sure null class separate from obstacle class
    elif params['transform'] == 'drop':
        trans = A.Compose(
            [
                A.Resize(params['height'], params['width']),
                A.CoarseDropout(max_holes=25, max_height=15, max_width=15, min_holes=1, min_height=5, min_width=5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )
    elif params['transform'] == 'scale':
        trans = A.Compose(
            [
                A.Resize(params['height'], params['width']),
                A.IAAAffine(scale=(0.8, 1.2), mode='constant', cval=0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )
    elif params['transform'] == 'crop':
        trans = A.Compose(
            [
                A.Resize(params['height'], params['width']),
                A.RandomCropNearBBox()
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )
    elif params['transform'] == 'solar flare':
        trans = A.Compose(
            [
                A.Resize(params['height'], params['width']),
                A.RandomSunFlare(),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )
    else:
        trans = A.Compose(
            [
                A.Resize(params['height'], params['width']),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )

    # load training dataset
    image_path = 'data/freiburg_forest_annotated/train/rgb'
    mask_path = 'data/freiburg_forest_annotated/train/GT_color'
    dataset = FreiburgForestDataset(image_path, mask_path, transform=trans, encode=True)

    # todo: correct this
    # split train into train/validation
    train_count = int(np.ceil(0.8 * len(dataset)))
    valid_count = int(np.floor(0.2 * len(dataset)))
    train_data, validation_data = random_split(dataset, (train_count, valid_count))
    train_loader = DataLoader(dataset=train_data, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(dataset=validation_data, batch_size=params['batch_size'], shuffle=True)

    # load model
    model = Unet(6)
    learning_rate = params['lr']
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=learning_rate)

    # track hyperparameters
    log_param('model', params['model'])
    log_param('dataset', params['dataset'])
    log_param('train set size', train_count)
    log_param('validation set size', valid_count)
    log_param('lr', params['lr'])
    log_param('batch size', params['batch_size'])
    log_param('loss', 'Cross Entropy')
    log_param('optim', 'Adam')
    log_param('epochs', params['epochs'])
    log_param('height', params['height'])
    log_param('width', params['width'])

    # train the model
    print('----------------------training----------------------')
    train_model(model=model, train_loader=train_loader, valid_loader=valid_loader,
                criterion=criterion, optimizer=optim, epochs=params['epochs'])


if __name__ == '__main__':
    main()
