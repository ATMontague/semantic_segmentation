import argparse
import yaml
import torch
from torch.optim import Adam
import torch.nn as nn
from src.datasets import FreiburgForestDataset
from src import KvasirSegDataset, CVCClinicDB
from torch.utils.data import DataLoader, random_split
import numpy as np
from src.models.unet import Unet
from mlflow import log_metric, log_param
from tqdm import tqdm
from torchmetrics.classification import IoU, Precision, Accuracy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset


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
    # todo: move outside of function and pass as arg
    THRESH = 0.5
    if model.num_classes == 1:
        precision = Precision(compute_on_step=False, threshold=THRESH, multiclass=False,
                              num_classes=1, average='none')
        accuracy = Accuracy(compute_on_step=False, threshold=THRESH, multiclass=False,
                            num_classes=1, average='none')
        # according to docs we need num_classes=2
        iou = IoU(compute_on_step=False, threshold=THRESH, num_classes=2)
    else:
        precision = Precision(compute_on_step=False, num_classes=model.num_classes,
                                    average='none', mdmc_average='global')
        accuracy = Accuracy(compute_on_step=False, num_classes=model.num_classes,
                                  average='none', mdmc_average='global')
        iou = IoU(compute_on_step=False, num_classes=model.num_classes, reduction='none')

    precision = precision.to(device)
    accuracy = accuracy.to(device)
    iou = iou.to(device)

    for e in tqdm(range(0, epochs)):

        # track training & validation loss
        train_loss = 0.0
        validation_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for images, masks in train_loader:
            images = images.to(device)  # shape = (b, 3, h, w)
            masks = masks.to(device)  # shape = (b, 1, h, w)

            target = masks.squeeze(dim=1)
            # target = target.type(torch.LongTensor)
            # target = target.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward
            output = model(images)  # shape = (b, num_classes, h, w)

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
                target = target.type(torch.IntTensor)
                target = target.to(device)

                # forward. softmax to convert to probs, then convert to classes
                output = model(images)
                prediction = torch.sigmoid(output)
                prediction = torch.where(prediction > THRESH, 1, 0)

                # metric calculation
                precision(prediction.view(-1), target.view(-1))
                accuracy(prediction.view(-1), target.view(-1))
                iou(prediction, target)

                # backward
                target = target.type(torch.FloatTensor)
                target = target.to(device)
                loss = criterion(output, target)
                validation_loss += loss.item() * images.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        validation_loss = validation_loss / len(valid_loader.sampler)
        log_metric('training loss', train_loss)
        log_metric('validation loss', validation_loss)

        # calculate and log metrics
        log_metric('IoU', torch.mean(iou.compute()).item())
        log_metric('Precision', torch.mean(precision.compute()).item())
        log_metric('Accuracy', torch.mean(accuracy.compute()).item())

        # save model if validation loss has decreased
        if validation_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, validation_loss))
            save_name = 'checkpoints/{}.pt'.format(model.name)
            torch.save(model.state_dict(), save_name)
            valid_loss_min = validation_loss


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
                A.RandomCropNearBBox(),
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
    if params['dataset'] == 'Polyp':
        kvasir = KvasirSegDataset(params['img_loc'][0], params['mask_loc'][0], None, transform=trans)
        cvc = CVCClinicDB(params['img_loc'][1], params['mask_loc'][1], transform=trans)
        dataset = ConcatDataset([kvasir, cvc])
    else:
        raise NotImplementedError
        image_path = params['img_loc']
        mask_path = params['mask_loc']
        if params['bbox_loc'] is not None:
            bbox_loc = params['bbox_loc']

        if params['dataset'] == 'Freiburg Forest':
            dataset = FreiburgForestDataset(image_path, mask_path, transform=trans, encode=True)
        elif params['dataset'] == 'Kvasir':
            dataset = KvasirSegDataset(image_path, mask_path, bbox_loc, transform=trans)

    # todo: save test_loader to disc
    assert((params['train_ratio'] + params['validation_ratio'] + params['test_ratio']) == 1)
    train_count = int(np.ceil(params['train_ratio'] * len(dataset)))
    valid_count = int(np.floor(params['validation_ratio'] * len(dataset)))
    test_count = len(dataset) - train_count - valid_count
    train_data, validation_data, test_data = random_split(dataset, (train_count, valid_count, test_count))
    train_loader = DataLoader(dataset=train_data, batch_size=params['batch_size'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=validation_data, batch_size=params['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=params['batch_size'], shuffle=True)
    assert(train_count + valid_count + test_count == len(dataset))

    # print(len(dataset))
    # print('train: ', train_count)
    # print('valid: ', valid_count)
    # print('test: ', test_count)
    # print('batches')
    # print('train: ', len(train_loader))
    # print('val: ', len(valid_loader))
    # raise NotImplementedError

    # load model
    if params['model'] == 'Unet':
        model = Unet(num_classes=params['classes'])
    else:
        raise NotImplementedError

    if params['loss'] == 'Cross Entropy':
        if model.num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    if params['optimizer'] == 'Adam':
        optim = Adam(model.parameters(), lr=params['lr'])
    elif params['optimizer']  == 'SGD':
        optim = SGD(model.parameters(), lr=params['lr'])

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
