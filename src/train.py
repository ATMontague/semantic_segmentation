import torch
from torch.optim import Adam
import torch.nn as nn
from datasets import FreiburgForestDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
from models.unet import Unet
import matplotlib.pyplot as plt
from mlflow import log_metric, log_param

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 2
BATCH_SIZE = 5


def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=2):

    # move model to appropriate device
    model = model.to(device)

    # track change in validation loss
    valid_loss_min = np.Inf
    train_loss_over_time = []
    valid_loss_over_time = []

    for e in range(epochs):
        print('EPOCH: ', e)
        # track training & validation loss
        train_loss = 0.0
        validation_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for images, labels in train_loader:
            images = images.to(device)  # shape = (batch_size, num_channels, height, width) = (n, 3, H, W)
            labels = labels.to(device)  # shape = (batch_size, num_channels, height, width) = (n, 1, H, W)

            # reverse normalization to get class labels
            labels = labels*255  # todo: see if this is correct or see if there is a better way
            target = labels.squeeze(1)
            target = torch.tensor(target, dtype=torch.long)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward
            output = model(images)  # shape = (batch_size, num_classes, height, width) = (n, 6, H, W)

            # backward
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            # reverse normalization to get class labels
            labels = labels*255
            target = labels.squeeze(1)
            target = torch.tensor(target, dtype=torch.long)

            # forward
            output = model(images)

            # backward
            loss = criterion(output, target)
            validation_loss += loss.item() * images.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        validation_loss = validation_loss / len(valid_loader.sampler)
        train_loss_over_time.append(train_loss)
        valid_loss_over_time.append(validation_loss)

        # save model if validation loss has decreased
        if validation_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, validation_loss))
            torch.save(model.state_dict(), '../models/best_model.pt')
            valid_loss_min = validation_loss
            log_metric('validation loss', valid_loss_min)
    return train_loss_over_time, valid_loss_over_time


def main():

    # load training dataset
    image_path = '../data/freiburg_forest_annotated/train/rgb'
    mask_path = '../data/freiburg_forest_annotated/train/GT_color'
    dataset = FreiburgForestDataset(image_path, mask_path, transform_images=False, encode=True)

    # split train into train/validation
    train_count = int(np.ceil(0.8 * len(dataset)))
    valid_count = int(np.floor(0.2 * len(dataset)))
    train_data, validation_data = random_split(dataset, (train_count, valid_count))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=False)

    # load model
    model = Unet()
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=learning_rate)

    # track hyperparameters
    log_param('lr', learning_rate)
    log_param('batch size', BATCH_SIZE)
    log_param('loss', 'Cross Entropy')
    log_param('optim', 'Adam')
    log_param('epochs', EPOCHS)

    # train the first model

    print('----------------------training----------------------')
    train_loss, valid_loss = train_model(model=model, train_loader=train_loader, valid_loader=valid_loader,
                                         criterion=criterion, optimizer=optim, epochs=EPOCHS)

    # plot train loss
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(frameon=False)
    plt.savefig('model_loss.png')


if __name__ == '__main__':
    main()
