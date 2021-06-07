import torch
from torch.optim import Adam
import torch.nn as nn
from datasets import FreiburgForestDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
from models.simple import Net


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=2):

    # move model to appropriate device
    model = model.to(device)

    # track change in validation loss
    valid_loss_min = np.Inf
    train_loss_over_time = []

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
            images = images.to(device)  # shape = (batch_size, num_channels, height, width) = (n, 3, 250, 250)
            labels = labels.to(device)  # shape = (batch_size, num_channels, height, width) = (n, 1, 250, 250)

            # reverse normalization to get class labels
            labels = labels*255  # todo: see if this is correct or see if there is a better way
            target = labels.squeeze(1)
            target = torch.tensor(target, dtype=torch.long)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward: gives 3 tensors
            output = model(images)  # shape = (batch_size, num_classes, height, width) = (n, 6, 250, 250)

            # backward
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_loss_over_time.append(train_loss)

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
        temp = 0.000001
        train_loss = (train_loss + temp) / (len(train_loader.sampler) + temp)
        validation_loss = (validation_loss + temp) / (len(valid_loader.sampler) + temp)

        # save model if validation loss has decreased
        if validation_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                validation_loss))
            torch.save(model.state_dict(), 'best_model.pt')
            valid_loss_min = validation_loss


def main():

    # load training dataset
    image_path = '../data/freiburg_forest_annotated/train/rgb'
    mask_path = '../data/freiburg_forest_annotated/train/GT_color'
    dataset = FreiburgForestDataset(image_path, mask_path, transform_images=False, encode=True)

    # split train into train/validation
    train_count = int(np.ceil(0.8 * len(dataset)))
    valid_count = int(np.floor(0.2 * len(dataset)))
    train_data, validation_data = random_split(dataset, (train_count, valid_count))
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
    valid_loader = DataLoader(dataset=validation_data, batch_size=1, shuffle=False)

    # load model
    model = Net()
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=learning_rate)

    # train the first model
    print('----------------------training----------------------')
    train_model(model=model, train_loader=train_loader, valid_loader=valid_loader,
                criterion=criterion, optimizer=optim, epochs=5)


if __name__ == '__main__':
    main()
