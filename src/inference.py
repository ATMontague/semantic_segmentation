import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from models.simple import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def decode_segmentation_map(image, num_channels, dataset):
    """
    Convert a 'class' image (tensor where each pixel corresponds to a given class) to an image
    where each pixel has the appropriate color that corresponds to its correct class

    :param image:
    :param num_channels:
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


def main():
    # load previously computed model
    model = Net()
    model = model.to(device)
    m = 'best_model.pt'
    model.load_state_dict(torch.load(m, map_location=torch.device('cpu')))
    model.eval()

    # get image to test
    img = '../data/freiburg_forest_annotated/test/rgb/b1-09517_Clipped.jpg'
    img = Image.open(img)
    transforms = T.Compose([T.Resize(size=(250, 250)), T.ToTensor()])
    input = transforms(img).unsqueeze(0)
    print('input shape: ', input.shape)

    ####################################
    # forward pass through the network #
    ####################################
    output = model(input)
    print('output shape: ', output.shape)
    output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    ######################################
    print('squeezing...')
    print('output shape: ', output.shape)
    print('output classes: ', np.unique(output))
    test = output.flatten()
    unique, count = np.unique(test, return_counts=True)
    d = dict(zip(unique, count))
    print('count of each class: ', d)
    ############################################

    rgb = decode_segmentation_map(output, 5, 'freiburg')
    img = Image.fromarray(rgb)
    img.save('test.png')


if __name__ == '__main__':
    main()
