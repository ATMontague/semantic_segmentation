import numpy as np


def zero_pad(arr, pad):
    """
    Take an array and pad the border with zeros.
    :param arr: python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    :param pad: integer, amount of padding around each image on vertical and horizontal dimensions
    :return: arr_pad: padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """

    arr_pad = np.pad(arr, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))
    return arr_pad

def convolution():
    pass


def main():

    arr = np.ones(shape=(2, 3, 3, 3))
    print('original')
    print(arr.shape)
    new_arr = zero_pad(arr=arr,pad=2)
    print('after padding')
    print(new_arr.shape)
    print(new_arr[0][:, :, 0]) # first image, all x and y coordinates for the 1st channel (channel 0)


if __name__ == '__main__':
    main()