import torch
import numpy as np


def main():
    input_torch = torch.randn(1, 3, 2, 5, requires_grad=True)
    print('input torch:', input_torch.shape)

    one_hot = np.array([[[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0], [1, 1, 1, 0, 0]],
                        [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]])

    print('one hot encoded matrix:', one_hot.shape)
    print(one_hot)
    target = np.array([np.argmax(a, axis=0) for a in one_hot])
    print('target: ', target.shape)
    print(target)

    target_torch = torch.tensor(target)

    loss = torch.nn.CrossEntropyLoss()
    output = loss(input_torch, target_torch)
    output.backward()


if __name__ == '__main__':
    main()
