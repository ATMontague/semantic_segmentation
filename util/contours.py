import cv2
import imutils
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='../data/Kvasir-SEG/images/cju0roawvklrq0799vmjorwfv.jpg')
    parser.add_argument('--mask', default='../data/Kvasir-SEG/masks/cju0roawvklrq0799vmjorwfv.jpg')
    return parser.parse_args()


def main():
    # todo: determine how to handle white pixel in top-left of data/Kvasir-SEG/masks/cju0roawvklrq0799vmjorwfv.jpg'
    # todo: determine how many maks have this problem (exists in multiple masks)
    args = get_args()

    COLOR = (0, 0, 255)

    img = cv2.imread(args.image)
    mask_rgb = cv2.imread(args.mask)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    k = 11
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    # identify contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # draw bounding box
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(mask_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite('output.png', mask_rgb)


if __name__ == '__main__':
    main()
