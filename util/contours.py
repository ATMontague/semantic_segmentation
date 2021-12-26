import cv2
import imutils
import argparse

'''
using contour detection we can find the center of specified regions

goal: find center of each polyp prediction map
2ndary goal: find the number of polyps present in image
'''


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

    # img = '../data/Kvasir-SEG/images/cju0qx73cjw570799j4n5cjze.jpg'
    # img = cv2.imread(img)
    # mask = '../data/Kvasir-SEG/masks/cju0qx73cjw570799j4n5cjze.jpg'
    # mask_rgb = cv2.imread(mask)
    # mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(args.image)
    mask_rgb = cv2.imread(args.mask)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    k = 11
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    # identify contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # draw contour lines and centroid for each polyp detected
    for c in contours:
        cv2.drawContours(mask_rgb, c, -1, COLOR, 3)

        M = cv2.moments(c)
        c_x = int(M['m10'] / M['m00'])
        c_y = int(M['m01'] / M['m00'])
        cv2.circle(img=mask_rgb, center=(c_x, c_y), radius=10, color=(0, 255, 0), thickness=-1)


    cv2.imwrite('output.png', mask_rgb)


if __name__ == '__main__':
    main()