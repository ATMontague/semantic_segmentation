import cv2
import glob
import os


def check_masks(masks):

    with open('mask_errors.txt', 'w') as f:
        for m in masks:
            img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            # really strange, there are small values in [1, 5] in the top left of some images
            top_left = img[0:8, 0:8]
            if img[0, 0] != 0:
                # print(top_left)
                # print()
                f.write(m)
                f.write('\n')


def identify_resolutions(masks):
    """
    Identify the different resolution images in dataset.
    """

    res = {}
    for m in masks:
        img = cv2.imread(m)
        as_str = str(img.shape)
        if as_str not in res:
            res[as_str] = 1
        else:
            res[as_str] += 1
    for key, val in res.items():
        print('{}: {}'.format(key, val))


def update_bas_masks(masks, bad_mask_list):
    """
    Replace the bad pixel and save all masks in a new folder.
    """

    new_loc = '../data/Kvasir-SEGv2/masks/'

    with open(bad_mask_list, 'r') as f:
        data = f.read().splitlines()
        mask_errors = data

    for m in masks:
        # not ideal but too lazy to move good images
        img = cv2.imread(m)
        if m in mask_errors:
            # img[0, 0] = (0, 0, 0)
            img[0:8, 0:8] = (0, 0, 0)
        # move to new location
        name = new_loc + m[25:]
        cv2.imwrite(name, img)


def main():
    path = '../data/Kvasir-SEG/masks'
    masks = sorted(glob.glob(os.path.join(path, '*')))

    check_masks(masks)
    # resolutions = identify_resolutions(masks)

    bad = 'mask_errors.txt'
    update_bas_masks(masks, bad)


if __name__ == '__main__':
    main()
