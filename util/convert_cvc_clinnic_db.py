import os
from libtiff import TIFF    # pip install libtiff
import imageio
import glob


def convert(src, dst):
    """
    Usage:
        formatting `tif/tiff` files to `jpg/png` files
    :param _src_path:
    :param _dst_path:
    :return:
    """
    tif = TIFF.open(src, mode='r')
    image = tif.read_image()
    imageio.imwrite(dst, image)


if __name__ == '__main__':
    src = '../data/CVC-ClinicDB/'
    src_images = sorted(glob.glob(os.path.join(src + 'Ground Truth', '*')))
    src_masks = sorted(glob.glob(os.path.join(src + 'Original', '*')))

    dst = '../data/CVC-ClinicDB_png/'
    dst_images = dst + 'Ground Truth/'
    dst_masks = dst + 'Original/'

    count = 0
    for img_name in src_images:
        tif = TIFF.open(img_name, mode='r')
        image = tif.read_image()
        save_loc = '{}{}.png'.format(dst_images, count)
        count += 1
        imageio.imwrite(save_loc, image)

    count = 0
    for img_name in src_masks:
        tif = TIFF.open(img_name, mode='r')
        image = tif.read_image()
        save_loc = '{}{}.png'.format(dst_masks, count)
        count += 1
        imageio.imwrite(save_loc, image)

