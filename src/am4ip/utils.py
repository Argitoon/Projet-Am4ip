
import os
import numpy as np


def expanded_join(path: str, *paths: str) -> str:
    """Path concatenation utility function.
    Automatically handle slash and backslash depending on the OS but also relative path user.

    :param path: Most left path.
    :param paths: Most right parts of path to concatenate.
    :return: A string which contains the absolute path.
    """
    return os.path.expanduser(os.path.join(path, *paths))


def random_crop(im_ref, im, crop_size):
    """Given a pair of images, perform a random crop (no padding) at the same location.

    :param im_ref: A PIL.Image variable.
    :param im: A PIL.Image variable. Should be of same size as im_ref.
    :param crop_size: Size of the crop. Must be a tuple of (h, w).
    :return:
    """
    assert im.size == im_ref.size, "Size between reference image and distorted image mismatched."
    w, h = im.size
    if w == crop_size[1] and h == crop_size[0]:
        return im_ref, im

    h_off_range = h - crop_size[0]
    w_off_range = w - crop_size[1]

    if h_off_range > 0:
        h_off = np.random.randint(0, h_off_range, 1, dtype=np.int32)[0]
        h_m = crop_size[0]
    else:
        h_off = 0
        h_m = h

    if w_off_range > 0:
        w_off = np.random.randint(0, w_off_range, 1, dtype=np.int32)[0]
        w_m = crop_size[1]
    else:
        w_off = 0
        w_m = w

    im = im.crop((w_off, h_off, w_off + w_m, h_off + h_m))
    im_ref = im_ref.crop((w_off, h_off, w_off + w_m, h_off + h_m))

    return im_ref, im