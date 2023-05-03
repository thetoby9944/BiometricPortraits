import numpy as np


def enhance_contrast(image_matrix, bins=256):
    """
    Contreast based on https://msameeruddin.hashnode.dev/image-equalization-contrast-enhancing-in-python

    :param image_matrix:
    :param bins:
    :return:
    """
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq


def equalize_this(img, gray_scale=False, bins=256):
    """
    Equalizing based on https://msameeruddin.hashnode.dev/image-equalization-contrast-enhancing-in-python

    :param img: np.array
    :param gray_scale: Is Grayscale?
    :param bins:
    :return:
    """
    image_src = img
    if not gray_scale:
        r_image = image_src[:, :, 0]
        g_image = image_src[:, :, 1]
        b_image = image_src[:, :, 2]

        r_image_eq = enhance_contrast(image_matrix=r_image)
        g_image_eq = enhance_contrast(image_matrix=g_image)
        b_image_eq = enhance_contrast(image_matrix=b_image)

        image_eq = np.dstack(tup=(r_image_eq, g_image_eq, b_image_eq))
    else:
        image_eq = enhance_contrast(image_matrix=image_src)

    return image_eq.astype(np.uint8)
