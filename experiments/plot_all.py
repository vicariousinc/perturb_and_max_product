import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage


def semilogx_std(x, y, std, color="blue"):
    plt.semilogx(x, y, color=color)
    lower_bound = y - std
    upper_bound = y + std
    ax = plt.gca()
    ax.fill_between(x, lower_bound, upper_bound, facecolor=color, alpha=0.5)


def plot_std(x, y, std, color="blue"):
    plt.plot(x, y, color=color)
    lower_bound = y - std
    upper_bound = y + std
    ax = plt.gca()
    ax.fill_between(x, lower_bound, upper_bound, facecolor=color, alpha=0.5)


def plot_images(images, zoom_times=0, filename=None, display=True, nr=None):
    n_images, H, W = images.shape
    images = images - images.min()
    images /= images.max() + 1e-10

    if nr is None:
        nr = nc = np.ceil(np.sqrt(n_images)).astype(int)
    else:
        nc = n_images // nr
        assert n_images == nr * nc
    big_image = np.ones(((H + 1) * nr + 1, (W + 1) * nc + 1, 3))
    big_image[..., :3] = 0
    big_image[:: H + 1] = [0.502, 0, 0.502]
    im = 0
    for r in range(nr):
        for c in range(nc):
            if im < n_images:
                big_image[
                    (H + 1) * r + 1 : (H + 1) * r + 1 + H,
                    (W + 1) * c + 1 : (W + 1) * c + 1 + W,
                    :,
                ] = images[im, :, :, None]
            im += 1

    if display:
        plt.figure(figsize=(10, 10))
        plt.imshow(big_image, interpolation="none")
    for _ in range(zoom_times):
        big_image = ndimage.zoom(big_image, [2, 2, 1], order=0)
    if filename:
        pass
        # imwrite(filename, img_as_ubyte(big_image))
    return big_image
