from PIL import Image as im
import matplotlib.pyplot as plt
from skimage.transform import resize
import imageio
import imageio.v3 as iio
import torch
import matplotlib
from pathlib import Path
import numpy as np
imageio.plugins.freeimage.download()

def write_obj(vertices, indices, filename):
    with open(filename, 'w') as (f):
        for v in vertices:
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for i in indices:
            f.write('f %d %d %d\n' % (i[0] + 1, i[1] + 1, i[2] + 1))


def read_2d_obj(filename, flip_orientation=False):
    with open(filename, 'r') as (f):
        lines = f.readlines()
    vertices = []
    indices = []
    values = []
    for line in lines:
        if line.startswith('v '):
            v = [float(x) for x in line.split()[1:]]
            vertices.append(v[:2])
        if line.startswith('l '):
            l = [int(x) - 1 for x in line.split()[1:]]
            if flip_orientation:
                l = l[::-1]
            indices.append(l[:2])
        if line.startswith('c '):
            c = [float(x) for x in line.split()[1:]]
            values.append(c[0])

    return np.array(vertices), np.array(indices), np.array(values)


def write_2d_obj(filename, vertices, indices, values):
    with open(filename, 'w') as (f):
        for v in vertices:
            f.write('v %f %f\n' % (v[0], v[1]))

        for i in indices:
            f.write('l %d %d\n' % (i[0] + 1, i[1] + 1))

        for v in values:
            f.write('c %f\n' % v)


def read_3d_obj(filename):
    with open(filename, 'r') as (f):
        lines = f.readlines()
    vertices = []
    indices = []
    values = []
    for line in lines:
        if line.startswith('v '):
            v = [float(x) for x in line.split()[1:]]
            vertices.append(v)
        if line.startswith('f '):
            l = [int(x) - 1 for x in line.split()[1:]]
            indices.append(l)
        if line.startswith('c '):
            c = [float(x) for x in line.split()[1:]]
            values.append(c[0])

    return np.array(vertices), np.array(indices), np.array(values)


def write_3d_obj(filename, vertices, indices, values):
    with open(filename, 'w') as (f):
        for v in vertices:
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for i in indices:
            f.write('f %d %d %d\n' % (i[0] + 1, i[1] + 1, i[2] + 1))

        for v in values:
            f.write('c %f\n' % v)


def color_map(data, vmin=-1., vmax=1.):
    my_cm = matplotlib.cm.get_cmap('viridis')
    normed_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    mapped_data = my_cm(normed_data)
    return mapped_data


def linear_to_srgb(l):
    s = np.zeros_like(l)
    m = l <= 0.00313066844250063
    s[m] = l[m] * 12.92
    s[~m] = 1.055*(l[~m]**(1.0/2.4))-0.055
    return s


def srgb_to_linear(s):
    l = np.zeros_like(s)
    m = s <= 0.0404482362771082
    l[m] = s[m] / 12.92
    l[~m] = ((s[~m]+0.055)/1.055) ** 2.4
    return l


def to_srgb(image):
    return np.clip(linear_to_srgb(to_numpy(image)), 0, 1)


def to_linear(image):
    return srgb_to_linear(to_numpy(image))


def to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    else:
        return np.array(data)


def read_image(image_path, is_srgb=None):
    image_path = Path(image_path)
    image = iio.imread(image_path)
    image = np.atleast_3d(image)
    if image.dtype == np.uint8 or image.dtype == np.int16:
        image = image.astype("float32") / 255.0
    elif image.dtype == np.uint16 or image.dtype == np.int32:
        image = image.astype("float32") / 65535.0

    if is_srgb is None:
        if image_path.suffix in ['.exr', '.hdr', '.rgbe']:
            is_srgb = False
        else:
            is_srgb = True

    if is_srgb:
        image = to_linear(image)

    return image


def write_image(image_path, image, is_srgb=None):
    image_path = Path(image_path)
    image = to_numpy(image)
    image = np.atleast_3d(image)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if is_srgb is None:
        if image_path.suffix in ['.exr', '.hdr', '.rgbe']:
            is_srgb = False
        else:
            is_srgb = True

    if is_srgb:
        image = to_srgb(image)

    if image_path.suffix == '.exr':
        image = image.astype(np.float32)
    else:
        image = (image * 255).astype(np.uint8)

    iio.imwrite(image_path, image)


def read_png(png_path, is_srgb=True):
    image = iio.imread(png_path, extension='.png')
    if image.dtype == np.uint8 or image.dtype == np.int16:
        image = image.astype("float32") / 255.0
    elif image.dtype == np.uint16 or image.dtype == np.int32:
        image = image.astype("float32") / 65535.0

    if len(image.shape) == 4:
        image = image[0]

    # Only read the RGB channels
    if len(image.shape) == 3:
        image = image[:, :, :3]

    if is_srgb:
        return to_linear(image)
    else:
        return image


def write_png(png_path, image):
    image = to_srgb(to_numpy(image))
    image = (image * 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    iio.imwrite(png_path, image, extension='.png')


def write_jpg(jpg_path, image):
    image = to_srgb(to_numpy(image))
    image = (image * 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    rgb_im = im.fromarray(image).convert('RGB')
    rgb_im.save(jpg_path, format='JPEG', quality=95)


def read_exr(exr_path):
    image = iio.imread(exr_path, extension='.exr')
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    return image


def write_image(image_path, image, is_srgb=None):
    image_path = Path(image_path)

    image_ext = image_path.suffix
    iio_plugins = {
        '.exr': 'EXR-FI',
        '.hdr': 'HDR-FI',
        '.png': 'PNG-FI',
    }
    iio_flags = {
        '.exr': imageio.plugins.freeimage.IO_FLAGS.EXR_NONE,
    }
    hdr_formats = ['.exr', '.hdr', '.rgbe']

    image = to_numpy(image)
    image = np.atleast_3d(image)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if image_ext in hdr_formats:
        is_srgb = False if is_srgb is None else is_srgb
    else:
        is_srgb = True if is_srgb is None else is_srgb
    if is_srgb:
        image = to_srgb(image)

    if image_ext in hdr_formats:
        image = image.astype(np.float32)
    else:
        image = (image * 255).astype(np.uint8)

    flags = iio_flags.get(image_ext)
    if flags is None:
        flags = 0

    iio.imwrite(image_path, image,
                flags=flags,
                plugin=iio_plugins.get(image_ext))


def write_exr(exr_path, image):
    exr_path = Path(exr_path)
    assert exr_path.suffix == '.exr'
    write_image(exr_path, image, is_srgb=False)
    # image = to_numpy(image).astype(np.float32)
    # if len(image.shape) == 3:
    #     if image.shape[2] < 3:
    #         padding = np.zeros((image.shape[0], image.shape[1], 3 - image.shape[2]), dtype=np.float32)
    #         image = np.concatenate((image, padding), axis=2)
    #     image = np.expand_dims(image, axis=0)
    # try:
    #     iio.imwrite(exr_path, image)
    # except OSError:
    #     imageio.plugins.freeimage.download()
    #     iio.imwrite(exr_path, image, extension='.exr', flags=imageio.plugins.freeimage.IO_FLAGS.EXR_NONE)


def resize_image(image, height, width):
    return resize(image, (height, width))


def print_quartiles(image):
    percentile = [0, 25, 50, 75, 100]
    percentile = [np.percentile(image, p) for p in percentile]
    print(percentile)


def subplot(images, vmin=0.0, vmax=1.0):
    n = len(images)
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i], vmin=vmin, vmax=vmax, cmap="viridis")
        plt.axis("off")


class FileStream:
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'rb')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def read(self, count: int, dtype=np.byte):
        data = self.file.read(count * np.dtype(dtype).itemsize)
        return np.frombuffer(data, dtype=dtype)
