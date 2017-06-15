import numpy as np
import os
import sys
from PIL import Image, ImageOps


def merge(images, size):
    """
    Generates a merged image from all the input images.
    :param images: Images to be merged.
    :param size: [number of rows, number of columns]
    :return: Merged image.
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[int(j) * h:int(j) * h + h, int(i) * w:int(i) * w + w, :] = image

    return img


def load_dataset(path, data_set='birds', image_size=64):
    # TODO: Remove redundant code from this function.
    """
    Loads the images from the specified path
    :param path: string indicating the dataset path.
    :param data_set: 'birds' -> loads data from birds directory, 'flowers' -> loads data from the flowers directory.
    :param image_size: size of images in the returned array
    :return: numpy array, shape : [number of images, image_size, image_size, 3]
    """
    if data_set == 'birds':
        image_dirs = os.listdir(path)[0]
        number_of_images = len(image_dirs)
        images = []
        print("{} images are being loaded...".format(data_set[:-1]))
        for i in image_dirs:
            for c, j in enumerate(os.listdir(path + i)):
                images.append(np.array(ImageOps.fit(Image.open(path + i + '/' + j),
                                                    (image_size, image_size), Image.ANTIALIAS))/127.5 - 1.)
                sys.stdout.write("\r Loading : {}/{}"
                                 .format(c + 1, number_of_images))
                print("\n")
        images = np.reshape(images, [-1, image_size, image_size, 3])
        return images.astype(np.float32)

    elif data_set == 'roses':
        image_dirs = os.listdir(path)
        number_of_images = len(image_dirs)
        images = []
        print("{} images are being loaded...".format(data_set[:-1]))
        for c, i in enumerate(image_dirs):
            images.append(np.array(ImageOps.fit(Image.open(path + '/' + i),
                                                (image_size, image_size), Image.ANTIALIAS))/127.5 - 1.)
            sys.stdout.write("\r Loading : {}/{}"
                             .format(c + 1, number_of_images))
        print("\n")
        images = np.reshape(images, [-1, image_size, image_size, 3])
        return images.astype(np.float32)

    elif data_set == 'black_birds':
        image_dirs = os.listdir(path)
        number_of_images = len(image_dirs)
        images = []
        print("{} images are being loaded...".format(data_set[:-1]))
        for c, i in enumerate(image_dirs):
            images.append(np.array(ImageOps.fit(Image.open(path + i),
                                                (image_size, image_size), Image.ANTIALIAS))/127.5 - 1.)
            sys.stdout.write("\r Loading : {}/{}".format(c + 1, number_of_images))
            print("\n")
        images = np.reshape(images, [-1, image_size, image_size, 3])
        return images.astype(np.float32)


def next_batch(data, batch_size):
    """
    Returns a random chosen batch from the data array.
    :param data: numpy array consisting the entire dataset
    :param batch_size: should I even explain.
    :return: [batch_size, default image size, default image size, 3]
    """
    return np.random.permutation(data)[:batch_size]
