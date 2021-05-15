import tensorflow as tf

def read_image(filename):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    # if image_type == 'tiff':
    #     image = tfio.experimental.image.decode_tiff(string)
    #     image = image[:,:,:3]
    # else:
    image = tf.image.decode_image(string, channels=3)
    return image


def check_image_size(image, patchsize):
    shape = tf.shape(image)
    return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
    image = tf.image.random_crop(image, (patchsize, patchsize, 3))
    image = tf.cast(image, tf.float32)
    return image/255.


def create_dataset_iter(files, batchsize, patchsize, train=True):
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if train:
        dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x: read_image(x))
    dataset = dataset.filter(
        lambda x: check_image_size(x, patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x, patchsize))
    dataset = dataset.batch(batchsize, drop_remainder=train)
    dataset_iter = iter(dataset)
    return dataset_iter
