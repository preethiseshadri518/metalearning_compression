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


def crop_image(image, patchsize, celeb_a=False):
    image = tf.image.random_crop(image, (patchsize, patchsize, 3))
    image = tf.cast(image, tf.float32)
    return image/255.


def parse_tfrecord_tf(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([1], tf.int64)})
    data, label, shape = features['data'], features['label'], features['shape']
    img = tf.io.decode_raw(data, tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    return img


def create_dataset_iter(files, batchsize, patchsize, name, train=True):
    filenames = tf.data.Dataset.from_tensor_slices(files)
    if name == 'celeb_a':
        dataset = filenames.flat_map(lambda x: tf.data.TFRecordDataset(x))
        dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    else:
        dataset = filenames.shuffle(len(files), reshuffle_each_iteration=True)

    if train:
        dataset = dataset.repeat()

    if name == 'celeb_a':
        dataset = dataset.map(lambda x: parse_tfrecord_tf(x))
    else:
        dataset = dataset.map(lambda x: read_image(x))

    dataset = dataset.filter(
        lambda x: check_image_size(x, patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x, patchsize)) # if train
    dataset = dataset.batch(batchsize, drop_remainder=train)
    dataset_iter = iter(dataset)
    return dataset_iter
