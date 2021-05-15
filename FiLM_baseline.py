import tensorflow as tf
# import tensorflow_io as tfio
import tensorflow_compression as tfc
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import glob
import os
import argparse
import random
import csv
import warnings
import datetime
from os import path
from string import ascii_uppercase

from data_helpers import *

np.random.seed(0)

SCALE_MIN = 0.11
SCALE_MAX = 255.
NUM_SCALES = 64


class LinearTransform(tf.keras.layers.Layer):
    """
    Layer that implements y=m*x+b, where m and b are learnable parameters.
    """
    def __init__(self, gamma_initializer="ones", beta_initializer="zeros"):
        # super().__init__(dtype=dtype, **kwargs)
        super(LinearTransform, self).__init__()
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer

    def build(self, input_shape):
        num_channels = int(input_shape[-1])
        self.gamma = self.add_weight(
            "gamma",
            shape=[num_channels],
            initializer=self.gamma_initializer,
            trainable=True
        )
        self.beta = self.add_weight(
            "beta",
            shape=[num_channels],
            initializer=self.beta_initializer,
            dtype=self.dtype,
            trainable=True
        )

    def call(self, inputs):
        return self.gamma * inputs + self.beta


def AnalysisTransform(num_filters=128, kernel_size=[5,5,5,5]):
    input_image = tf.keras.Input((None, None, 3), name='encoder_input')

    encoder_0 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        name='encoder_0')(input_image)
    encoder_0 = LinearTransform()(encoder_0)
    encoder_0 = tf.keras.layers.ReLU()(encoder_0)

    encoder_1 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='encoder_1')(encoder_0)
    encoder_1 = LinearTransform()(encoder_1)
    encoder_1 = tf.keras.layers.ReLU()(encoder_1)

    encoder_2 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        name='encoder_2')(encoder_1)
    encoder_2 = LinearTransform()(encoder_2)
    encoder_2 = tf.keras.layers.ReLU()(encoder_2)

    encoder_output = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[3],
        strides=2,
        padding='same',
        name='encoder_output')(encoder_2)

    return tf.keras.Model(input_image, encoder_output, name='encoder_network')


def SynthesisTransform(num_filters=128, kernel_size=[5,5,5,5]):
    decoder_input = tf.keras.Input((None, None, num_filters), name='decoder_input')

    decoder_0 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        name='decoder_0')(decoder_input)
    decoder_0 = LinearTransform()(decoder_0)
    decoder_0 = tf.keras.layers.ReLU()(decoder_0)

    decoder_1 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='decoder_1')(decoder_0)
    decoder_1 = LinearTransform()(decoder_1)
    decoder_1 = tf.keras.layers.ReLU()(decoder_1)

    decoder_2 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        name='decoder_2')(decoder_1)
    decoder_2 = LinearTransform()(decoder_2)
    decoder_2 = tf.keras.layers.ReLU()(decoder_2)

    decoder_output = tf.keras.layers.Conv2DTranspose(filters=3,
        kernel_size=kernel_size[3],
        strides=2,
        padding='same',
        name='decoder_output')(decoder_2)

    return tf.keras.Model(decoder_input, decoder_output, name='decoder_network')


def HyperAnalysisTransform(num_filters=128, kernel_size=[3,5,5]):
    hyper_encoder_input = tf.keras.Input((None, None, num_filters), name='hyper_encoder_input')

    hyper_encoder_0 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=1,
        padding='same',
        name='hyper_encoder_0')(hyper_encoder_input)
    hyper_encoder_0 = LinearTransform()(hyper_encoder_0)
    hyper_encoder_0 = tf.keras.layers.ReLU()(hyper_encoder_0)

    hyper_encoder_1 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='hyper_encoder_1')(hyper_encoder_0)
    hyper_encoder_1 = LinearTransform()(hyper_encoder_1)
    hyper_encoder_1 = tf.keras.layers.ReLU()(hyper_encoder_1)

    hyper_encoder_output = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        name='hyper_encoder_output')(hyper_encoder_1)

    return tf.keras.Model(hyper_encoder_input, hyper_encoder_output, name='hyper_encoder_network')


def HyperSynthesisTransform(num_filters=128, kernel_size=[5,5,3]):
    hyper_decoder_input = tf.keras.Input((None, None, num_filters), name='hyper_decoder_input')

    hyper_decoder_0 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        name='hyper_decoder_0')(hyper_decoder_input)
    hyper_decoder_0 = LinearTransform()(hyper_decoder_0)
    hyper_decoder_0 = tf.keras.layers.ReLU()(hyper_decoder_0)

    hyper_decoder_1 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='hyper_decoder_1')(hyper_decoder_0)
    hyper_decoder_1 = LinearTransform()(hyper_decoder_1)
    hyper_decoder_1 = tf.keras.layers.ReLU()(hyper_decoder_1)

    hyper_decoder_output = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=1,
        padding='same',
        name='hyper_decoder_output')(hyper_decoder_1)

    return tf.keras.Model(hyper_decoder_input, hyper_decoder_output, name='hyper_decoder_network')


class BMSHJ2018Model(tf.keras.Model):
    """Main model class."""
    def __init__(self, lmbda, num_filters, num_scales, scale_min, scale_max):
        super().__init__()
        self.lmbda = lmbda
        self.num_scales = num_scales
        offset = tf.math.log(scale_min)
        factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
            num_scales - 1.)
        self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
        self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
        self.analysis_transform = AnalysisTransform(num_filters=num_filters) # input_shape
        self.hyper_analysis_transform = HyperAnalysisTransform(num_filters=num_filters) # self.analysis_transform.output.shape[1:]
        self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters=num_filters) # self.hyper_analysis_transform.output.shape[1:]
        self.synthesis_transform = SynthesisTransform(num_filters=num_filters) # self.analysis_transform.output.shape[1:]
        # self.build((None,input_shape[0],input_shape[1],3))

    def call(self, x, training):
        """Computes rate and distortion losses."""
        entropy_model = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
            compression=False)
        side_entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=False)

        y = self.analysis_transform(x)
        z = self.hyper_analysis_transform(abs(y))
        z_hat, side_bits = side_entropy_model(z, training=training)
        indexes = self.hyper_synthesis_transform(z_hat)
        y_hat, bits = entropy_model(y, indexes, training=training)
        x_hat = self.synthesis_transform(y_hat)

        # Total number of bits divided by total number of pixels.
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
        bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
        # Mean squared error across pixels.
        mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        mse *= 255 ** 2
        # The rate-distortion Lagrangian.
        loss = bpp + self.lmbda * mse
        return loss, bpp, mse

    def compile(self, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.bpp = tf.keras.metrics.Mean(name="bpp")
        self.mse = tf.keras.metrics.Mean(name="mse")

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss, bpp, mse = self(inputs, training=True)
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

    def test_step(self, x):
        loss, bpp, mse = self(x, training=False)
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}


def train(args):
    print(args)
    model_name = f'lmbda={args.lmbda}_steps={args.total_steps}_batchsize={args.batchsize}_patchsize={args.patchsize}_filters={args.num_filters}'

    # get training data files for each dataset
    clicm_files = glob.glob('/extra/ucibdl0/preethis/datasets/clic/train_m/*/*.png') # ~1000 images
    clicp_files = glob.glob('/extra/ucibdl0/preethis/datasets/clic/train_p/*/*.png') # ~600 images
    div2k_files =  glob.glob('/extra/ucibdl0/preethis/datasets/div2k/train/*/*.png') # ~800 images

    # coco has a lot more data ~40000 images, so sampling a smaller subset
    coco_files = glob.glob('/extra/ucibdl0/preethis/datasets/coco/train/*.jpg') # 3000 images (unless others specified)
    coco_files = np.random.choice(coco_files, args.max_images, replace=False)
    city_files = glob.glob('/extra/ucibdl0/preethis/datasets/cityscapes/train/*/*.png') # ~3000 images

    # not using because of issues with decoding TIFF image format (TODO: come back to this)
    # raise_files = glob.glob('/extra/ucibdl0/preethis/datasets/RAISE/train/*.TIFF')

    # create separate iters for each dataset
    # doing this instead of one single iter to account for variable dataset sizes
    clicm_iter = create_dataset_iter(clicm_files, args.batchsize, args.patchsize, train=True)
    clicp_iter = create_dataset_iter(clicp_files, args.batchsize, args.patchsize, train=True)
    div2k_iter = create_dataset_iter(div2k_files, args.batchsize, args.patchsize, train=True)
    coco_iter = create_dataset_iter(coco_files, args.batchsize, args.patchsize, train=True)
    city_iter = create_dataset_iter(city_files, args.batchsize, args.patchsize, train=True)

    train_iter_list = [clicm_iter, clicp_iter, coco_iter, div2k_iter, city_iter]

    # construct model
    model = BMSHJ2018Model(args.lmbda, args.num_filters, NUM_SCALES, SCALE_MIN, SCALE_MAX)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    if args.ckpt:
        model.load_weights(ckpt)

    # training
    print('begin training')
    train_log_dir = f'FiLM/logs/baseline/{model_name}'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    for step in range(args.start_step, args.total_steps):
        # sample a dataset
        index = random.randint(0, len(train_iter_list)-1) # endpoints are inclusive
        dataset = train_iter_list[index]
        data = next(dataset)
        # compute loss
        loss_dict  = model.train_step(data)

        # log training progress
        if step % 500 == 0 and step > 0:
            print(step, loss_dict)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_dict['loss'].numpy(), step=step)
                tf.summary.scalar('bpp', loss_dict['bpp'].numpy(), step=step)
                tf.summary.scalar('mse', loss_dict['mse'].numpy(), step=step)
            if step % 5000 == 0: # checkpoint
                model.save_weights(f'FiLM/checkpoints/baseline/{model_name}')

    # save model
    model.save(f'FiLM/models/baseline/{model_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--total_steps", type=int, default=1000000,
                        help="Total number of training steps/iterations")
    parser.add_argument("--start_step", type=int, default=0,
                        help="Start step (if loading checkpoint)")
    parser.add_argument("--patchsize", type=int, default=256,
                        help="Size of image patches for training and validation")
    parser.add_argument("--num_filters", type=int, default=128,
                        help="Number of filters per layer")
    parser.add_argument("--batchsize", type=int, default=16,
                        help="Batch size used to train compression model")
    parser.add_argument("--lmbda", type=float, default=0.01,
                        help="Lambda used for rate-distortion tradeoff")
    parser.add_argument("--max_images", type=int, default=3000,
                        help="Maximum number of images a dataset can have")
    parser.add_argument("--ckpt", default='',
                        help="Load checkpoint if using")

    args = parser.parse_args()

    train(args)
