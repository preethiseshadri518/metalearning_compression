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
import warnings
import csv
import datetime
from os import path
from string import ascii_uppercase

from data_helpers import *

np.random.seed(0)

SCALE_MIN = 0.11
SCALE_MAX = 255.
NUM_SCALES = 64


def mean(layer):
    return tf.keras.backend.mean(layer, axis=0)


def expand_dims(layer):
    return tf.expand_dims(layer, axis=0)


def ContextModel(context_length=64, num_filters=128, kernel_size=[5,5,5,5]):
    # Using sequential model here because functional model doesn't allow for
    # input shape to contain None if there are dense layers
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        activation='relu',
        name='context_0'))

    model.add(tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        activation='relu',
        name='context_1'))

    model.add(tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        activation='relu',
        name='context_2'))

    model.add(tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[3],
        strides=2,
        padding='same',
        activation='relu',
        name='context_3'))

    # output of context network is c. Pool c_1,...,c_m into a single context embedding
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Lambda(mean))
    model.add(tf.keras.layers.Lambda(expand_dims))
    model.add(tf.keras.layers.Dense(context_length, name='context_embedding'))

    return model


def AnalysisTransform(num_filters=128, context_length=64, kernel_size=[5,5,5,5]):
    input_image = tf.keras.Input((None,None,3), name='encoder_input')
    context_embedding = tf.keras.Input(shape=(context_length))

    gamma_e0 = tf.keras.layers.Dense(num_filters, name='gamma_e0')(context_embedding)
    beta_e0 = tf.keras.layers.Dense(num_filters, name='beta_e0')(context_embedding)
    gamma_e1 = tf.keras.layers.Dense(num_filters, name='gamma_e1')(context_embedding)
    beta_e1 = tf.keras.layers.Dense(num_filters, name='beta_e1')(context_embedding)
    gamma_e2 = tf.keras.layers.Dense(num_filters, name='gamma_e2')(context_embedding)
    beta_e2 = tf.keras.layers.Dense(num_filters, name='beta_e2')(context_embedding)

    encoder_0 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        name='encoder_0')(input_image)
    encoder_0 = tf.keras.layers.Multiply()([encoder_0, gamma_e0]) # scale by gamma along channel dim
    encoder_0 = tf.keras.layers.Add()([encoder_0, beta_e0]) # shift with beta along channel dim
    encoder_0 = tf.keras.layers.ReLU()(encoder_0)

    encoder_1 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='encoder_1')(encoder_0)
    encoder_1 = tf.keras.layers.Multiply()([encoder_1, gamma_e1])
    encoder_1 = tf.keras.layers.Add()([encoder_1, beta_e1])
    encoder_1 = tf.keras.layers.ReLU()(encoder_1)

    encoder_2 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        name='encoder_2')(encoder_1)
    encoder_2 = tf.keras.layers.Multiply()([encoder_2, gamma_e2])
    encoder_2 = tf.keras.layers.Add()([encoder_2, beta_e2])
    encoder_2 = tf.keras.layers.ReLU()(encoder_2)

    encoder_output = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[3],
        strides=2,
        padding='same',
        name='encoder_output')(encoder_2)

    return tf.keras.Model([input_image, context_embedding], encoder_output, name='encoder_network')

def SynthesisTransform(num_filters=128, context_length=64, kernel_size=[5,5,5,5]):
    decoder_input = tf.keras.Input((None,None,num_filters), name='decoder_input')
    context_embedding = tf.keras.Input(shape=(context_length))

    gamma_d0 = tf.keras.layers.Dense(num_filters, name='gamma_d0')(context_embedding)
    beta_d0 = tf.keras.layers.Dense(num_filters, name='beta_d0')(context_embedding)
    gamma_d1 = tf.keras.layers.Dense(num_filters, name='gamma_d1')(context_embedding)
    beta_d1 = tf.keras.layers.Dense(num_filters, name='beta_d1')(context_embedding)
    gamma_d2 = tf.keras.layers.Dense(num_filters, name='gamma_d2')(context_embedding)
    beta_d2 = tf.keras.layers.Dense(num_filters, name='beta_d2')(context_embedding)

    decoder_0 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        name='decoder_0')(decoder_input)
    decoder_0 = tf.keras.layers.Multiply()([decoder_0, gamma_d0]) # scale by gamma along channel dim
    decoder_0 = tf.keras.layers.Add()([decoder_0, beta_d0]) # shift with beta along channel dim
    decoder_0 = tf.keras.layers.ReLU()(decoder_0)

    decoder_1 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='decoder_1')(decoder_0)
    decoder_1 = tf.keras.layers.Multiply()([decoder_1, gamma_d1])
    decoder_1 = tf.keras.layers.Add()([decoder_1, beta_d1])
    decoder_1 = tf.keras.layers.ReLU()(decoder_1)

    decoder_2 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        name='decoder_2')(decoder_1)
    decoder_2 = tf.keras.layers.Multiply()([decoder_2, gamma_d2])
    decoder_2 = tf.keras.layers.Add()([decoder_2, beta_d2])
    decoder_2 = tf.keras.layers.ReLU()(decoder_2)

    decoder_output = tf.keras.layers.Conv2DTranspose(filters=3,
        kernel_size=kernel_size[3],
        strides=2,
        padding='same',
        name='decoder_output')(decoder_2)

    return tf.keras.Model([decoder_input, context_embedding], decoder_output, name='decoder_network')

def HyperAnalysisTransform(num_filters=128, context_length=64, kernel_size=[3,5,5]):
    hyper_encoder_input = tf.keras.Input((None,None,num_filters), name='hyper_encoder_input')
    context_embedding = tf.keras.Input(shape=(context_length))

    gamma_he0 = tf.keras.layers.Dense(num_filters, name='gamma_he0')(context_embedding)
    beta_he0 = tf.keras.layers.Dense(num_filters, name='beta_he0')(context_embedding)
    gamma_he1 = tf.keras.layers.Dense(num_filters, name='gamma_he1')(context_embedding)
    beta_he1 = tf.keras.layers.Dense(num_filters, name='beta_he1')(context_embedding)

    hyper_encoder_0 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=1,
        padding='same',
        name='hyper_encoder_0')(hyper_encoder_input)
    hyper_encoder_0 = tf.keras.layers.Multiply()([hyper_encoder_0, gamma_he0])
    hyper_encoder_0 = tf.keras.layers.Add()([hyper_encoder_0, beta_he0])
    hyper_encoder_0 = tf.keras.layers.ReLU()(hyper_encoder_0)

    hyper_encoder_1 = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='hyper_encoder_1')(hyper_encoder_0)
    # hyper_encoder_1 = LinearTransform()(hyper_encoder_1)
    hyper_encoder_1 = tf.keras.layers.ReLU()(hyper_encoder_1)

    hyper_encoder_output = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=2,
        padding='same',
        name='hyper_encoder_output')(hyper_encoder_1)

    return tf.keras.Model([hyper_encoder_input, context_embedding], hyper_encoder_output, name='hyper_encoder_network')

def HyperSynthesisTransform(num_filters=128, context_length=64, kernel_size=[5,5,3]):
    hyper_decoder_input = tf.keras.Input((None,None,num_filters), name='hyper_decoder_input')
    context_embedding = tf.keras.Input(shape=(context_length))

    gamma_hd0 = tf.keras.layers.Dense(num_filters, name='gamma_hd0')(context_embedding)
    beta_hd0 = tf.keras.layers.Dense(num_filters, name='beta_hd0')(context_embedding)
    gamma_hd1 = tf.keras.layers.Dense(num_filters, name='gamma_hd1')(context_embedding)
    beta_hd1 = tf.keras.layers.Dense(num_filters, name='beta_hd1')(context_embedding)

    hyper_decoder_0 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[0],
        strides=2,
        padding='same',
        name='hyper_decoder_0')(hyper_decoder_input)
    hyper_decoder_0 = tf.keras.layers.Multiply()([hyper_decoder_0, gamma_hd0])
    hyper_decoder_0 = tf.keras.layers.Add()([hyper_decoder_0, beta_hd0])
    hyper_decoder_0 = tf.keras.layers.ReLU()(hyper_decoder_0)

    hyper_decoder_1 = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[1],
        strides=2,
        padding='same',
        name='hyper_decoder_1')(hyper_decoder_0)
    hyper_decoder_1 = tf.keras.layers.Multiply()([hyper_decoder_1, gamma_hd1])
    hyper_decoder_1 = tf.keras.layers.Add()([hyper_decoder_1, beta_hd1])
    hyper_decoder_1 = tf.keras.layers.ReLU()(hyper_decoder_1)

    hyper_decoder_output = tf.keras.layers.Conv2DTranspose(filters=num_filters,
        kernel_size=kernel_size[2],
        strides=1,
        padding='same',
        name='hyper_decoder_output')(hyper_decoder_1)

    return tf.keras.Model([hyper_decoder_input, context_embedding], hyper_decoder_output, name='hyper_decoder_network')


class BMSHJ2018Model(tf.keras.Model):
    """Main model class."""
    def __init__(self, lmbda, context_length, num_filters, num_scales, scale_min, scale_max):
        super().__init__()
        self.lmbda = lmbda
        self.num_scales = num_scales
        offset = tf.math.log(scale_min)
        factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
            num_scales - 1.)
        self.context_model = ContextModel(context_length=context_length, num_filters=num_filters)
        self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
        self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
        self.analysis_transform = AnalysisTransform(context_length=context_length, num_filters=num_filters)
        self.hyper_analysis_transform = HyperAnalysisTransform(context_length=context_length, num_filters=num_filters)
        self.hyper_synthesis_transform = HyperSynthesisTransform(context_length=context_length, num_filters=num_filters)
        self.synthesis_transform = SynthesisTransform(context_length=context_length, num_filters=num_filters)
        # self.build((None,input_shape[0],input_shape[1],3))

    def call(self, inputs, training):
        """Computes rate and distortion losses."""
        x, x_c = inputs
        entropy_model = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
            compression=False)
        side_entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=False)

        c = self.context_model(x_c)
        c_tiled = tf.tile(c, [x.shape[0], 1])
        y = self.analysis_transform([x, c_tiled])
        z = self.hyper_analysis_transform([abs(y), c_tiled])
        z_hat, side_bits = side_entropy_model(z, training=training)
        indexes = self.hyper_synthesis_transform([z_hat, c_tiled])
        y_hat, bits = entropy_model(y, indexes, training=training)
        x_hat = self.synthesis_transform([y_hat, c_tiled])

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
    model_name = f'lmbda={args.lmbda}_steps={args.total_steps}_batchsize_c={args.batchsize_context}_batchsize_d={args.batchsize_data}_patchsize={args.patchsize}_filters={args.num_filters}'

    # get training data files for each dataset
    clicm_files = glob.glob('/extra/ucibdl0/preethis/datasets/clic/train_m/*/*.png') # ~1000 images
    clicp_files = glob.glob('/extra/ucibdl0/preethis/datasets/clic/train_p/*/*.png') # ~600 images
    div2k_files =  glob.glob('/extra/ucibdl0/preethis/datasets/div2k/train/*/*.png') # ~800 images

    # coco has a lot more data ~40000 images, so sampling a smaller subset
    coco_files = glob.glob('/extra/ucibdl0/preethis/datasets/coco/train/*.jpg')
    coco_files = np.random.choice(coco_files, args.max_images, replace=False) # 3000 images (unless others specified)
    city_files = glob.glob('/extra/ucibdl0/preethis/datasets/cityscapes/train/*/*.png') # ~3000 images

    # not using because of issues with decoding TIFF image format (TODO: come back to this)
    # raise_files = glob.glob('/extra/ucibdl0/preethis/datasets/RAISE/train/*.TIFF')

    # create separate iters for each dataset (one for context model and another for compression model)
    clicm_iter = create_dataset_iter(clicm_files, args.batchsize_data, args.patchsize, train=True)
    clicm_context_iter = create_dataset_iter(clicm_files, args.batchsize_context, args.patchsize, train=True)

    clicp_iter = create_dataset_iter(clicp_files, args.batchsize_data, args.patchsize, train=True)
    clicp_context_iter = create_dataset_iter(clicp_files, args.batchsize_context, args.patchsize, train=True)

    coco_iter = create_dataset_iter(coco_files, args.batchsize_data, args.patchsize, train=True)
    coco_context_iter = create_dataset_iter(coco_files, args.batchsize_context, args.patchsize, train=True)

    div2k_iter = create_dataset_iter(div2k_files, args.batchsize_data, args.patchsize, train=True)
    div2k_context_iter = create_dataset_iter(div2k_files, args.batchsize_context, args.patchsize, train=True)

    city_iter = create_dataset_iter(city_files, args.batchsize_data, args.patchsize, train=True)
    city_context_iter = create_dataset_iter(city_files, args.batchsize_context, args.patchsize, train=True)

    train_iter_list = [[clicm_iter, clicm_context_iter],
                       [clicp_iter, clicp_context_iter],
                       [coco_iter, coco_context_iter],
                       [div2k_iter, div2k_context_iter],
                       [city_iter, city_context_iter]]

    # construct model
    model = BMSHJ2018Model(args.lmbda, args.context_length, args.num_filters, NUM_SCALES, SCALE_MIN, SCALE_MAX)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    if args.ckpt:
        model.load_weights(ckpt)

    # training
    print('begin training')
    train_log_dir = f'FiLM/logs/context/{model_name}'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    for step in range(args.start_step, args.total_steps):
        # sample a dataset
        index = random.randint(0, len(train_iter_list)-1) # endpoints are inclusive
        dataset = train_iter_list[index]
        # get batch of data from dataset iterators (separate iterators for data and context)
        data = [next(dataset[0]), next(dataset[1])]
        # compute loss
        loss_dict  = model.train_step(data)

        if step % 500 == 0 and step > 0: # to keep track of progress
            print(step, loss_dict)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_dict['loss'].numpy(), step=step)
                tf.summary.scalar('bpp', loss_dict['bpp'].numpy(), step=step)
                tf.summary.scalar('mse', loss_dict['mse'].numpy(), step=step)
            if step % 10000 == 0:
                model.save_weights(f'FiLM/checkpoints/context/{model_name}')

    # save model
    model.save(f'FiLM/models/context/{model_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--total_steps", type=int, default=1000000,
                        help="Total number of training steps/iterations")
    parser.add_argument("--start_step", type=int, default=0,
                        help="Start step (if loading checkpoint)")
    parser.add_argument("--patchsize", type=int, default=256,
                        help="Size of image patches for training and validation")
    parser.add_argument("--context_length", type=int, default=64,
                        help="Length of context embedding (output of context network)")
    parser.add_argument("--num_filters", type=int, default=128,
                        help="Number of filters per layer")
    parser.add_argument("--batchsize_data", type=int, default=16,
                        help="Batch size used to train compression model")
    parser.add_argument("--batchsize_context", type=int, default=256,
                        help="Batch size used to train context model")
    parser.add_argument("--lmbda", type=float, default=0.01,
                        help="Lambda used for rate-distortion tradeoff")
    parser.add_argument("--max_images", type=int, default=3000,
                        help="Maximum number of images a dataset can have")
    parser.add_argument("--ckpt", default='',
                        help="Load checkpoint if using")

    args = parser.parse_args()

    train(args)
