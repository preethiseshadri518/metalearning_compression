import tensorflow as tf
import tensorflow_compression as tfc
import glob
import argparse
import random
import csv
import datetime
from os import path

from data_helpers import *
from FiLM_cond_model import *


def train(args):
    print(args)
    model_name = f'lmbda={args.lmbda}_steps={args.total_steps}_batchsize_c={args.batchsize_context}_batchsize_d={args.batchsize_data}_patchsize={args.patchsize}_filters={args.num_filters}'

    # get training data files for each dataset
    clicm_files = glob.glob('/extra/ucibdl0/preethis/datasets/clic/train_m/*/*.png') # ~1000 images
    clicp_files = glob.glob('/extra/ucibdl0/preethis/datasets/clic/train_p/*/*.png') # ~600 images
    clic_files = clicm_files + clicp_files # combining mobile and professional
    celeba_files = glob.glob('/extra/ucibdl0/preethis/datasets/celeb_a/data/celeba-tfr/train/*')

    # create separate iters for each dataset (one for context model and another for compression model)
    clic_iter = create_dataset_iter(clic_files, args.batchsize_data, args.patchsize, 'clic', train=True)
    clic_context_iter = create_dataset_iter(clic_files, args.batchsize_context, args.patchsize, 'clic', train=True)

    celeba_iter = create_dataset_iter(celeba_files, args.batchsize_data, args.patchsize, 'celeb_a', train=True)
    celeba_context_iter = create_dataset_iter(celeba_files, args.batchsize_context, args.patchsize, 'celeb_a', train=True)

    train_iter_list = [[clic_iter, clic_context_iter],
                       [celeba_iter, celeba_context_iter]]

    # construct model
    model = BMSHJ2018Model(args.lmbda, args.context_length, args.num_filters, NUM_SCALES, SCALE_MIN, SCALE_MAX)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    # load checkpoint if using
    if args.ckpt:
        model.load_weights(args.ckpt)

    # training
    print('begin training')
    train_log_dir = f'FiLM_datasub/logs/context/{model_name}'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    for step in range(args.start_step, args.total_steps):
        # sample a dataset
        index = random.randint(0, len(train_iter_list)-1) # endpoints are inclusive
        dataset = train_iter_list[index]
        # get batch of data from dataset iterators (separate iterators for data and context)
        data = [next(dataset[0]), next(dataset[1])]
        # compute loss
        loss_dict  = model.train_step(data, var_sub=None)

        if step % 500 == 0: # to keep track of progress
            print(step, loss_dict)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_dict['loss'].numpy(), step=step)
                tf.summary.scalar('bpp', loss_dict['bpp'].numpy(), step=step)
                tf.summary.scalar('mse', loss_dict['mse'].numpy(), step=step)
            if step % args.ckpt_freq == 0:
                model.save_weights(f'FiLM_datasub/checkpoints/context/{model_name}/')

    # save model
    model.save_weights(f'FiLM_datasub/checkpoints/context/{model_name}/')
    model.save(f'FiLM_datasub/models/context/{model_name}')


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
    parser.add_argument("--ckpt_freq", type=int, default=5000,
                        help="How frequently (number of steps) to save checkpoint")
    parser.add_argument("--ckpt", default='',
                        help="Load checkpoint if using")

    args = parser.parse_args()

    train(args)
