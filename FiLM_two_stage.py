import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
import glob
import argparse
import random
import csv
import datetime
from os import path

from data_helpers import *
from FiLM_cond_model import *


def train(args):
    lmbda = float(args.trained_model_dir.split('lmbda=')[1].split('_')[0])
    patchsize = int(args.trained_model_dir.split('patchsize=')[1].split('_')[0])
    total_steps = int(args.trained_model_dir.split('steps=')[1].split('_')[0])
    num_filters = int(args.trained_model_dir.split('filters=')[1][:-1])
    print(lmbda, total_steps, args.context_length, patchsize, num_filters)

    # get training data files for each dataset
    clicm_files = glob.glob('/extra/ucibdl0/preethis/datasets/clic/train_m/*/*.png') # ~1000 images
    clicp_files = glob.glob('/extra/ucibdl0/preethis/datasets/clic/train_p/*/*.png') # ~600 images
    clic_files = clicm_files + clicp_files # combining mobile and professional
    celeba_files = glob.glob('/extra/ucibdl0/preethis/datasets/celeb_a/data/celeba-tfr/train/*')

    # create separate iters for each dataset (one for context model and another for compression model)
    clic_iter = create_dataset_iter(clic_files, args.batchsize_data, patchsize, 'clic', train=True)
    clic_context_iter = create_dataset_iter(clic_files, args.batchsize_context, patchsize, 'clic', train=True)

    celeba_iter = create_dataset_iter(celeba_files, args.batchsize_data, patchsize, 'celeb_a', train=True)
    celeba_context_iter = create_dataset_iter(celeba_files, args.batchsize_context, patchsize, 'celeb_a', train=True)

    train_iter_list = [[clic_iter, clic_context_iter],
                       [celeba_iter, celeba_context_iter]]

    # construct context model
    model = BMSHJ2018Model(lmbda, args.context_length, num_filters, NUM_SCALES, SCALE_MIN, SCALE_MAX)

    if args.ckpt: # load partially trained checkpoint
        print('checkpoint:', args.ckpt)
        model.load_weights(args.ckpt)
    else: # otherwise load trained baseline model weights into initialized context model
        dummy_input = np.random.random((1,256,256,3))
        model([dummy_input, dummy_input]) # need this to create weights

        # get tensors/weights from context model
        names_base = [weight.name for weight in model.weights]
        weights_base = model.get_weights()

        # load trained model
        trained_model = tf.keras.models.load_model(args.trained_model_dir)
        print('trained model performance')
        print(trained_model(next(clic_iter)))

        # get tensors/weights from trained model
        names_trained = [weight.name for weight in trained_model.weights]
        weights_trained = trained_model.get_weights()
        weights_trained_dict = {}
        for name, weight in zip(names_trained, weights_trained):
            weights_trained_dict[name] = weight

        # create list of new weights for model
        # iterate over tensors/weights in model
        # if trained_model also has tensor, update weight with trained weights
        # otherwise just use initialized weights
        new_weights = []
        for en, (name, weight) in enumerate(zip(names_base, weights_base)):
            if name in weights_trained_dict:
                new_weights.append(weights_trained_dict[name])
            else:
                new_weights.append(weight)

        model.set_weights(new_weights)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    print('starting training')
    # training
    model_name = f'lmbda={lmbda}_steps1={total_steps}_steps2={args.context_steps}_batchsize_d={args.batchsize_data}_batchsize_c={args.batchsize_context}_patchsize={patchsize}_filters={num_filters}'
    print(model_name)
    train_log_dir = f'FiLM_datasub/logs/two_stage/{model_name}'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    for step in range(args.start_step, args.context_steps):
        # sample a dataset
        index = random.randint(0, len(train_iter_list)-1) # endpoints are inclusive
        dataset = train_iter_list[index]
        data = [next(dataset[0]), next(dataset[1])]
        # compute loss
        # only compute gradients for context model + FiLM layers
        loss_dict  = model.train_step(data, var_sub=['context', 'beta', 'gamma'])

        # log training progress
        if step % 500 == 0:
            print(step, loss_dict)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_dict['loss'].numpy(), step=step)
                tf.summary.scalar('bpp', loss_dict['bpp'].numpy(), step=step)
                tf.summary.scalar('mse', loss_dict['mse'].numpy(), step=step)
            if step % args.ckpt_freq == 0: # checkpoint
                model.save_weights(f'FiLM_datasub/checkpoints/two_stage/{model_name}/')

    # save model
    model.save_weights(f'FiLM_datasub/checkpoints/two_stage/{model_name}/')
    model.save(f'FiLM_datasub/models/two_stage/{model_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--context_steps", type=int, default=500000,
                        help="Total number of training steps/iterations")
    parser.add_argument("--start_step", type=int, default=0,
                        help="Start step (relevant if loading checkpoint)")
    parser.add_argument("--context_length", type=int, default=64,
                        help="Length of context embedding (output of context network)")
    parser.add_argument("--batchsize_data", type=int, default=16,
                        help="Batch size used to train compression model")
    parser.add_argument("--batchsize_context", type=int, default=256,
                        help="Batch size used to train context model")
    parser.add_argument("--ckpt_freq", type=int, default=2000,
                        help="How frequently (number of steps) to save checkpoint")
    parser.add_argument("--ckpt", default='',
                        help="Checkpoint for partially trained two-stage model")
    parser.add_argument("--trained_model_dir",
                        help="trained baseline model directory")

    args = parser.parse_args()

    train(args)
