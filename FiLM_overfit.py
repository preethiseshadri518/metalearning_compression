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
from FiLM_baseline_model import *


def train(args):
    lmbda = float(args.trained_model_dir.split('lmbda=')[1].split('_')[0])
    batchsize = int(args.trained_model_dir.split('batchsize=')[1].split('_')[0])
    patchsize = int(args.trained_model_dir.split('patchsize=')[1].split('_')[0])
    total_steps = int(args.trained_model_dir.split('steps=')[1].split('_')[0])
    num_filters = int(args.trained_model_dir.split('filters=')[1][:-1])
    print(lmbda, batchsize, patchsize, num_filters, args.lr, args.overfit_steps)

    # get training data files for each dataset
    print('get training data on overfitting datasets')
    # coco has a lot more data ~40000 images, so sampling a smaller subset
    coco_train_files = glob.glob('/extra/ucibdl0/preethis/datasets/coco/train/*.jpg')
    coco_train_files = np.random.choice(coco_train_files, args.max_images, replace=False)
    city_train_files = glob.glob('/extra/ucibdl0/preethis/datasets/cityscapes/train/*/*.png') # ~3000 images

    # create separate iters for each dataset
    coco_train_iter = create_dataset_iter(coco_train_files, batchsize, patchsize, 'coco', train=True)
    city_train_iter = create_dataset_iter(city_train_files, batchsize, patchsize, 'cityscapes', train=True)

    # get test data files for each dataset and create dataset iters
    print('get test data')
    coco_test_files = glob.glob('/extra/ucibdl0/preethis/datasets/coco/test/*.jpg')
    coco_test_files = np.random.choice(coco_test_files, size=2000, replace=False)
    city_test_files = glob.glob('/extra/ucibdl0/preethis/datasets/cityscapes/test/*/*.png')

    # create iterator for each dataset
    coco_test_iter = create_dataset_iter(coco_test_files, batchsize, patchsize, 'coco', train=False)
    city_test_iter = create_dataset_iter(city_test_files, batchsize, patchsize, 'cityscapes', train=False)

    iter_dict = {'coco': [coco_train_iter, coco_test_iter],
                 'cityscapes': [city_train_iter, city_test_iter]}
    print('datasets', iter_dict.keys())

    # construct model
    model = BMSHJ2018Model(lmbda, num_filters, NUM_SCALES, SCALE_MIN, SCALE_MAX)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))

    # training
    base = f'lmbda={lmbda}_steps={total_steps}_overfit={args.overfit_steps}_batchsize={batchsize}_patchsize={patchsize}_filters={num_filters}_lr={args.lr}'
    metrics_path = 'FiLM_datasub/overfit.csv'
    for name, data_iters in iter_dict.items():
        print(f'begin training for {name}')
        model_name = f'dataset={name}_{base}'
        print(model_name)
        # reload weights each time
        model.load_weights(args.trained_model_dir)
        print([np.sum(x) for x in model.get_weights()])

        # specify new directories each time
        train_log_dir = f'FiLM_datasub/logs_overfit/baseline/{model_name}'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for step in range(args.overfit_steps):
            data = next(data_iters[0])
            # compute loss
            loss_dict  = model.train_step(data, var_sub = ['linear_transform'])

            # log training progress
            if step % 500 == 0:
                print(step, loss_dict)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_dict['loss'].numpy(), step=step)
                    tf.summary.scalar('bpp', loss_dict['bpp'].numpy(), step=step)
                    tf.summary.scalar('mse', loss_dict['mse'].numpy(), step=step)
                if step % args.ckpt_freq == 0: # checkpoint
                    model.save_weights(f'FiLM_datasub/checkpoints/baseline/{model_name}/')

        # save model
        model.save_weights(f'FiLM_datasub/checkpoints_overfit/baseline/{model_name}/')
        model.save(f'FiLM_datasub/models_overfit/baseline/{model_name}')

        # compute metrics on test datasets
        total_loss_list = []
        bpp_loss_list = []
        mse_loss_list = []
        psnr_loss_list = []

        for en, batch in enumerate(data_iters[-1]):
            loss, bpp, mse  = model(batch, training=False)
            total_loss_list.append(loss.numpy())
            bpp_loss_list.append(bpp.numpy())
            mse_loss_list.append(mse.numpy())
            psnr = 20*np.log10(255) - 10*np.log10(mse)
            psnr_loss_list.append(psnr)

        print(name, np.average(total_loss_list), np.average(mse_loss_list), np.average(bpp_loss_list), np.average(psnr_loss_list))

        if path.exists(metrics_path):
            with open(metrics_path, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([str(datetime.datetime.now()), model_name, name, args.overfit_steps, args.lr, lmbda, np.average(total_loss_list), np.average(mse_loss_list), np.average(bpp_loss_list), np.average(psnr_loss_list)])
        else:
            with open(metrics_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time", "Model Name", "Dataset", "Overfit Steps", "Lr", "Lambda", "Total Loss", "MSE", "Rate (BPP)", "PSNR"])
                writer.writerow([str(datetime.datetime.now()), model_name, name, args.overfit_steps, args.lr, lmbda, np.average(total_loss_list), np.average(mse_loss_list), np.average(bpp_loss_list), np.average(psnr_loss_list)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--overfit_steps", type=int, default=200000,
                        help="Total number of training steps/iterations")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for overfitting")
    parser.add_argument("--max_images", type=int, default=5000,
                        help="Upper limit on number of images to use")
    parser.add_argument("--ckpt_freq", type=int, default=2000,
                        help="How frequently (number of steps) to save ckpt")
    parser.add_argument("--trained_model_dir",
                        help="trained baseline model directory")

    args = parser.parse_args()

    train(args)
