import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from model import *

NUM_POINT = 1024
BATCH_SIZE = 2

def predict_test():
    car_points = []
    checkpoint_directory = "/Users/kevin_sct/deepLearningStudy/pointnet/sem_seg/log/"

    # prepare your own data and label to do predict
    datafile_directory = 'predict/frame_10.h5'

    def log_string(out_str):
        print(out_str)

    def prepare_input(datafile_directory):
        # Load ALL data
        data_batch_list = []
        label_batch_list = []

        data_batch, label_batch = provider.loadDataFile(datafile_directory)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)

        data_batches = np.concatenate(data_batch_list, 0)
        label_batches = np.concatenate(label_batch_list, 0)

        remain = int(data_batches.shape[0] / NUM_POINT)  # here is 1024
        floor_to = remain * NUM_POINT

        data_batches = data_batches[:floor_to, ...].reshape(remain, 1024, 9)
        label_batches = label_batches[:floor_to, ...].reshape(remain, 1024, )

        return data_batches, label_batches

    predict_data, predict_label = prepare_input(datafile_directory)

    with tf.Graph().as_default():
        pointclouds_pl, _ = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
        batch = tf.Variable(0)
        # Get model
        pred = get_model(pointclouds_pl, is_training_pl)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print ("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(checkpoint_directory)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            log_string('----')
            current_data = predict_data[:, 0:NUM_POINT, :]

            print ("current_data.shape: ")
            print (current_data.shape)

            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE

            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE

                pred_val = sess.run(pred, feed_dict = {pointclouds_pl: predict_data[start_idx:end_idx, :, :], is_training_pl: False})
                pred_val = np.argmax(pred_val, 2)

                print ("************")
                print (type(pred_val))
                print (pred_val.shape)
                #print (pred_val[0][900:920])
                print ("************")

                for i in range(BATCH_SIZE):
                    offset = 0
                    for predicted_value in pred_val[i].tolist():

                        # if offset < 10:
                        #     print (predicted_value)

                        if predicted_value != 0:
                            car_points.append(current_data[start_idx + i][offset])
                        offset += 1

            print (len(car_points))
            print (car_points[0])

            return car_points