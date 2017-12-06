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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log_test', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 1024
NUM_CLASSES = 13

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ALL_FILES = provider.getDataFiles('indoor3d_sem_seg_hdf5_data/all_files.txt')
# room_filelist = [line.rstrip() for line in open('indoor3d_sem_seg_hdf5_data/room_filelist.txt')]
#
# # Load ALL data
# data_batch_list = []
# label_batch_list = []
# for h5_filename in ALL_FILES:
#     data_batch, label_batch = provider.loadDataFile(h5_filename)
#
#     print ("one batch size: " + str(data_batch.shape))
#
#     data_batch_list.append(data_batch)
#     label_batch_list.append(label_batch)
# data_batches = np.concatenate(data_batch_list, 0)
# label_batches = np.concatenate(label_batch_list, 0)
# print(data_batches.shape)
# print(label_batches.shape)
# print (type(data_batches))
# print (type(label_batches))
#
# test_area = 'Area_'+str(FLAGS.test_area)
# train_idxs = []
# test_idxs = []
# for i,room_name in enumerate(room_filelist):
#     if test_area in room_name:
#         test_idxs.append(i)
#     else:
#         train_idxs.append(i)
#
# train_data = data_batches[train_idxs,...]
# train_label = label_batches[train_idxs]
# test_data = data_batches[test_idxs,...]
# test_label = label_batches[test_idxs]
# print(train_data.shape, train_label.shape)
# print(test_data.shape, test_label.shape)



ALL_FILES = provider.getDataFiles('converted_KITTI/allFrames.txt')
# Load ALL data
data_batch_list = []
label_batch_list = []

for h5_filename in ALL_FILES:
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
# print(data_batches.shape)
# print(label_batches.shape)
# print (type(data_batches))
# print (type(label_batches))

data_batches = data_batches[:181248,...].reshape(177, 1024, 9)
label_batches = label_batches[:181248,...].reshape(177, 1024, )

train_data = data_batches[:141,...]
train_label = label_batches[:141]
test_data = data_batches[142:,...]
test_label = label_batches[142:]
# print(train_data.shape, train_label.shape)
# print(test_data.shape, test_label.shape)




def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
        predict_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'predict'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_" + str(epoch) + ".ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label) 
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 10 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)

        print ("**************")
        print ("The type of pred_val")
        print (type(pred_val))
        print (pred_val.shape)
        print (pred_val[0])
        print ("**************")

        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[int(l)] += 1
                total_correct_class[int(l)] += (pred_val[i-start_idx, j] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class[:2])/np.array(total_seen_class[:2], dtype=np.float))))  # Here, we only have two classes. Therefore, just select the first two bits.
    print (total_correct_class)
    print (total_seen_class)


# prepare your own data and label to do predict
datafile_directory = 'predict/frame_10.h5'
def prepare_input(datafile_directory):
    # Load ALL data
    data_batch_list = []
    label_batch_list = []

    data_batch, label_batch = provider.loadDataFile(datafile_directory)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)

    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)

    remain = int(data_batches.shape[0] / NUM_POINT) # here is 1024
    floor_to = remain * NUM_POINT

    data_batches = data_batches[:floor_to, ...].reshape(remain, 1024, 9)
    label_batches = label_batches[:floor_to, ...].reshape(remain, 1024, )

    return data_batches, label_batches

def predict_test():
    car_points = []
    checkpoint_directory = "/Users/kevin_sct/deepLearningStudy/pointnet/sem_seg/log/"
    predict_data, predict_label = prepare_input(datafile_directory)
    with tf.Graph().as_default():
        pointclouds_pl, _ = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
        batch = tf.Variable(0)
        bn_decay = get_bn_decay(batch)
        tf.summary.scalar('bn_decay', bn_decay)
        # Get model
        pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
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

if __name__ == "__main__":
    #train()

    predict_test()

    LOG_FOUT.close()