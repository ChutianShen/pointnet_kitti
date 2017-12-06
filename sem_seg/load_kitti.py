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
print(data_batches.shape)
print(label_batches.shape)
print (type(data_batches))
print (type(label_batches))

data_batches = data_batches.reshape(22, 1000, 9)
label_batches = label_batches.reshape()