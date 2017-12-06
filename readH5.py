import h5py

filename = './sem_seg/indoor3d_sem_seg_hdf5_data/ply_data_all_0.h5'
#filename = './sem_seg/converted_KITTI/frame_10.h5'
f = h5py.File(filename, 'r')


data_file = f['data'][:]
label_file = f['label'][:]

print (data_file.shape, label_file.shape)

print (type(label_file[0]))