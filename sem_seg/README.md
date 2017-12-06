## Semantic Segmentation of Indoor Scenes

### Dataset

First, use hdf5.ipynb to convert KITTI for detection to semantic segmentation. (Modify the basedir above the load_dataset function)

Second, use write_allFrames.py to generate a txt file containing all paths of all post-processed frames, which is the input of sem_seg of pointnet, together with the converted KITTI data.

Be careful on each directory path.

### Training

    python train_kitti.py 

Some important parameters: --max_epoch, --log_dir.
More details in the code. I didn't make change on original pointnet code on this part (add_argument).

Modify "ALL_FILES" path.
In the code, I hardcode the train_data, train_label, test_data, test_label size. Please modify them to suit your input data.

### Testing

python test.py


### Testing
The code is messy now. Clean the code firstly.
