f = open('allFrames.txt', 'w')
for i in range(108):
    f.write('converted_KITTI/frame_' + str(i) + '.h5' + '\n')

f.close()