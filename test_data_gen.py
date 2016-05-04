
## Test HDF5 data

from scipy.ndimage import imread
import numpy as np
import h5py
import os

local = '/Users/nicholasmoschopoulos/Documents/Classes/Semester8/CS280/project/data/clipped/hitch_hiker/offsets.npz'
ec2 = '/mnt/data/clipped/offsets.npz'

offsets = np.load(ec2)['offsets']
match = True
start = np.random.randint(0, len(offsets) - 100, 1000)
for i in start:
  with h5py.File('frame-%06d.h5' % i, 'r') as hf:
    d = np.array(hf.get('label'))
    match = (match and int(d[0, 0, 0, 0]) == int(offsets[i]))

print match

frames_correct = True
for i in start:
  with h5py.File('frame-%06d.h5' % i, 'r') as hf:
    left_arr = np.array(hf.get('left'))
    right_arr = np.array(hf.get('right'))
    offset = int(offsets[i])
    frames_correct = frames_correct and np.all(left_arr[0, 0, offset:10, :, :] == right_arr[0, 0, :10 - offset, :, :])

print frames_correct


frames_correct = True
for i in start:
  image = imread('/mnt/data/clipped/frame-%06d.jpg' % i)/255.0
  image = np.dot(image[:, :, :3], [0.299, 0.587, 0.114])
  with h5py.File('frame-%06d.h5' % i, 'r') as hf:
    left_arr = np.array(hf.get('left'))
    frames_correct = frames_correct and np.all(left_arr[0, 0, 0, :, :] == image)
    if not frames_correct: 
      print i
      break

print frames_correct


def view_side_by_side_hdf5(frame_num):
  frame_path = '/mnt/data/clipped/hh_data/frame-%06d.h5' % frame_num
  from matplotlib import pyplot as plt
  import matplotlib.gridspec as gridspec
  gs1 = gridspec.GridSpec(10, 10)
  gs1.update(wspace=0.025, hspace=0.05)
  ax = []
  with h5py.File(frame_path, 'r') as hf:
    offset = np.array(hf.get('label'))[0, 0, 0, 0]
    left_ims = np.array(hf.get('left'))
    right_ims = np.array(hf.get('right'))
    for v in xrange(10):
      ax.append(plt.subplot(gs1[v]))
      plt.imshow(left_ims[0, 0, v, :, :], cmap='gray')
      ax.append(plt.subplot(gs1[v + 10]))
      plt.imshow(right_ims[0, 0, v, :, :], cmap='gray')
    for a in ax:
      a.set_xticklabels([])
      a.set_yticklabels([])
    plt.show()

