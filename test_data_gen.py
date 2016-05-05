
## Test HDF5 data

from scipy.ndimage import imread
import numpy as np
import h5py
import os

local = '/Users/nicholasmoschopoulos/Documents/Classes/Semester8/CS280/project/data/clipped/hitch_hiker/offsets.npz'
ec2 = '/mnt/data/clipped/offsets.npz'

def offsets_match(h5_dir, offset_path=ec2):
  offsets = np.load(offset_path)['offsets']
  rand_sample = np.random.randint(0, len(offsets) - 100, 10000)
  files = os.path.join(h5_dir, 'frame-%06d.h5')
  match = True
  for i in rand_sample:
    with h5py.File(files % i, 'r') as hf:
      d = np.array(hf.get('label'))
      match = (match and (int(d[0, 0, 0, 0]) == int(offsets[i])))
  return match

def frames_match(h5_dir, frames_dir, offset_path=ec2):
  offsets = np.load(offset_path)['offsets']
  rand_sample = np.random.randint(0, len(offsets) - 100, 10000)
  h5s = os.path.join(h5_dir, 'frame-%06d.h5')
  frames = os.path.join(frames_dir, 'frame-%06d.jpg')
  match = True
  for i in start:
    image = imread(frames % i)/255.0
    image = np.dot(image[:, :, :3], [0.299, 0.587, 0.114])
    with h5py.File(h5s % i, 'r') as hf:
      left_arr = np.array(hf.get('left'))
      right_arr = np.array(hf.get('right'))
      offset = int(offsets[i])
      match = match and np.all(left_arr[0, 0, offset:10, :, :] == right_arr[0, 0, :10 - offset, :, :])
      match = match and np.all(left_arr[0, 0, 0, :, :] == image)
  return match

def test_binary_data(h5_dir):
  per1, per0, total = 0, 0, float(0)
  rand_sample = np.random.randint(0, len(os.listdir(h5_dir)) - 100, 10000)
  h5s = os.path.join(h5_dir, 'frame-%06d.h5')
  failed = []
  for i in rand_sample:
    with h5py.File(h5s % i, 'r') as hf:
      label = np.array(hf.get('label'))
      left = np.array(hf.get('left'))
      right = np.array(hf.get('right'))
      if int(label[0, 0, 0, 0]) == 1:
        if not np.all(left == right): failed.append(i)
        per1 += 1
      else:
        if np.all(left == right): failed.append(i)
        per0 += 1
      total += 1
  return per0/total, per1/total, failed

def view_side_by_side_hdf5(frame_num, h5_dir):
  frame_path = os.path.join(frame_dir, 'frame-%06d.h5' % frame_num)
  from matplotlib import pyplot as plt
  import matplotlib.gridspec as gridspec
  gs1 = gridspec.GridSpec(10, 10)
  gs1.update(wspace=0.025, hspace=0.05)
  ax = []
  with h5py.File(frame_path, 'r') as hf:
    offset = np.array(hf.get('label'))[0, 0, 0, 0]
    left_ims = np.array(hf.get('left'))
    right_ims = np.array(hf.get('right'))
    print offset
    for v in xrange(10):
      ax.append(plt.subplot(gs1[v]))
      plt.imshow(left_ims[0, 0, v, :, :], cmap='gray', vmin=0, vmax=1)
      ax.append(plt.subplot(gs1[v + 10]))
      plt.imshow(right_ims[0, 0, v, :, :], cmap='gray', vmin=0, vmax=1)
    for a in ax:
      a.set_xticklabels([])
      a.set_yticklabels([])
    plt.show()

