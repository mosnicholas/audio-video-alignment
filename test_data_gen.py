
## Test LMDB data

import lmdb
import numpy as np

for x in ('labels', 'frames'):
  for y in ('test', 'train'):
    for i in xrange(4):
      db = '%s_%s_%d' % (x, y, i)
      if lmdb.open(db).stat()['entries'] > 0:
        print db, lmdb.open(db).stat()['entries']

offsets = np.load('offsets.npz')['offsets']
match = True
for i in xrange(4):
  with lmdb.open('labels_test_%d' % i).begin() as txn:
    for k, v in txn.cursor():
      match = (match and int(v) == int(offsets[int(k)]))

print match

frames_correct = True
for i in xrange(4):
  with lmdb.open('frames_test_%d' % i).begin() as txn, \
        lmdb.open('labels_test_%d' % i).begin() as labels_txn:
    for x in xrange(i * 25, (i + 1) * 25):
      key = str(x)
      arr = np.fromstring(txn.get(key)).reshape(180, 240, 20)
      offset = int(labels_txn.get(key))
      frames_correct = frames_correct and np.all(arr[:, :, offset - 1:10] == arr[:, :, 10:10 + (10 - (offset - 1))])
      frames_correct = frames_correct and np.all(arr[:, :, 10 + (10 - (offset - 1)):] == 0)

print frames_correct

def view_side_by_side_lmdb():
  from matplotlib import pyplot as plt
  ax = []
  with lmdb.open('frames_test_0').begin() as txn, \
        lmdb.open('labels_test_0').begin() as labels_txn:
    for u in xrange(25):
      offset = labels_txn.get(str(u))
      arr = np.fromstring(txn.get(str(u))).reshape(180, 240, 20)
      for v in xrange(10):
        ax.append(plt.subplot(2, 10, v + 1))
        plt.imshow(arr[:, :, v], cmap='gray')
      for y in xrange(10):
        ax.append(plt.subplot(2, 10, 10 + y + 1))
        plt.imshow(arr[:, :, 10 + y], cmap='gray')
      for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')
      print 'running for key: %d, with offset %s. It matches actual data: %r' % (u, offset, int(offset) == int(offsets[u]))
      plt.subplots_adjust(wspace=0, hspace=0)
      plt.show()


## Test HDF5 data

import numpy as np
import h5py
import os

local = '/Users/nicholasmoschopoulos/Documents/Classes/Semester8/CS280/project/data/clipped/hitch_hiker/offsets.npz'
ec2 = '/mnt/data/clipped/hitch_hiker/offsets.npz'

offsets = np.load(ec2)['offsets']
match = True
start = np.random.randint(0, len(offsets) - 1000)
for i in xrange(start, start + 1000):
  with h5py.File('frame-%06d.h5' % i, 'r') as hf:
    d = np.array(hf.get('label'))
    match = (match and int(d[0, 0, 0, 0]) == int(offsets[i]))

print match

frames_correct = True
for i in xrange(start, start + 1000):
  with h5py.File('frame-%06d.h5' % i, 'r') as hf:
    left_arr = np.array(hf.get('left'))
    right_arr = np.array(hf.get('right'))
    offset = int(offsets[i])
    frames_correct = frames_correct and np.all(left_arr[0, offset:10, :, :] == right_arr[0, :10 - offset, :, :])

print frames_correct

def view_side_by_side_hdf5(frames):
  from matplotlib import pyplot as plt
  import matplotlib.gridspec as gridspec
  gs1 = gridspec.GridSpec(10, 10)
  gs1.update(wspace=0.025, hspace=0.05)
  ax = []
  with h5py.File(frames, 'r') as hf:
    offset = np.array(hf.get('label'))[0, 0, 0, 0] - 1
    left_ims = np.array(hf.get('left'))
    right_ims = np.array(hf.get('right'))
    for v in xrange(10):
      ax.append(plt.subplot(gs1[v]))
      plt.imshow(left_ims[0, v, :, :], cmap='gray')
      ax.append(plt.subplot(gs1[v + 10]))
      plt.imshow(right_ims[0, v, :, :], cmap='gray')
    for a in ax:
      a.set_xticklabels([])
      a.set_yticklabels([])
    plt.show()

