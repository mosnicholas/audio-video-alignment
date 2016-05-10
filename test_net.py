import caffe
import h5py
import numpy as np

net = caffe.Net('/home/ubuntu/video_align/videonet_deploy.prototxt', '/mnt/data/snapshots/videonet_iter_530.caffemodel', caffe.TEST)

def test_frame(frame_num):
  frame_path = '/mnt/data/clipped/hh_data/frame-%06d.h5' % frame_num
  with h5py.File(frame_path, 'r') as hf:
    left = np.array(hf.get('left'))
    right = np.array(hf.get('right'))
    label = np.array(hf.get('label'))
  net.blobs['left'].data[...] = left
  net.blobs['right'].data[...] = right
  out = net.forward()
  if int(label[0, 0, 0, 0]) == out['probs'].argmax():
    return True
  else:
    return False

def test_synthetic_net():
  height, width = 90, 120
  white_frame = np.random.rand(height, width)
  black_frame = np.zeros((height, width))
  right = np.ones((1, 1, 10, height, width))
  left = np.ones((1, 1, 10, height, width))
  left_i, right_i = 0, 3
  left[0, 0, left_i, :, :] = white_frame
  right[0, 0, right_i, :, :] = white_frame
  net.blobs['left'].data[...] = left
  net.blobs['right'].data[...] = right
  out = net.forward()
  return out['probs'].argmax() == np.abs(left_i - right_i)


def view_side_by_side_hdf5(frame_num):
  frame_path = '/mnt/data/clipped/hh_data/frame-%06d.h5' % frame_num
  from matplotlib import pyplot as plt
  import matplotlib.gridspec as gridspec
  ax = []
  with h5py.File(frame_path, 'r') as hf:
    left_ims = np.array(hf.get('left'))
    right_ims = np.array(hf.get('right'))
  net.blobs['left'].data[...] = left_ims
  net.blobs['right'].data[...] = right_ims
  out = net.forward()
  label = out['probs'].argmax()
  gs1 = gridspec.GridSpec(2, 20 - label)
  gs1.update(wspace=0.025, hspace=0.05)
  for v in xrange(10):
    ax.append(plt.subplot(gs1[v]))
    plt.imshow(left_ims[0, v, :, :], cmap='gray')
    ax.append(plt.subplot(gs1[v + label + 20]))
    plt.imshow(right_ims[0, v - label, :, :], cmap='gray')
  for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])
  plt.show()

# solver = caffe.SGDSolver('/home/ubuntu/video_align/siamese_videonet_solver.prototxt')
# for i in [(blob, solver.net.blobs['conv1'].data.shape) for blob in solver.net.blobs]: print i
# solver.restore('/mnt/data/snapshots/videonet_iter_18402.solverstate')


