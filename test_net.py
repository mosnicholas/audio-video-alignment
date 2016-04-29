import caffe
import h5py
import numpy as np

net = caffe.Net('/home/ubuntu/video_align/videonet_deploy.prototxt', '/mnt/data/snapshots/videonet_iter_5000.caffemodel', caffe.TEST)

def test_frame(frame):
  with h5py.File('/mnt/data/clipped/hh_data/frame-126630.h5', 'r') as hf:
    left = np.array(hf.get('left'))
    right = np.array(hf.get('right'))
    label = np.array(hf.get('label'))

  net.blobs['left'].data[...] = left
  net.blobs['right'].data[...] = right

  out = net.forward()
  if int(label[0, 0, 0, 0]) == out['probs'].argmax():
    print 'frame matched successfully'
  else:
    print 'frame failed'

def view_side_by_side_hdf5(frame):
  from matplotlib import pyplot as plt
  import matplotlib.gridspec as gridspec
  gs1 = gridspec.GridSpec(10, 10)
  gs1.update(wspace=0.025, hspace=0.05)
  ax = []
  with h5py.File(frame, 'r') as hf:
    left_ims = np.array(hf.get('left'))
    right_ims = np.array(hf.get('right'))
    net.blobs['left'].data[...] = left_ims
    net.blobs['right'].data[...] = right_ims
    out = net.forward()
    label = out['probs'].argmax() - 1
    for v in xrange(10):
      ax.append(plt.subplot(gs1[v]))
      plt.imshow(left_ims[0, v, :, :], cmap='gray')
      if v >= label:
        ax.append(plt.subplot(gs1[v+10]))
        plt.imshow(right_ims[0, v - label, :, :], cmap='gray')
    for a in ax:
      a.set_xticklabels([])
      a.set_yticklabels([])
    plt.show()
