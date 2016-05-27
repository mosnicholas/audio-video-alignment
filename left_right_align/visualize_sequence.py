import scipy.misc
import numpy as np
import h5py

def main():
  for i in range(20):
    seg_int = np.random.randint(4000, 10000)
    seg_int = seg_int - (seg_int % 2) + 1
    with h5py.File('/mnt/data/blender/seg-{:06d}.h5'.format(seg_int)) as hfile:
      right = np.array(hfile.get('right')).squeeze()
      left = np.array(hfile.get('left')).squeeze()
      label = int(np.array(hfile.get('label')).squeeze())
      print("From example {:06d} with label {:d}".format(seg_int, label))
      width = np.size(left, 2)
      height = np.size(left, 1)
      border_size = 6
      result = np.empty((height, 0))   
      for ind in range(10):
        result = np.concatenate((result, left[ind, :, :], right[ind, :, :]),  axis=1)
        if ind < 9:
          result = np.concatenate((result, np.ones((height, border_size))), axis=1)
    scipy.misc.toimage(result, cmin=0.0, cmax=1.0).save('vis/offset_vis_{:06d}.jpeg'.format(i))   
      
main()
