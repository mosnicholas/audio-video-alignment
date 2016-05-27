import argparse
import os
import random
import csv
import h5py
import numpy as np
from random import randint

parser = argparse.ArgumentParser(
  description='Generate data that makes a video of some increasing or decreasing brightness. The left or right is then offset by some number of frames. The offset is the label.')
parser.add_argument('--target_folder', default='/mnt/data/experiment_2',
  help='The parent directory for the dataset.')
parser.add_argument('--num_to_process', default=False,
  help='How many of the provided segments to process.')
parser.add_argument('--vary_offsets', default=False,
  help='If not specified, all offsets will be 6 frames.')
args = parser.parse_args()

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

import subprocess
def space_left():
  df = subprocess.Popen(["df", "/mnt"], stdout=subprocess.PIPE)
  output = df.communicate()[0]
  device, size, used, available, percent, mountpoint = \
    output.split("\n")[1].split()
  return int(size), int(available), int(used)

def main():

  num_segments_out = 50000

  if not os.path.isdir(args.target_folder):
    os.mkdir(args.target_folder)

  print("Generating varying offsets." if args.vary_offsets else "Generating offsets of 6 frames.")

  offsets = np.random.randint(1, 11, num_segments_out) if args.vary_offsets else 6.0 * np.ones(num_segments_out)
  offsets[::2] = 0

  test_inds = random.sample(range(num_segments_out), num_segments_out/40)
  in_train = np.array([True] * num_segments_out)
  in_train[test_inds] = False

  # Setup list of h5 filenames
  filenames_train = []
  filenames_test = []
  filenames_train_path = os.path.join(args.target_folder, 'train_h5_list.txt')
  filenames_test_path = os.path.join(args.target_folder, 'test_h5_list.txt')
  try:
    os.remove(filenames_train_path)
    os.remove(filenames_test_path)
  except OSError:
    pass

  full_file_path = ''

  seg1_ind = 0 # no offset
  seg2_ind = 1 # offset


  for frame_ind in xrange(num_segments_out):

    if frame_ind % 2 == 0:
      assert offsets[seg1_ind] == 0
      assert offsets[seg2_ind] > 0
      offset = offsets[seg2_ind]
      offset_left = np.random.randint(2) == 1

      asc = np.tile(np.linspace(0, 1, num=10 + offset).reshape((1, 1, offset + 10, 1, 1)), (1, 1, 1, 96, 128))

      h5_filename1 = 'seg-{:06d}.h5'.format(seg1_ind)
      h5_filename2 = 'seg-{:06d}.h5'.format(seg2_ind)
      h5_location1 = os.path.join(args.target_folder, h5_filename1)
      h5_location2 = os.path.join(args.target_folder, h5_filename2)

      #print(np.shape(curr_left_frames))
      #print(np.shape(curr_left_frames))
      #print(np.shape(curr_left_frames[:20:2,:,:]))
      #print(np.shape(stacked_left[0, :, :, :]))
      min_frame = np.random.randint(255)
      max_frame = np.random.randint(min_frame + 1, 256)
      range_frame = max_frame - min_frame

      stacked = (asc * range_frame) + min_frame
      stacked_left = np.empty((1, 1, 10, 96, 64))
      stacked_left[...] = stacked[0, 0, 0:10, :, 0:64]
      stacked_right = np.empty((1, 1, 10, 96, 64))
      stacked_right[...] = stacked[0, 0, 0:10, :, 64:]
      stacked_offset = np.empty((1, 1, 10, 96, 64))
      stacked_offset[...] = stacked[0, 0, offset:10+offset, :, 0:64] if offset_left else stacked[0, 0, offset:offset+10, :, 64:]

      stacked_left *= 1.0/255
      stacked_right *= 1.0/255
      stacked_offset *= 1.0/255

      with h5py.File(h5_location1, 'w') as f:
        f['left'] = stacked_left
        f['right'] = stacked_right
        label_mat = np.zeros((1, 1, 1, 1))
        label_mat[0, 0, 0, 0] = offsets[seg1_ind]
        f['label'] = label_mat
        label_mat_bin = np.zeros((1, 1, 1, 1))
        label_mat_bin[0, 0, 0, 0] = offsets[seg1_ind]
        f['label_bin'] = label_mat_bin
        #print("Writing to " + h5_location1)
        if in_train[seg1_ind]:
          filenames_train.append(h5_location1)
        else:
          filenames_test.append(h5_location1)

      with h5py.File(h5_location2, 'w') as f:
        f['left'] = stacked_offset if offset_left else stacked_left
        f['right'] = stacked_right if offset_left else stacked_offset
        label_mat = np.zeros((1, 1, 1, 1))
        label_mat[0, 0, 0, 0] = offsets[seg2_ind]
        f['label'] = label_mat
        label_mat_bin = np.zeros((1, 1, 1, 1))
        label_mat_bin[0, 0, 0, 0] = offsets[seg2_ind] > 0
        f['label_bin'] = label_mat_bin

      if in_train[seg2_ind]:
        filenames_train.append(h5_location2)
      else:
        filenames_test.append(h5_location2)

      if seg1_ind % 100 == 98 or frame_ind == num_segments_out - 2:
        print str(seg2_ind+1) + ' segments processed...'
        with open(filenames_train_path, 'a') as f:
          for filename_train in filenames_train:
            f.write(filename_train + '\n')
        with open(filenames_test_path, 'a') as f:
          for filename_test in filenames_test:
            f.write(filename_test + '\n')
        filenames_train = []
        filenames_test = []
      seg1_ind += 2
      seg2_ind += 2
  with h5py.File(os.path.join(args.target_folder, 'seg-000000.h5'), 'r') as f:
	  print 'Final datum written: '
	  print f['left']
	  print f['right']
	  print f['label']
  with open(filenames_train_path, 'r') as fnames_train:
    num_train_paths = len(fnames_train.readlines())
  with open(filenames_test_path, 'r') as fnames_test:
    num_test_paths = len(fnames_test.readlines())
  print("Recorded {:d} paths.".format(num_train_paths + num_test_paths))

main()
