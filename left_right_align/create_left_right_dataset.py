import os
import argparse
import csv
import h5py
import shutil
import subprocess
import time
import numpy as np
import moviepy
import multiprocessing
import random
from scipy import misc
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip, CompositeVideoClip, TextClip

parser = argparse.ArgumentParser(
  description='Download a source video and create a left / right alignment dataset.')
parser.add_argument('--target_folder', default='/mnt/data/blender',
  help='The parent directory for the dataset.')
parser.add_argument('--youtube_url', default='https://www.youtube.com/watch?v=eRsGyueVLvQ',
  help='If specified, the youtube url will be downloaded as source.')
parser.add_argument('--middle_gap_pixel_size', default=0,
  help='The size of the gap between the left and right images.')
parser.add_argument('--resume', default=False,
  help='The index to start counting at for output files.')
parser.add_argument('--frame_stride', default=False,
  help='Output sequences will effectively reduce the frame rate by this factor.')
parser.add_argument('--num_sequences', default=50000,
  help='How many sequences to output if space allows')
args = parser.parse_args()

youtube_urls = ["https://www.youtube.com/watch?v=aqz-KE-bpKQ&list=PL6B3937A5D230E335&index=9", \
"https://www.youtube.com/watch?v=SkVqJ1SGeL0&index=1&list=PL6B3937A5D230E335", \
"https://www.youtube.com/watch?v=lqiN98z6Dak&list=PL6B3937A5D230E335&index=2", \
"https://www.youtube.com/watch?v=Y-rmzh0PI3c&list=PL6B3937A5D230E335&index=3", \
"https://www.youtube.com/watch?v=Z4C82eyhwgU&list=PL6B3937A5D230E335&index=4", \
"https://www.youtube.com/watch?v=eRsGyueVLvQ&index=5&list=PL6B3937A5D230E335", \
"https://www.youtube.com/watch?v=R6MlUcmOul8&list=PL6B3937A5D230E335&index=6", \
"https://www.youtube.com/watch?v=TLkA0RELQ1g&index=8&list=PL6B3937A5D230E335"]

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
  create_examples()

def download_youtube_video(url, video_path, create_subtitles=False):
  try:
    os.remove(video_path)
  except OSError:
    pass
  command = 'youtube-dl -o %s %s' % (video_path, url) 
  if subprocess.check_call(command, shell=True) == 0:
    print("Download success.")
    time.sleep(15)
    return True
  print("Download failed.")
  return False

def write_save_state(curr_seg, curr_video): 
  save_state_path = os.path.join(args.target_folder, 'save_state.txt')
  with open(save_state_path, 'w') as save_state_file:
    save_state_file.write(str(curr_seg) + "," + str(curr_video))
   
def load_video(index):
  video_path = os.path.join(args.target_folder, 'video.mp4')
  result = download_youtube_video(youtube_urls[index], video_path)
  if result:
    print('Downloaded video: {:s}'.format(youtube_urls[index]))
    return VideoFileClip(video_path, audio=False)
 
def create_examples():

  stride = 2

  if (args.resume):
    curr_seg, curr_video = open(os.path.join(args.target_folder, 'save_state.txt'), 'r').read().split(",")
    curr_seg, curr_video = int(curr_seg), int(curr_video)
    print("Resuming from segment {:d}.".format(curr_seg))
  else:
    curr_seg = 0
    curr_video = 0
    write_save_state(0, 0)
    
  print("Loading video")

  video = load_video(curr_video)
  print("Resizing and trimming source video")
  start_frame = 0
  #print("Clipping to duration [{:d}, {:d}]".format(start_time, video.duration))
  video = video.resize((128, 96))
  print("Generating left and right source videos...")
  width = (np.size(video.get_frame(0), 1) - args.middle_gap_pixel_size) / stride
  num_frames = int(video.fps * video.duration)

  num_segments_out = int(args.num_sequences)

  offsets = np.random.randint(0, 11, num_segments_out)

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

  frame_ind = curr_seg * stride if args.resume else 0 # If you resume from seg 10, that is stored at 69, so start from 50
  
  curr_frames = np.empty((1, 1, 1, 96, 128))
  while curr_video < len(youtube_urls) and curr_seg < num_segments_out:

    curr_frame = np.empty((1, 1, 1, 96, 128)) 
    curr_frame_rgb = video.get_frame(float(frame_ind)/video.fps) 
    curr_frame[...] = rgb2gray(curr_frame_rgb)
    curr_frames = np.concatenate((curr_frames, curr_frame), axis=2)
    if np.shape(curr_frames)[2] > 20:
      curr_frames = curr_frames[:, :, 1:, :, :]

    if np.shape(curr_frames)[2] == 20 and frame_ind % stride == 0: # Make an example every 2 frames
      offset = offsets[curr_seg]
      offset_left = np.random.randint(2) == 1
      left_frame = np.empty((1, 1, 10, 96, width))
      left_frame[...] = curr_frames[0, 0, offset:10+offset, :, 0:width] if offset_left else curr_frames[0, 0, 0:10, :, 0:width]
      right_frame = np.empty((1, 1, 10, 96, width))
      right_frame[...] = curr_frames[0, 0, 0:10, :, width:] if offset_left else curr_frames[0, 0, offset:10+offset, :, width:]

      left_frame *= 1.0/255
      right_frame *= 1.0/255

      # Save h5 file
      h5_location = os.path.join(args.target_folder, 'seg-{:06d}.h5'.format(curr_seg))
      with h5py.File(h5_location, 'w') as f:
        f['left'] = left_frame
        f['right'] = right_frame
        label_mat = np.zeros((1, 1, 1, 1))
        label_mat[0, 0, 0, 0] = offsets[curr_seg]
        f['label'] = label_mat
        label_mat_bin = np.zeros((1, 1, 1, 1))
        label_mat_bin[0, 0, 0, 0] = offsets[curr_seg] > 0
        f['label_bin'] = label_mat_bin
        #print("Writing to " + h5_location1)
        if in_train[curr_seg]:
          filenames_train.append(h5_location)
        else:
          filenames_test.append(h5_location)

      # Save state
      curr_seg += 1
      write_save_state(curr_seg, curr_video)
    
      # Update filename list
      if curr_seg % 100 == 0 or curr_seg == num_segments_out:
        print str(curr_seg) + ' segments processed...'
        print("On frame {:d} of {:d}".format(frame_ind, num_frames))
        with open(filenames_train_path, 'a') as f:
          for filename_train in filenames_train:
            f.write(filename_train + '\n')
        with open(filenames_test_path, 'a') as f:
          for filename_test in filenames_test:
            f.write(filename_test + '\n')
        filenames_train = []
        filenames_test = []

    frame_ind += 1
    if frame_ind == num_frames:
      curr_video += 1
      video = load_video(curr_video)
      video = video.resize((128, 96))
      num_frames = int(video.duration * video.fps)
      frame_ind = 0
      curr_frames = np.empty((1, 1, 1, 96, 128))

  print("Created {:d} segments from {:d} sources videos.".format(curr_seg, curr_video))

main()
