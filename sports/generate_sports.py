import shutil
from moviepy.editor import VideoFileClip
import h5py
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Downloads the Sports 1M Dataset.")
parser.add_argument("--num_classes", default=False, help="Stop adding videos from new classes after seeing this many.")
args = parser.parse_args()

def download_url(url): 
  target_filename = os.path.join("/mnt/data/sports/", "curr_vid.mp4")
  command = 'youtube-dl "{:s}" -o {:s}'.format(url, target_filename)
  if subprocess.call(command, shell=True) == 0:
    return True
  return False

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_label_num(label, labels):
  if label in labels:
    return labels[label]
  else:
    if (not args.num_classes == False) and len(labels) > args.num_classes:
        return -1
    labels[label] = len(labels)
    return labels[label]

import subprocess 
def space_left():
  df = subprocess.Popen(["df", "/mnt"], stdout=subprocess.PIPE)
  output = df.communicate()[0]
  device, size, used, available, percent, mountpoint = \
    output.split("\n")[1].split() 
  return int(size), int(available), int(used)

def main():
  labels = {}
  
  segs_dld = download_videos("train_partition.txt", "/mnt/data/sports/train_files.txt", labels, True)
  segs_dld_test = download_videos("test_partition.txt", "/mnt/data/sports/test_files.txt", labels, False)
  print("Created {:d} examples and {:d} testing examples.".format(segs_dld, segs_dld_test))       
  print("Saw {:d} labels.".format(len(labels)))

def download_videos(url_list_path, h5_list_path, labels, train):
  print('Downloading videos for {:s} set.'.format("training" if train else "testing"))
  seg_ind = 0
  done = False
  num_vids = 0
  total_space, available, used = space_left()
  space_limit = .8*total_space if train else .9*total_space

  if os.path.isfile(h5_list_path):
    os.remove(h5_list_path)

  if os.path.isfile('/mnt/data/sports/curr_vid.mp4'):
    os.remove('/mnt/data/sports/curr_vid.mp4')

  with open(url_list_path, 'r') as tr_file: 
    for line in tr_file.readlines():
      url = line.split(' ')[0]
      labels_str = line.split(' ')[1]
      labels_str = labels_str.replace("\n","").split(',')
      label_nums = [get_label_num(label, labels) for label in labels_str]
      label_nums = filter(lambda x: x >= 0, label_nums)
      if download_url(url) and len(label_nums) > 0:
        num_vids += 1
        video = VideoFileClip("/mnt/data/sports/curr_vid.mp4", audio=False)
        print('Resizing source video...')
        video = video.resize((128, 96))
        print('Cropping source video...')
        video = video.crop(x1=32, width=64) # Resize to half frame to match left right video problem
        print('Saving frames from video...')
        num_frames = int(video.duration * video.fps)
        curr_frames = []
        starting_frames = np.random.choice(num_frames-10, 30, replace=False)
        for first_frame_ind in starting_frames:
          for frame_ind in range(first_frame_ind, first_frame_ind + 10):
            frame_t = float(frame_ind) / video.fps
            curr_frame = video.get_frame(frame_t)
            curr_frame = rgb2gray(curr_frame)
            curr_frames.append(curr_frame)
          stacked_frame = np.empty((1, 1, 10, 96, 64))
          stacked_frame[...] = curr_frames
          if np.max(stacked_frame) > 1:
            stacked_frame *= 1.0/255
          for label_num in label_nums:
            h5_filename = '/mnt/data/sports/h5/seg-{:06d}.h5'.format(seg_ind)
            with h5py.File(h5_filename, 'w') as hfile:
              hfile['sequence'] = stacked_frame
              label_arr = np.empty((1,1,1,1))
              label_arr[...] = label_num
              hfile['label'] = label_arr 
              seg_ind += 1
              if seg_ind % 100 == 0:
                print('Wrote {:d} segments'.format(seg_ind))
                total, available, used = space_left()
                if used > space_limit:
                  done = True
                  break
            with open(h5_list_path, 'a') as h5_list:
              h5_list.write(h5_filename + "\n")
          curr_frames = []
          if done:
            break
        os.remove("/mnt/data/sports/curr_vid.mp4")
        if done:
          break
      else:
        if len(label_nums) == 0:
          print("Only new labels in this video and limit has been reached. Skipping video")
        else:
          print("Failed to download video. Skipping {:s}".format(url))

  print("Downloaded {:d} source videos.".format(num_vids))
  return seg_ind

main()
