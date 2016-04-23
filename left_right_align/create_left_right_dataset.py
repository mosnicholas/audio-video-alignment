import os
import argparse
import csv
import shutil
import subprocess
import numpy as np
import moviepy
from scipy import misc
from random import randint, random as rand
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip

parser = argparse.ArgumentParser(
  description='Create a left / right alignment dataset from a local video file.')
parser.add_argument('--target_folder', default='./data/dataset',
  help='The parent directory for the dataset.')
parser.add_argument('--source_path', default='./data/source/hitch_hiker.mp4',
  help='The path to the source video')
parser.add_argument('--youtube_id', default=False,
  help='If specified, the youtube url will be downloaded as source.')
parser.add_argument('--youtube_target', default='./data/source/',
  help='Location to store the downloaded source video.')
parser.add_argument('--middle_gap_pixel_size', default=10,
  help='The size of the gap between the left and right images.')
parser.add_argument('--output_starting_ind', default=1,
  help='The index to start counting at for output files.')
parser.add_argument('--output_images', default=True,
  help='Output sequences of .jpeg files. If false, .mp4 videos will be generated.')
args = parser.parse_args()

args.youtube_id = 'XIeFKTbg3Aw'

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def download_youtube_video(youtube_id, target_path, create_subtitles=False):
  target_filename = os.path.join(target_path, 'hitch_hiker')
  command = 'youtube-dl %s -o %s' % (youtube_id, target_filename)
  if subprocess.call(command, shell=True) == 0:
    if create_subtitles: create_millisecond_subtitles(target_filename)
  else:
    return False
  return True

def main():
  if args.youtube_id:
    print 'Downloading video with youtube-dl...'
    download_youtube_video(args.youtube_id, args.youtube_target)
  else:
    split_video()

def split_video():

  movie_title = os.path.split(args.source_path)[-1]
  offset_csv = os.path.join(args.target_folder, 'offsets.csv')
  offsets = []
  video = VideoFileClip(args.source_path, audio=False)
  framerate = video.fps
  width = (np.size(video.get_frame(0), 1) - args.middle_gap_pixel_size) / 2
  left_video = moviepy.video.fx.all.crop(video, x1=0, width=width)
  right_video = moviepy.video.fx.all.crop(video, x1=width + args.middle_gap_pixel_size, width=width)
  right_frame_iterator = right_video.iter_frames()
  output_ind = args.output_starting_ind

  for ind, left_frame in enumerate(left_video.iter_frames()):
    left_frame = rgb2gray(left_frame)
    right_frame = rgb2gray(right_frame_iterator.next())
    if (ind % 20 == 0):
      left_frames = []
      right_frames = []
      offset_frames = []
      first_start = randint(0, 9) + ind
      offset = randint(1,10)
      second_start = first_start + offset
      offset_left = randint(0, 1) == 1
    if (ind >= first_start and ind < first_start + 10):
      right_frames.append(right_frame)
      left_frames.append(left_frame)
    if (ind >= second_start and ind < second_start + 10):
      if (offset_left):
        offset_frames.append(left_frame)
      else:
        offset_frames.append(right_frame)
    if (ind % 20 == 19):
      if args.output_images:
        for frame_ind, left_frame in enumerate(left_frames):
          misc.toimage(left_frame, cmin=np.min(left_frame), cmax=np.max(left_frame)).save(os.path.join(args.target_folder, ('seg-{:06d}-frame-{:02d}-left.jpeg').format(output_ind, frame_ind)))
        for frame_ind, right_frame in enumerate(right_frames):
          misc.toimage(right_frame, cmin=np.min(right_frame), cmax=np.max(right_frame)).save(os.path.join(args.target_folder, ('seg-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, frame_ind)))
      else:
        left_video_out = ImageSequenceClip(left_frames, fps=framerate)
        left_video_out.write_videofile(os.path.join(args.target_folder, 'seg-{:06d}-left.mp4' % output_ind), codec='libx264', audio=False)
        right_video_out = ImageSequenceClip(right_frames, fps=framerate)
        right_video_out.write_videofile(os.path.join(args.target_folder, 'seg-{:06d}-right.mp4' % output_ind), codec='libx264', audio=False)
      offsets.append({ 'id': '%06d' % output_ind, 'offset_frames': 0 })
      output_ind += 1
      if (offset_left):
        if args.output_images:
          for frame_ind, offset_frame in enumerate(offset_frames):
            misc.toimage(offset_frame, cmin=np.min(left_frame), cmax=np.max(left_frame)).save(os.path.join(args.target_folder, ('seg-{:06d}-frame-{:02d}-left.jpeg').format(output_ind, frame_ind)))
          for frame_ind, right_frame in enumerate(right_frames):
            misc.toimage(right_frame, cmin=np.min(right_frame), cmax=np.max(right_frame)).save(os.path.join(args.target_folder, ('seg-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, frame_ind)))
        else:
          left_video_out = ImageSequenceClip(offset_frames, fps=framerate)
          left_video_out.write_videofile(os.path.join(args.target_folder, 'seg-{:06d}-left.mp4' % output_ind), codec='libx264', audio=False)
          right_video_out = ImageSequenceClip(right_frames, fps=framerate)
          right_video_out.write_videofile(os.path.join(args.target_folder, 'seg-{:06d}-right.mp4' % output_ind), codec='libx264', audio=False)
      else:
        if args.output_images:
          for frame_ind, left_frame in enumerate(left_frames):
            misc.toimage(left_frame, cmin=np.min(left_frame), cmax=np.max(left_frame)).save(os.path.join(args.target_folder, ('seg-{:06d}-frame-{:02d}-left.jpeg').format(output_ind, frame_ind)))
          for frame_ind, offset_frame in enumerate(offset_frames):
            misc.toimage(offset_frame, cmin=np.min(right_frame), cmax=np.max(right_frame)).save(os.path.join(args.target_folder, ('seg-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, frame_ind)))
        else:
          left_video_out = ImageSequenceClip(left_frames, fps=framerate)
          left_video_out.write_videofile(os.path.join(args.target_folder, 'seg-{:06d}-left.mp4' % output_ind), codec='libx264', audio=False)
          right_video_out = ImageSequenceClip(offset_frames, fps=framerate)
          right_video_out.write_videofile(os.path.join(args.target_folder, 'seg-{:06d}-right.mp4' % output_ind), codec='libx264', audio=False)
      offsets.append({ 'id': '{:06d}'.format(output_ind), 'offset_frames': offset })
      output_ind += 1
    if (ind % 1000 == 0):
      print('Finished processing {:d} datapoints.'.format(output_ind))
  os.remove(offset_csv)
  with open(offset_csv, 'w') as offset_csv_file:
    w = csv.DictWriter(offset_csv_file, fieldnames=['id', 'offset_frames'])
    w.writeheader()
    w.writerows(offsets)
  return True

main()