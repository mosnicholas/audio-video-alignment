import os
import argparse
import csv
import shutil
import subprocess
import numpy as np
import moviepy
import multiprocessing
from scipy import misc
from random import randint, random as rand
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip, CompositeVideoClip, TextClip

parser = argparse.ArgumentParser(
  description='Create a left / right alignment dataset from a local video file.')
parser.add_argument('--target_folder', default='/mnt/data/dataset',
  help='The parent directory for the dataset.')
parser.add_argument('--source_path', default='/mnt/data/source/hitch_hiker.mp4',
  help='The path to the source video')
parser.add_argument('--youtube_id', default=False,
  help='If specified, the youtube url will be downloaded as source.')
parser.add_argument('--youtube_target', default='./data/source/',
  help='Location to store the downloaded source video.')
parser.add_argument('--middle_gap_pixel_size', default=0,
  help='The size of the gap between the left and right images.')
parser.add_argument('--output_starting_ind', default=1,
  help='The index to start counting at for output files.')
parser.add_argument('--output_images', default=True,
  help='Output sequences of .jpeg files. If false, .mp4 videos will be generated.')
parser.add_argument('--presentation_movie', default=False,
  help='Output sequences of .jpeg files. If false, .mp4 videos will be generated.')
parser.add_argument('--frame_stride', default=False,
  help='Output sequences will effectively reduce the frame rate by this factor.')
args = parser.parse_args()

#args.youtube_id = 'XIeFKTbg3Aw'

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def download_youtube_video(youtube_id, target_path, create_subtitles=False):
  target_filename = os.path.join(target_path, 'hitch_hiker')
  command = 'youtube-dl %s -o %s' % (youtube_id, target_filename)
  if subprocess.call(command, shell=True) == 0:
    return True
  return False

def main():
  if args.youtube_id:
    print 'Downloading video with youtube-dl...'
    download_youtube_video(args.youtube_id, args.youtube_target)
  if args.presentation_movie:
    split_video_pres()
  elif args.frame_stride:
    split_video_stride()
  else:
    split_video()

def split_video_pres():

  movie_title = os.path.split(args.source_path)[-1]
  video = VideoFileClip(args.source_path, audio=False)
  video = video.subclip((1, 8, 36), (1, 8, 41))
  video = video.resize((128, 96))
  framerate = video.fps
  width = (np.size(video.get_frame(0), 1) - args.middle_gap_pixel_size) / 2
  left_video = moviepy.video.fx.all.crop(video, x1=0, width=width)
  right_video = moviepy.video.fx.all.crop(video, x1=width + args.middle_gap_pixel_size, width=width)
  output_ind = args.output_starting_ind
  offsets_filename = "pres-offsets.txt"
  offset_csv = os.path.join(args.target_folder, offsets_filename)
  file_prefix = "pres"

  pres_offset = 10

  all_left_frames = []
  all_right_frames = []

  left_frame_iterator = left_video.iter_frames()
  for ind, right_frame in enumerate(right_video.iter_frames()):
    if ind >= pres_offset:
      left_frame = rgb2gray(left_frame_iterator.next())
      right_frame = rgb2gray(right_frame)
      if (ind % 10 == 0): # INITIALIZE
        left_frames = []
        right_frames = []
      all_right_frames.append(np.transpose(np.tile(right_frame, (3, 1, 1))))
      right_frames.append(right_frame)
      all_left_frames.append(np.transpose(np.tile(left_frame, (3, 1, 1))))
      left_frames.append(left_frame)
      if (ind % 10 == 9): # SAVE SEGMENT FRAMES TO JPEG
        for frame_ind, left_frame in enumerate(left_frames):
          misc.toimage(left_frame, cmin=np.min(left_frame), cmax=np.max(left_frame)).save(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-left.jpeg').format(output_ind, frame_ind)))
        for frame_ind, right_frame in enumerate(right_frames):
          misc.toimage(right_frame, cmin=np.min(right_frame), cmax=np.max(right_frame)).save(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, frame_ind)))
        output_ind += 1
      if (ind % 1000 == 0):
        print('Finished processing {:d} datapoints.'.format(output_ind))
  with open(offset_csv, 'w') as offset_csv_file:
    offset_csv_file.write(str(pres_offset) + "\n")
  leftMovie = ImageSequenceClip(all_left_frames, fps=framerate);
  rightMovie = ImageSequenceClip(all_right_frames, fps=framerate);
  rightMovie.set_pos((64, 0))
  #labelClip = TextClip("offset: " + str(pres_offset))
  #labelClip.set_position("center")
  compositeMovie = CompositeVideoClip([leftMovie, rightMovie], size=(128, 96))
  compositeMovie.write_videofile(os.path.join(args.target_folder, "pres_video.mp4"), codec='libx264', audio=False)
  return True

def split_video_stride():
  movie_title = os.path.split(args.source_path)[-1]
  video = VideoFileClip(args.source_path, audio=False)
  if (args.presentation_movie):
    video = video.subclip((1, 8, 36), (1, 8, 41))
  video = video.set_fps(1).subclip(1, 20000).resize((128, 96))
  if (args.presentation_movie):
    video.write_videofile(os.path.join(args.target_folder, 'pres_video.mp4'), codec='libx264', audio=False)
  framerate = video.fps
  width = (np.size(video.get_frame(0), 1) - args.middle_gap_pixel_size) / 2
  num_frames = int(video.fps * video.duration)
  left_video = moviepy.video.fx.all.crop(video, x1=0, width=width)
  right_video = moviepy.video.fx.all.crop(video, x1=width + args.middle_gap_pixel_size, width=width)
  left_video = left_video.set_fps(1).set_duration(num_frames)
  right_video = right_video.set_fps(1).set_duration(num_frames)

  output_ind = args.output_starting_ind
  file_prefix = "seg"
  offsets = [0]*20000

  print "Using stride 2..."
  stride = 3
  poss_left_frames = []
  poss_right_frames = []

  num_cpus = 1 #multiprocessing.cpu_coun
  process_inds_stride(left_video, right_video, 0, 20000, offsets)
  frames_per_process = 20000/num_cpus
  '''for i in xrange(num_cpus):
    start_ind = i * frames_per_process
    end_ind = 20000 if i == num_cpus - 1 else (i + 1)*frames_per_process
    multiprocessing.Process(
      target=process_inds_stride,
      args=(left_video, right_video, start_ind, end_ind, offsets)
    ).start()'''

offsets_recorded = 0
def record_offsets(start_ind, end_ind, offsets, offsets_to_add):
  offsets[start_ind:end_ind] = offsets_to_add
  global offsets_recorded
  offsets_recorded += 1
  print str(offsets_recorded) + " offsets recorded of " + str(multiprocessing.cpu_count())
  if offsets_recorded == multiprocessing.cpu_count():
    with open(os.path.join(args.target_folder, 'offsets.csv'), 'w') as offset_csv_file:
      w = csv.DictWriter(offset_csv_file, fieldnames=['id', 'offset_frames'])
      w.writeheader()
      w.writerows(offsets)

def process_inds_stride(left_video, right_video, start_ind, end_ind, full_offsets):
  file_prefix = 'seg'
  offsets = []
  output_ind = start_ind

  for ind in range(start_ind, end_ind)[::2]:

    offset = randint(1,10)
    offset_left = randint(0, 1) == 1 # coin flip

    frames_out = range(ind, ind + 19)[::2]
    offset_frames_out = range(ind + offset, ind + offset + 19)[::2]
    #left_frames_out = poss_left_frames[0:28:3]
    #offset_frames_out = poss_left_frames[offset:28+offset:3] if offset_left else poss_right_frames[offset:28+offset:3]

    output_ind += 1

    for arr_ind, frame_ind in enumerate(frames_out):
      #misc.toimage(right_frame, cmin=np.min(right_frame), cmax=np.max(right_frame)).save(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, frame_ind)))
      right_video.save_frame(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, arr_ind)), frame_ind)
      left_video.save_frame(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-left.jpeg').format(output_ind, arr_ind)), frame_ind)

    offsets.append({ 'id': '%06d' % output_ind, 'offset_frames': 0 })
    output_ind += 1

    if (offset_left):
      for arr_ind, frame_ind in enumerate(frames_out):
        right_video.save_frame(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, arr_ind)), frame_ind)
      for arr_ind, frame_ind in enumerate(offset_frames_out):
        left_video.save_frame(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-left.jpeg').format(output_ind, arr_ind)), frame_ind)
    else:
      for arr_ind, frame_ind in enumerate(offset_frames_out):
        right_video.save_frame(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, arr_ind)), frame_ind)
      for arr_ind, frame_ind in enumerate(frames_out):
        left_video.save_frame(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-left.jpeg').format(output_ind, arr_ind)), frame_ind)
    output_ind += 1
    offsets.append({ 'id': '{:06d}'.format(output_ind), 'offset_frames': offset })
    if (ind % 10 == 0):
      #print('Finished processing {:d} outputs.'.format(output_ind-1))
      print('At video frame ' + str(ind) + ' of ' + str(20000))
  record_offsets(start_ind, end_ind, full_offsets, offsets)
  with open(os.path.join(args.target_folder, 'offsets.csv'), 'w') as offset_csv_file:
    w = csv.DictWriter(offset_csv_file, fieldnames=['id', 'offset_frames'])
    w.writeheader()
    w.writerows(offsets)
  return True

def split_video():

  movie_title = os.path.split(args.source_path)[-1]
  video = VideoFileClip(args.source_path, audio=False)
  if (args.presentation_movie):
    video = video.subclip((1, 8, 36), (1, 8, 41))
  video = video.resize((128, 96))
  if (args.presentation_movie):
    video.write_videofile(os.path.join(args.target_folder, 'pres_video.mp4'), codec='libx264', audio=False)
  framerate = video.fps
  width = (np.size(video.get_frame(0), 1) - args.middle_gap_pixel_size) / 2
  left_video = moviepy.video.fx.all.crop(video, x1=0, width=width)
  right_video = moviepy.video.fx.all.crop(video, x1=width + args.middle_gap_pixel_size, width=width)
  right_frame_iterator = right_video.iter_frames()
  output_ind = args.output_starting_ind
  file_prefix = "seg"
  offsets_filename = "offsets.csv" 
  offset_csv = os.path.join(args.target_folder, offsets_filename)
  offsets = []

  for ind, left_frame in enumerate(left_video.iter_frames()):
    if ind > 800:
      break
    left_frame = rgb2gray(left_frame)
    right_frame = rgb2gray(right_frame_iterator.next())
    if (ind % 20 == 0): # INITIALIZE
      left_frames = []
      right_frames = []
      offset_frames = []
      first_start = ind
      offset = randint(1,10)
      second_start = first_start + offset
      offset_left = randint(0, 1) == 1
    if (ind >= first_start and ind < first_start + 10): # ADD FRAMES
      right_frames.append(right_frame)
      left_frames.append(left_frame)
    if (ind >= second_start and ind < second_start + 10): # ADD OFFSET FRAMES
      if (offset_left):
        offset_frames.append(left_frame)
      else:
        offset_frames.append(right_frame)
    if (ind % 20 == 19): # SAVE SEGMENT FRAMES TO JPEG
      if args.output_images:
        assert len(left_frames) == 10, 'Only added ' + str(len(left_frames)) + ' left frames on segment ' + str(output_ind) + '. Should have 10.'
        assert len(right_frames) == 10, 'Only added ' + str(len(right_frames)) + ' right frames on segment ' + str(output_ind) + '. Should have 10.'
        assert len(offset_frames) == 10, 'Only added ' + str(len(offset_frames)) + ' offset frames on segment ' + str(output_ind) + '. Should have 10.'
        for frame_ind, left_frame in enumerate(left_frames):
          misc.toimage(left_frame, cmin=np.min(left_frame), cmax=np.max(left_frame)).save(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-left.jpeg').format(output_ind, frame_ind)))
        for frame_ind, right_frame in enumerate(right_frames):
          misc.toimage(right_frame, cmin=np.min(right_frame), cmax=np.max(right_frame)).save(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, frame_ind)))
      else:
        left_video_out = ImageSequenceClip(left_frames, fps=framerate)
        left_video_out.write_videofile(os.path.join(args.target_folder, file_prefix + '-{:06d}-left.mp4' % output_ind), codec='libx264', audio=False)
        right_video_out = ImageSequenceClip(right_frames, fps=framerate)
        right_video_out.write_videofile(os.path.join(args.target_folder, file_prefix + '-{:06d}-right.mp4' % output_ind), codec='libx264', audio=False)
      offsets.append({ 'id': '%06d' % output_ind, 'offset_frames': 0 })
      output_ind += 1
      if (offset_left):
        if args.output_images:
          for frame_ind, offset_frame in enumerate(offset_frames):
            misc.toimage(offset_frame, cmin=np.min(offset_frame), cmax=np.max(offset_frame)).save(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-left.jpeg').format(output_ind, frame_ind)))
          for frame_ind, right_frame in enumerate(right_frames):
            misc.toimage(right_frame, cmin=np.min(right_frame), cmax=np.max(right_frame)).save(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, frame_ind)))
        else:
          left_video_out = ImageSequenceClip(offset_frames, fps=framerate)
          left_video_out.write_videofile(os.path.join(args.target_folder, file_prefix + '-{:06d}-left.mp4' % output_ind), codec='libx264', audio=False)
          right_video_out = ImageSequenceClip(right_frames, fps=framerate)
          right_video_out.write_videofile(os.path.join(args.target_folder, file_prefix + '-{:06d}-right.mp4' % output_ind), codec='libx264', audio=False)
      else:
        if args.output_images:
          for frame_ind, left_frame in enumerate(left_frames):
            misc.toimage(left_frame, cmin=np.min(left_frame), cmax=np.max(left_frame)).save(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-left.jpeg').format(output_ind, frame_ind)))
          for frame_ind, offset_frame in enumerate(offset_frames):
            misc.toimage(offset_frame, cmin=np.min(offset_frame), cmax=np.max(offset_frame)).save(os.path.join(args.target_folder, (file_prefix + '-{:06d}-frame-{:02d}-right.jpeg').format(output_ind, frame_ind)))
        else:
          left_video_out = ImageSequenceClip(left_frames, fps=framerate)
          left_video_out.write_videofile(os.path.join(args.target_folder, file_prefix + '-{:06d}-left.mp4' % output_ind), codec='libx264', audio=False)
          right_video_out = ImageSequenceClip(offset_frames, fps=framerate)
          right_video_out.write_videofile(os.path.join(args.target_folder, file_prefix + '-{:06d}-right.mp4' % output_ind), codec='libx264', audio=False)
      offsets.append({ 'id': '{:06d}'.format(output_ind), 'offset_frames': offset })
      output_ind += 1
    if (ind % 100 == 0):
      print('Finished processing {:d} datapoints.'.format(output_ind))
  with open(offset_csv, 'w') as offset_csv_file:
    w = csv.DictWriter(offset_csv_file, fieldnames=['id', 'offset_frames'])
    w.writeheader()
    w.writerows(offsets)
  return True

main()