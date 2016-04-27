from random import randint
import time
import os
from moviepy.editor import VideoFileClip

DATA_FOLDER = 'data'
RAW_VIDEO_FOLDER = os.path.join(DATA_FOLDER, 'raw')
movie_path = os.path.join(RAW_VIDEO_FOLDER, 'hitch_hiker.mp4')
video = VideoFileClip(movie_path)
fps = video.fps
first_frame = int(23 * fps) + 1
num_trials = 2000

def v1():
  t1 = time.time()
  for i in xrange(first_frame, first_frame+num_trials):
    index_a = randint(4, 8)
    index_b = randint(4, index_a)
    v1 = video.subclip(i/fps, (i + index_a + 2)/fps)
    v2 = video.subclip((i + index_b)/fps, (i + 10)/fps)
  t2 = time.time()
  return float(t2 - t1)/num_trials

def v2():
  t1 = time.time()
  for i in xrange(first_frame, first_frame+num_trials):
    frames = [video.get_frame(j/fps) for j in xrange(i, i+10)]
    index_a = randint(4, 8)
    index_b = randint(4, index_a)
    f1 = frames[:index_a + 1]
    f2 = frames[index_b:]
  t2 = time.time()
  return float(t2 - t1)/num_trials


