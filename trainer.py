
import os
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip

DATA_FOLDER = 'data'
CLIPPED_VIDEO_FOLDER = os.path.join(DATA_FOLDER, 'clipped')
TRUMP_OUTPUT_CLIPPED_FOLDER = os.path.join(CLIPPED_VIDEO_FOLDER, 'trump')

def split_data():
  num_files = len(os.listdir(os.path.join(TRUMP_OUTPUT_CLIPPED_FOLDER, 'audio'))) - 1
  file_indices = np.arange(num_files)
  np.random.shuffle(file_indices)
  train_indices, test_indices = file_indices[:num_files * 4/5.0], file_indices[num_files * 4/5.0:]
  return train_indices, test_indices

def clip_audio(audio_clip):
  duration = audio_clip.duration
  cut = np.random.rand() * duration * 0.5
  cut_location = np.random.randint(5)
  if cut_location < 2: # cut from start
    start, end = cut, duration
  elif cut_location < 4: # cut from end
    start, end = 0, -cut
  else: # cut from both ends
    start, end = cut * 1/3.0, -cut * 1/3.0
  return audio_clip.subclip(start, end), start

def train(train_indices):
  audio_files = os.listdir(os.path.join(TRUMP_OUTPUT_CLIPPED_FOLDER, 'audio'))
  audio_files.remove('.gitkeep')
  video_files = os.listdir(os.path.join(TRUMP_OUTPUT_CLIPPED_FOLDER, 'video'))
  video_files.remove('.gitkeep')
  for sample_index in train_indices:
    audio_filepath = os.path.join(TRUMP_OUTPUT_CLIPPED_FOLDER, 'audio', audio_files[sample_index])
    video_filepath = os.path.join(TRUMP_OUTPUT_CLIPPED_FOLDER, 'video', video_files[sample_index])
    
    audio_sample = AudioFileClip(audio_filepath)
    video_sample = VideoFileClip(video_filepath)
    
    clipped_audio_sample, x_alignment = clip_audio(audio_sample)

def eval_l(test_indices):
  pass

def train_and_test():
  train_indices, test_indices = split_data()
  train(train_indices)
  eval_l(test_indices)


if __name__ == '__main__':
  train_and_test()
