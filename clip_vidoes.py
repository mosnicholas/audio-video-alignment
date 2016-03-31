
import os
import shutil
import subprocess
from moviepy.editor import VideoFileClip

VIDEO_YOUTUBE_ID = 'RDrfE9I8_hs'
DATA_FOLDER = 'data'
RAW_VIDEO_FOLDER = os.path.join(DATA_FOLDER, 'raw')
CLIPPED_VIDEO_FOLDER = os.path.join(DATA_FOLDER, 'clipped')

TRUMP_FILENAME = 'china.mp4'
TRUMP_OUTPUT_RAW_FILEPATH = os.path.join(RAW_VIDEO_FOLDER, 'trump', TRUMP_FILENAME)
TRUMP_OUTPUT_CLIPPED_FOLDER = os.path.join(CLIPPED_VIDEO_FOLDER, 'trump')

def create_millisecond_subtitles():
  with open(os.path.join(RAW_VIDEO_FOLDER, 'trump', 'china.srt'), 'w') as china:
    for i in xrange(1, 3 * 60 * 100):
      china.write(str(i))
      time = 1 + i * 10
      china.write("\n")
      seconds = (time / 1000) % 60
      minutes = (time / 60000) % 60
      currTime = "00:%02d:%02d,%03d" % (minutes, seconds, time % 1000)
      nextMillisecond = time % 1000 + 10
      if (nextMillisecond > 1000):
        nextMillisecond = 1
        seconds += 1
        minutes += seconds/60
        seconds = seconds % 60
      nextTime = "00:%02d:%02d,%03d" % (minutes, seconds, nextMillisecond)
      china.write("%s --> %s" %(currTime, nextTime))
      china.write("\n")
      china.write("time: " + currTime)
      china.write("\n")
      china.write("\n")
  return True

def download_trump_video():
  if not os.path.isfile(TRUMP_OUTPUT_RAW_FILEPATH):
    command = 'youtube-dl %s -o %s' % (VIDEO_YOUTUBE_ID, TRUMP_FILENAME)
    if subprocess.call(command, shell=True) == 0:
      create_millisecond_subtitles()
      shutil.move(TRUMP_FILENAME, TRUMP_OUTPUT_RAW_FILEPATH)
    else:
      return False
  return True

def clip_trump_videos():
  # To write video to file: clip.write_videofile(outfile, codec='libx264', audio_codec='aac', temp_audiofile='china-%02d.m4a' % i, remove_temp=True)
  # moviepy help: http://zulko.github.io/blog/2014/06/21/some-more-videogreping-with-python/ 
  #               https://zulko.github.io/moviepy/ref/ref.html
  cuts = [(1.55, 2.54), ()]
  video = VideoFileClip(TRUMP_OUTPUT_RAW_FILEPATH)
  subclips = [video.subclip(start, end) for (start, end) in cuts]
  for i in xrange(len(subclips)):
    clip = subclips[i]
    video_outfile = os.path.join(TRUMP_OUTPUT_CLIPPED_FOLDER, 'video', 'china-%02d.mp4' % i)
    audio_outfile = os.path.join(TRUMP_OUTPUT_CLIPPED_FOLDER, 'audio', 'china-%02d.m4a' % i)
    clip.write_videofile(video_outfile, codec='libx264', audio=False)
    clip.audio.write_audiofile(audio_outfile, codec='aac')
  return True

if __name__ == '__main__':
  create_millisecond_subtitles()

