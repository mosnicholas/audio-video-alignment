import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import h5py
import moviepy

# Make sure that caffe is on the python path:
caffe_root = '~/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '~/audio-video-alignment/left_right_align/siamese_caffenet_deploy.prototxt'
PRETRAINED = '/mnt/data/snapshots/scn_iter_20000.solverstate'
DATASET_PATH = '/mnt/data/dataset_prepared'
EXAMPLES_TXT = os.path.join(DATASET_PATH, 'examples_test.txt')

def main():
	with open(EXAMPLES_TXT, 'r') as examples_file:
		examples_paths = csv.reader(examples_file, delimiter=",")
		predLabels = []
		labels = []
		tenBest = [] # (absDiff, image1, image2, label, predLabel, path)
		tenWorst = []
		for ind, path in enumerate(examples_paths):
			with h5py.File(path, 'r') as hFile:
				leftFrames = hFile['left']
				rightFrames = hFile['right']
				label = hFile['label']
				predLabel = test_image(leftFrames, rightFrames)
				absDiff = abs(float(label) - pred(label))
				if len(tenBest) < 10 or absDiff < tenBest[-1][0]:
					tenBest.append((absDiff, leftFrames, rightFrames, label, predLabel, path))
					tenBest = sorted(tenBest, key=lambda x: x[0])
					if len(tenBest) > 10:
						tenBest = tenBest[:-1]
				if len(tenWorst) < 10 or absDiff > tenWorst[0][0]:
					tenWorst.append((absDiff, leftFrames, rightFrames, label, predLabel, path))
					tenWorst = sorted(tenWorst, key=lambdax: x[0])
					if len(tenWorst) > 10:
						tenWorst = tenWorst[1:]
				predLabels.append(predLabel)
				labels.append(label)
			if ind % 100 == 0: print "Tested " + str(ind)
	writeVideos(best, "success")
	for worst in tenWorst:
		writeVideo(worst, "failure")
	with open("results.csv", "w") as csvFile:
		w = csv.DictWriter(csvFile, fieldnames=["label", "prediction"])
		w.writeheader()
		w.writerows(np.hstack(np.transpose(np.array(labels)), np.transpose(np.array(predLabels))))
	predLabels = np.array(predLabels)
	threshold = np.median(predLabels)
	print 'Using threshold ' + threshold + ' to separate binary labels.'
	predLabelsBina = np.greater(predLabels, threshold)
	labelsBina = np.greater(labels, 0)
	print "Accuracy with threshold: {0:.2f}".format(100*float(np.sum(np.equal(predLabelsBina, labelsBina))) / labelsBina.size)


def writeVideo(vidTuples, prefix):
	for ind, vidTuple in enumerate(vidTuples):
		leftFrames = vidTuple[1]
		rightFrames = vidTuple[2]
		label = vidTuple[3]
		predLabel = vidTuple[4]
		path = vidTuple[5]
		print 'frames shape ' + str(np.shape(leftFrames))

		leftFramesList = [np.reshape(leftFrames[1, frame_ind, :, :], (96, 64)) for frame_ind in range(10)]
		rightFramesList = [np.reshape(rightFrames[1, frame_ind, :, :], (96, 64)) for frame_ind in range(10)]
		leftMovie = moviepy.ImageSequenceClip(leftFramesList);
		rightMovie = moviepy.ImageSequenceClip(rightFramesList);
		rightMovie.set_position((64, 0))
		labelClip = TextClip("truth: " + str(label) + "pred: " + str(predLabel), fontsize=70, color="white")
		labelClip.set_position("center")
		compositeMovie = moviepy.CompositeVideoClip([leftMovie, rightMovie, labelClip])
		pathParts = os.path.split(path)[-1]
		gifPath = prefix + '-{:02d}-'.format(ind) + gifPath[:-3] + '.gif'
		gifPath = os.path.join(pathParts, gifPath)
		compositeMovie.to_gif(gifPath, fps=10)
		print 'Saved to ' + gifPath


def test_image(left, right):
	caffe.set_mode_gpu()
	net = caffe.Classifier(MODEL_FILE, PRETRAINED,
	                       image_dims=(10, 96, 64))
	left = caffe.io.load_image(left)
	right = caffe.io.load_image(right)
	prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
	print 'prediction shape:', prediction[0].shape
	plt.plot(prediction[0])
	print 'predicted class:', prediction[0].argmax()
	plt.show()