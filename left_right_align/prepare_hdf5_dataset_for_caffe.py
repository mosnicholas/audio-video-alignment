# specify filenames of parts one per line

import argparse
import os
import random
import csv
import caffe
import h5py
import numpy as np
import multiprocessing
from scipy.ndimage import imread

parser = argparse.ArgumentParser(
  description='Generate the train and test LMDB for the left / right dataset. \
  This assumes all segments have ten left and ten right files. All ten frames \
  will be stacked in the channel dimension, then right will be stacked on \
  left likewise. Test gets 1000 segments and train gets the rest.')
parser.add_argument('--target_folder', default='/mnt/data/dataset_prepared',
  help='The parent directory for the dataset.')
parser.add_argument('--source_folder', default='/mnt/data/dataset',
  help='The path to the source video')
parser.add_argument('--resume_from', default=False,
	help='The segment to resume processing from.')
args = parser.parse_args()

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def read_offsets(filename):
	offsets = []
	with open(os.path.join(args.source_folder, filename), 'rb') as offsets_file:
		offsets_reader = csv.reader(offsets_file, delimiter=',')
		for ind, row in enumerate(offsets_reader):
			if not ind == 0:
				offsets.append(int(row[1]))
	max_offset = max(offsets)
	return map(lambda offset: float(offset) / max_offset, offsets)

def fun(f,q_in,q_out):
    while True:
        i,x = q_in.get()
        if i is None:
            break
        q_out.put((i,f(x)))

def parmap(f, X, nprocs = multiprocessing.cpu_count()):
    q_in   = multiprocessing.Queue(1)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun,args=(f,q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]

def process_ind(segment_ind, train_images_txn, train_labels_txn, test_images_txn, test_labels_txn, in_train, offsets):
		filename_base = 'seg-{:06d}'.format(segment_ind + 1)
		path_filename_base = os.path.join(args.source_folder, filename_base)
		sample_frame = np.array(imread(path_filename_base + '-frame-{:02d}'.format(0) + '-right.jpeg'))
		stacked = np.zeros((np.size(sample_frame, 0), np.size(sample_frame, 1), 20))
		right_frames = []
		for frame_ind in range(0, 10):
			stacked[:, :, frame_ind] = imread(path_filename_base + '-frame-{:02d}'.format(frame_ind) + '-right.jpeg')
			stacked[:, :, 10 + frame_ind] = imread(path_filename_base + '-frame-{:02d}'.format(frame_ind) + '-left.jpeg')
		stacked_data = caffe.io.array_to_datum(stacked)
		if in_train[segment_ind]:
			train_images_txn.put(filename_base, stacked_data.SerializeToString())
			train_labels_txn.put(filename_base, offsets[segment_ind])
		else:
			test_images_txn.put(filename_base, stacked_data.SerializeToString())
			test_labels_txn.put(filename_base, offsets[segment_ind])
		if segment_ind % 50 == 0:
			print str(segment_ind) + ' segments processed...'

def process_inds(seg_start_ind, seg_end_ind, process_ind, in_train, offsets):

	sample_frame = np.array(imread(os.path.join(args.source_folder, 'seg-000001-frame-00-right.jpeg')))
	filenames_train = []
	filenames_test = []
	filenames_train_path = os.path.join(args.target_folder, 'examples_train.txt')
	filenames_test_path = os.path.join(args.target_folder, 'examples_test.txt')
	try:
		os.remove(filenames_train_path)
		os.remove(filenames_test_path)
	except OSError:
		pass

	for segment_ind in range(seg_start_ind, seg_end_ind):
		image_filename = 'seg-{:06d}-image.h5'.format(segment_ind + 1)
		#label_filename = 'seg-{:06d}-label.h5'.format(segment_ind + 1)
		path_filename_base = os.path.join(args.source_folder, image_filename)
		stacked_left = np.zeros((np.size(sample_frame, 0), np.size(sample_frame, 1), 10))
		stacked_right = np.zeros((np.size(sample_frame, 0), np.size(sample_frame, 1), 10))
		for frame_ind in range(0, 10):
			stacked_left[:, :, frame_ind] = imread(os.path.join(args.source_folder, 'seg-{:06d}-frame-{:02d}-right.jpeg'.format(segment_ind + 1, frame_ind)))
			stacked_right[:, :, frame_ind] = imread(os.path.join(args.source_folder, 'seg-{:06d}-frame-{:02d}-left.jpeg'.format(segment_ind + 1, frame_ind)))
		if in_train[segment_ind]:
			with h5py.File(image_filename, 'w') as f:
				f['left'] = np.transpose(stacked_left, (2, 1, 0))
				f['right'] = np.transpose(stacked_right, (2, 1, 0))
				f['label'] = offsets[segment_ind]
				#print np.shape(np.transpose(stacked_left, (2, 1, 0)))
			#with h5py.File(label_filename, 'w') as f:
			#	f['label'] = offsets[segment_ind]
			#filenames_train.append(image_filename)
		else:
			with h5py.File(image_filename, 'w') as f:
				f['left'] = np.transpose(stacked_left, (2, 1, 0))
				f['right'] = np.transpose(stacked_right, (2, 1, 0))
				f['label'] = offsets[segment_ind]
			#with h5py.File(label_filename, 'w') as f:
			#	f['label'] = offsets[segment_ind]
			#filenames_test.append(image_filename)
		if segment_ind % 100 == 0 or segment_ind == seg_end_ind - 1:
			print str(segment_ind) + ' segments processed...'
			with open(filenames_train_path, 'w') as f:
				for filename_train in filenames_train:
					f.write(filename_train + '\n')
			with open(filenames_test_path, 'w') as f:
				for filename_test in filenames_test:
					f.write(filename_test + '\n')
			filenames_train = []
			filenames_test = []



def main():
	offsets = read_offsets('offsets.csv')
	num_segments = 1
	while os.path.isfile(os.path.join(args.source_folder, 'seg-{:06d}-frame-00-left.jpeg'.format(num_segments))):
		num_segments += 1
	print 'Processing ' + str(num_segments) + ' segments.'

	if not os.path.isdir(args.target_folder):
		os.mkdir(args.target_folder)

	segment_inds = range(num_segments)

	test_inds_csv_path = os.path.join(args.target_folder, 'test_inds.csv')
	if (args.resume_from):
		with open(test_inds_csv_path, 'rb') as test_inds_csv:
			test_inds_reader = csv.reader(test_inds_csv, delimiter=',')
			test_inds = [row[0] for row in test_inds_reader]
	else:
		test_inds = random.sample(segment_inds, 1000)
		with open(test_inds_csv_path, 'w') as test_inds_csv:
			w = csv.writer(test_inds_csv)
			test_inds_to_write = [[ind] for ind in test_inds]
			w.writerows(test_inds_to_write)

	in_train = np.array([True] * num_segments)
	in_train[test_inds] = False

	if args.resume_from:
		offsets = offsets[resume_from:]

	process_inds(0, len(offsets), 0, in_train, offsets)

main()