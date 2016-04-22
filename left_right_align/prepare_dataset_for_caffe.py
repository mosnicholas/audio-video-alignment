import argparse
import os
import random
import csv
import caffe
import lmdb
import numpy as np
import multiprocessing
from scipy.ndimage import imread

parser = argparse.ArgumentParser(
  description='Generate the train and test LMDB for the left / right dataset. \
  This assumes all segments have ten left and ten right files. All ten frames \
  will be stacked in the channel dimension, then right will be stacked on \
  left likewise. Test gets 1000 segments and train gets the rest.')
parser.add_argument('--target_folder', default='./data/dataset_prepared',
  help='The parent directory for the dataset.')
parser.add_argument('--source_folder', default='./data/dataset/',
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
		for row in offsets_reader:
			offsets.append(row[1])
	return offsets

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

def process_ind(segment_ind, train_images_writer, train_labels_writer, test_images_writer, test_labels_writer, in_train, offsets):
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
			train_images_writer.put(filename_base, stacked_data.SerializeToString())
			train_labels_writer.put(filename_base, offsets[segment_ind])
		else:
			test_images_writer.put(filename_base, stacked_data.SerializeToString())
			test_labels_writer.put(filename_base, offsets[segment_ind])
		if segment_ind % 50 == 0:
			print str(segment_ind) + ' segments processed...'

def main():
	offsets = read_offsets('offsets.csv')
	num_segments = 1
	while os.path.isfile(os.path.join(args.source_folder, 'seg-{:06d}-frame-00-left.jpeg'.format(num_segments))):
		num_segments += 1

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

	train_images_lmdb = lmdb.open(os.path.join(args.target_folder, 'left_right_images_train'), map_size=int(1e12))
	train_labels_lmdb = lmdb.open(os.path.join(args.target_folder, 'left_right_labels_train'), map_size=int(1e12))
	test_images_lmdb = lmdb.open(os.path.join(args.target_folder, 'left_right_images_test'), map_size=int(1e12))
	test_labels_lmdb = lmdb.open(os.path.join(args.target_folder, 'left_right_labels_test'), map_size=int(1e12))

	with train_images_lmdb.begin(write=True) as train_images_writer, \
		train_labels_lmdb.begin(write=True) as train_labels_writer, \
		test_images_lmdb.begin(write=True) as test_images_writer, \
		test_labels_lmdb.begin(write=True) as test_labels_writer:

		if args.resume_from:
			start_index = args.resume_from
		else:
			start_index = 0

		parmap(lambda x: process_ind(x, train_images_writer, train_labels_writer, test_images_writer, test_labels_writer, in_train, offsets), range(start_index, num_segments))
	train_images_lmdb.close()
	train_labels_lmdb.close()
	test_images_lmdb.close()
	test_labels_lmdb.close()
	
main()