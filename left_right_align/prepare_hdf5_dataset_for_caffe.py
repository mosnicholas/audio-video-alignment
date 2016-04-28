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
parser.add_argument('--num_to_process', default=False,
	help='How many of the provided segments to process.')
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

	full_file_path = ''

	for segment_ind in range(seg_start_ind, seg_end_ind):
		h5_filename = 'seg-{:06d}-image.h5'.format(segment_ind + 1)
		#label_filename = 'seg-{:06d}-label.h5'.format(segment_ind + 1)
		path_filename_base = os.path.join(args.source_folder, h5_filename)
		stacked_left = np.zeros((1, 10, np.size(sample_frame, 0), np.size(sample_frame, 1)))
		stacked_right = np.zeros((1, 10, np.size(sample_frame, 0), np.size(sample_frame, 1)))
		for frame_ind in range(0, 10):
			stacked_left[0, frame_ind, :, :] = imread(os.path.join(args.source_folder, 'seg-{:06d}-frame-{:02d}-right.jpeg'.format(segment_ind + 1, frame_ind)))
			stacked_right[0, frame_ind, :, :] = imread(os.path.join(args.source_folder, 'seg-{:06d}-frame-{:02d}-left.jpeg'.format(segment_ind + 1, frame_ind)))
		
		if stacked_left.max() > 1 and stacked_right > 1:
			stacked_left = .0039216 * stacked_left # Div by 255
			stacked_right = .0039216 * stacked_right
		full_file_path = os.path.join(args.target_folder, h5_filename)
		if in_train[segment_ind]:
			with h5py.File(full_file_path, 'w') as f:
				f['left'] = stacked_left
				f['right'] = stacked_right
				label_mat = np.zeros((1, 1, 1, 1))
				label_mat[0, 0, 0, 0] = offsets[segment_ind]
				f['label'] = label_mat
				#print np.shape(np.transpose(stacked_left, (2, 1, 0)))
			#with h5py.File(label_filename, 'w') as f:
			#	f['label'] = offsets[segment_ind]
			filenames_train.append(full_file_path)
		else:
			with h5py.File(full_file_path, 'w') as f:
				f['left'] = stacked_left
				f['right'] = stacked_right
				label_mat = np.zeros((1, 1, 1, 1))
				label_mat[0, 0] = offsets[segment_ind]
				f['label'] = label_mat
			#with h5py.File(label_filename, 'w') as f:
			#	f['label'] = offsets[segment_ind]
			filenames_test.append(full_file_path)
		if segment_ind % 100 == 0 or segment_ind == seg_end_ind - 1:
			print str(segment_ind) + ' segments processed...'
			with open(filenames_train_path, 'a') as f:
				for filename_train in filenames_train:
					f.write(filename_train + '\n')
			with open(filenames_test_path, 'a') as f:
				for filename_test in filenames_test:
					f.write(filename_test + '\n')
			filenames_train = []
			filenames_test = []
	with h5py.File(full_file_path, 'r') as f:
		print 'Final datum written: '
		print f['left']
		print f['right']
		print f['label']



def main():
	args.num_to_process = int(args.num_to_process)
	num_source_frames = 1
	while os.path.isfile(os.path.join(args.source_folder, 'frame-{:06d}-left.jpeg'.format(num_source_frames))):
		num_source_frames += 1
	print 'Counted ' + str(num_source_frames) + ' source segments.'
	num_source_frames = min(args.num_to_process, num_source_frames) if args.num_to_process else num_source_frames
	num_segments_out = num_source_frames - 50 # approx for safety

	if not os.path.isdir(args.target_folder):
		os.mkdir(args.target_folder)

	segment_inds = range(num_source_frames)

	test_inds_csv_path = os.path.join(args.target_folder, 'test_inds.csv')
	if (args.resume_from):
		with open(os.path.join(args.target_folder, 'offsets.csv'), 'rb') as offsets_csv:
			offsets_reader = csv.reader(offsets_csv, delimiter=',')
			offsets = [row[0] for row in offsets_reader]
			if args.num_to_process:
				offsets = offsets[:args.num_to_process]
		with open('test_inds_csv_path', 'rb') as test_inds_csv:
			test_inds_reader = csv.reader(test_inds_csv, delimiter=',')
			test_inds = [row[0] for row in test_inds_reader]
	else:
		test_inds = random.sample(segment_inds, num_source_frames/10)
		with open(test_inds_csv_path, 'w') as test_inds_csv:
			w = csv.writer(test_inds_csv)
			test_inds_to_write = [[ind] for ind in test_inds]
			w.writerows(test_inds_to_write)
		offsets = np.random.randint(1, 11, num_segments_out)
		offsets[::2] = 0
		np.savetxt(os.path.join(args.target_folder, 'offsets.csv'), offsets)

	in_train = np.array([True] * num_source_frames)
	in_train[test_inds] = False

	# Setup list of h5 filenames
	filenames_train = []
	filenames_test = []
	filenames_train_path = os.path.join(args.target_folder, 'examples_train.txt')
	filenames_test_path = os.path.join(args.target_folder, 'examples_test.txt')
	try:
		os.remove(filenames_train_path)
		os.remove(filenames_test_path)
	except OSError:
		pass

	sample_frame = np.array(imread(os.path.join(args.source_folder, 'seg-000001-frame-00-right.jpeg')))
	full_file_path = ''

	seg1_ind = 0 # no offset
	seg2_ind = 1 # offset

	curr_left_frames = np.zeros((29, 96, 64))
	curr_right_frames = np.zeros((29, 96, 64))

	for frame_ind in xrange(args.resume_from, num_segments_out):
		seg1_ind += 2
		seg2_ind += 2
		assert offsets[seg1_ind] == 0
		assert offsets[seg2_ind] > 0

		print np.shape(rgb2gray(imread(os.path.join(args.source_folder, 'frame-{:06d}-left.jpeg'.format(frame_ind)))))
		print np.shape(curr_left_frames)
		left_img = np.zeros((1, 96, 64))
		left_img[0, :, :] = rgb2gray(imread(os.path.join(args.source_folder, 'frame-{:06d}-left.jpeg'.format(frame_ind))))
		right_img = np.zeros((1, 96, 64))
		right_img[0, :, :] = rgb2gray(imread(os.path.join(args.source_folder, 'frame-{:06d}-right.jpeg'.format(frame_ind))))

		curr_left_frames = np.concatenate((curr_left_frames, left_img), 0)
		curr_right_frames = np.concatenate((curr_right_frames, right_img), 0)

		if np.size(curr_left_frames, 0) > 29:
			curr_left_frames = curr_left_frames[1,:,:]
			curr_right_frames = curr_right_frames[1,:,:]

		if frame_ind % 2 == 0:
			h5_filename1 = 'seg-{:06d}.h5'.format(seg1_ind)
			h5_filename2 = 'seg-{:06d}.h5'.format(seg2_ind)
			h5_location1 = os.path.join(args.source_folder, h5_filename1)
			h5_location2 = os.path.join(args.source_folder, h5_filename2)

			stacked_left = np.zeros((1, 10, 96, 64))
			np.shape(curr_left_frames)
			np.shape(curr_left_frames[:20:2,:,:])
			np.shape(stacked_left[0, :, :, :])
			stacked_left[0, :, :, :] = curr_left_frames[:20:2,:,:]
			stacked_right = np.zeros((1, 10, 96, 64))
			stacked_right[0, :, :, :] = curr_right_frames[:20:2,:,:]
			stacked_offset = np.zeros((1, 10, 96, 64))
			stacked_offset = curr_left_frames[offset:20+2*offset:2,:,:] if offset_left else curr_right_frames[offset:20+2*offset:2,:,:]
			if np.max(curr_left_frames) > 1 or np.max(curr_right_frames) > 1:
				stacked_left *= 1.0/255
				stacked_right *= 1.0/255
				stacked_offset *= 1.0/255

			with h5py.File(h5_location1, 'w') as f:
				f['left'] = stacked_left
				f['right'] = stacked_right
				label_mat = np.zeros((1, 1, 1, 1))
				label_mat[0, 0, 0, 0] = offsets[seg1_ind]
				f['label'] = label_mat

			if in_train[seg1_ind]:
				filenames_train.append(h5_location1)
			else:
				filenames_test.append(h5_location1)

			with h5py.File(h5_location2, 'w') as f:
				if offset_left:
					f['left'] = stacked_offset
					f['right'] = stacked_right
				else:
					f['left'] = stacked_left
					f['right'] = stacked_offset
				label_mat = np.zeros((1, 1, 1, 1))
				label_mat[0, 0, 0, 0] = offsets[seg2_ind]
				f['label'] = label_mat

			if in_train[seg2_ind]:
				filenames_train.append(h5_location2)
			else:
				filenames_test.append(h5_location2)

			if seg1_ind % 100 == 0 or seg2_ind % 100 == 0 or frame_ind == num_segments_out - 1:
				print str(seg2_ind) + ' segments processed...'
				with open(filenames_train_path, 'a') as f:
					for filename_train in filenames_train:
						f.write(filename_train + '\n')
				with open(filenames_test_path, 'a') as f:
					for filename_test in filenames_test:
						f.write(filename_test + '\n')
				filenames_train = []
				filenames_test = []
			with h5py.File(full_file_path, 'r') as f:
				print 'Final datum written: '
				print f['left']
				print f['right']
				print f['label']
main()
