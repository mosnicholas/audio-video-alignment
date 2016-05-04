import numpy as np
import caffe
import os
import argparse


parser = argparse.ArgumentParser(
  description='Generate the train and test LMDB for the left / right dataset. \
  This assumes all segments have ten left and ten right files. All ten frames \
  will be stacked in the channel dimension, then right will be stacked on \
  left likewise. Test gets 1000 segments and train gets the rest.')
parser.add_argument('--snapshot_name', default='',
  help='snapshot filename.')
args = parser.parse_args()

def test2_solver():
	#solver = caffe.SGDSolver('./siamese_caffenet_solver.prototxt')
	#solver.restore(os.path.join('/mnt/data/snapshots', args.snapshot_name))
	net = caffe.Net('./siamese_caffenet_deploy.prototxt', os.path.join('/mnt/data/snapshots', args.snapshot_name), caffe.TEST)

	asc = np.tile(np.linspace(0, 1, num=10).reshape((1, 1, 10, 1, 1)), (1, 1, 1, 96, 64))

	for _ in range(20):
		min_frame1 = np.random.randint(255)
		max_frame1 = np.random.randint(min_frame1 + 1, 256)
		range1 = max_frame1 - min_frame1
		min_frame2 = np.random.randint(255)
		max_frame2 = np.random.randint(min_frame1 + 1, 256)
		range2 = max_frame2 - min_frame2
		if range1 > range2:
			greater = (min_frame1, range1)
			lesser = (min_frame2, range2)
		else:
			greater = (min_frame2, range2)
			lesser = (min_frame1, range1)

		stacked_left1 = asc * greater[1] + greater[0]
		stacked_right1 = asc * lesser[1] + lesser[0]
		print("Left is {:.3f} to {:.3f}. Right is {:.3f} to {:.3f}." \
			.format(greater[0], greater[0] + greater[1], lesser[0], lesser[0] + lesser[1]))

		stacked_left2 = asc * lesser[1] + lesser[0]
		stacked_right2 = asc * greater[1] + greater[0]
		print("Left is {:.3f} to {:.3f}. Right is {:.3f} to {:.3f}." \
			.format(lesser[0], lesser[0] + lesser[1], greater[0], greater[0] + greater[1]))

		stacked_left1 *= 1.0/255
		stacked_right1 *= 1.0/255
		stacked_left2 *= 1.0/255
		stacked_right2 *= 1.0/255

		net.blobs['left'].data[...] = stacked_left1
		#print(stacked_left1)
		net.blobs['right'].data[...] = stacked_right1
		#print(stacked_right1)
		net.forward()
		result = net.blobs['out'].data
		print("Expected {:d}, got {:s}.".format(0, result))

		net.blobs['left'].data[...] = stacked_left2
		#print(stacked_left1)
		net.blobs['right'].data[...] = stacked_right2
		#print(stacked_right1)
		net.forward()
		result = net.blobs['out'].data
		print("Expected {:d}, got {:s}.".format(1, result))

test2_solver()