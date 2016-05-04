import numpy as np
import caffe
import os


parser = argparse.ArgumentParser(
  description='Generate the train and test LMDB for the left / right dataset. \
  This assumes all segments have ten left and ten right files. All ten frames \
  will be stacked in the channel dimension, then right will be stacked on \
  left likewise. Test gets 1000 segments and train gets the rest.')
parser.add_argument('--snapshot_name', default='',
  help='snapshot filename.')
args = parser.parse_args()

def test2_solver():
	solver = caffe.SGDSolver('./train_siamese_caffenet.prototxt')
	solver.restore(os.path.join('/mnt/data/snapshots', args.snapshot_name))

	for _ in range(1):
		min_frame1 = np.random.randint(255)
		max_frame1 = np.random.randint(min_frame1 + 1, 256)
		range1 = max_frame1 - min_frame1
		min_frame2 = np.random.randint(255)
		max_frame2 = np.random.randint(min_frame1 + 1, 256)
		range2 = max_frame2 - min_frame2
		if range1 > range2
			greater = (min_frame1, range1)
			lesser = (min_frame2, range2)
		else:
			greater = (min_frame2, range2)
			lesser = (min_frame1, range1)

		stacked_left1 = asc * greater[1] + greater[0]
		stacked_right1 = asc * lesser[1] + lesser[0]

		stacked_left2 = asc * lesser[1] + lesser[0]
		stacked_right2 = asc * greater[1] + greater[0]

		stacked_left1 *= 1.0/255
		stacked_right1 *= 1.0/255
		stacked_left2 *= 1.0/255
		stacked_right2 *= 1.0/255

		solver.net.blobs['left'].data[...] = stacked_left1
		print(stacked_left1)
		solver.net.blobs['right'].data[...] = stacked_right1
		print(stacked_right1)
		result = solver.net.forward()
		print("Expected {:d}, got {:d}.".format(0, result))

		solver.net.blobs['left'] = stacked_left2
		print(stacked_left1)
		solver.net.blobs['right'] = stacked_right2
		print(stacked_right1)
		result = solver.net.forward()
		print("Expected {:d}, got {:d}.".format(1, result))