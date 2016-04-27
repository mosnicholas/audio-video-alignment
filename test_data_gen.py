
import lmdb
import numpy as np

for x in ('labels', 'frames'):
  for y in ('test', 'train'):
    for i in xrange(4):
      db = '%s_%s_%d' % (x, y, i)
      if lmdb.open(db).stat()['entries'] > 0:
        print db, lmdb.open(db).stat()['entries']

offsets = np.load('offsets.npz')['offsets']
match = True
for i in xrange(4):
  with lmdb.open('labels_test_%d' % i).begin() as txn:
    for k, v in txn.cursor():
      match = (match and int(v) == int(offsets[int(k)]))

print match

frames_correct = True
for i in xrange(4):
  with lmdb.open('frames_test_%d' % i).begin() as txn, \
        lmdb.open('labels_test_%d' % i).begin() as labels_txn:
    for x in xrange(i * 25, (i + 1) * 25):
      key = str(x)
      arr = np.fromstring(txn.get(key)).reshape(180, 240, 20)
      offset = int(labels_txn.get(key))
      frames_correct = frames_correct and np.all(arr[:, :, offset - 1:10] == arr[:, :, 10:10 + (10 - (offset - 1))])
      frames_correct = frames_correct and np.all(arr[:, :, 10 + (10 - (offset - 1)):] == 0)

print frames_correct

def view_side_by_side():
  from matplotlib import pyplot as plt
  ax = []
  with lmdb.open('frames_test_0').begin() as txn, \
        lmdb.open('labels_test_0').begin() as labels_txn:
    for u in xrange(25):
      offset = labels_txn.get(str(u))
      arr = np.fromstring(txn.get(str(u))).reshape(180, 240, 20)
      for v in xrange(10):
        ax.append(plt.subplot(2, 10, v + 1))
        plt.imshow(arr[:, :, v], cmap='gray')
      for y in xrange(10):
        ax.append(plt.subplot(2, 10, 10 + y + 1))
        plt.imshow(arr[:, :, 10 + y], cmap='gray')
      for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')
      print 'running for key: %d, with offset %s. It matches actual data: %r' % (u, offset, int(offset) == int(offsets[u]))
      plt.subplots_adjust(wspace=0, hspace=0)
      plt.show()

