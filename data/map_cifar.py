from scipy.io import loadmat, savemat
import numpy as np
import pdb
import random
from skimage import io



data1 = loadmat('data_batch_1.mat')
data2 = loadmat('data_batch_2.mat')
data3 = loadmat('data_batch_3.mat')
data4 = loadmat('data_batch_4.mat')
data5 = loadmat('data_batch_5.mat')
X1 = data1['data'].reshape(-1,3,32,32).transpose(0,2,3,1)
X2 = data2['data'].reshape(-1,3,32,32).transpose(0,2,3,1)
X3 = data3['data'].reshape(-1,3,32,32).transpose(0,2,3,1)
X4 = data4['data'].reshape(-1,3,32,32).transpose(0,2,3,1)
X5 = data5['data'].reshape(-1,3,32,32).transpose(0,2,3,1)
y1 = data1['labels']
y2 = data2['labels']
y3 = data3['labels']
y4 = data4['labels']
y5 = data5['labels']
trainx = np.concatenate((X1,X2,X3,X4,X5), axis=0)
trainy = np.concatenate((y1,y2,y3,y4,y5), axis=0).reshape(-1)

test = loadmat('test_batch.mat')
testx = test['data'].reshape(-1,3,32,32).transpose(0,2,3,1)
testy = test['labels'].reshape(-1)

cls_mapping = np.array([0, 1, 2, 3, 4, 5, -1, 6, 7, 8])
trainy = cls_mapping[trainy]
testy = cls_mapping[testy]

# Remove all samples from skipped classes
train_mask = trainy != -1
test_mask = testy != -1

pdb.set_trace()
trainx = trainx[train_mask].astype('uint8')
trainy = trainy[train_mask].astype('uint8')
testx = testx[test_mask].astype('uint8')
testy = testy[test_mask].astype('uint8')

seed = random.randint(0, 100000)
np.random.seed(seed)
np.random.shuffle(trainx)
np.random.seed(seed)
np.random.shuffle(trainy)

io.imsave('test.png', trainx[0,:])
print trainy[0]

savemat('cifar_train.mat', {'X': trainx, 'y': trainy})
savemat('cifar_test.mat', {'X': testx, 'y': testy})
