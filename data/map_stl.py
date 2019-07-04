from scipy.io import loadmat, savemat
import numpy as np
import pdb
from skimage.transform import resize
from skimage.transform import downscale_local_mean
from skimage import io
import random





data = loadmat('train.mat')
trainx = data['X'].reshape(-1,3,96,96).transpose(0,3,2,1)
trainy = data['y'].reshape(-1)


test = loadmat('test.mat')
testx = test['X'].reshape(-1,3,96,96).transpose(0,3,2,1)
testy = test['y'].reshape(-1)

cls_mapping = np.array([0, 2, 1, 3, 4, 5, 6, -1, 7, 8])
trainy = cls_mapping[trainy-1]
testy = cls_mapping[testy-1]

# Remove all samples from skipped classes
train_mask = trainy != -1
test_mask = testy != -1

trainx = trainx[train_mask]
trainy = trainy[train_mask].astype('uint8')
testx = testx[test_mask]
testy = testy[test_mask].astype('uint8')

trainx = downscale_local_mean(trainx, (1, 3, 3, 1))
testx = downscale_local_mean(testx, (1, 3, 3, 1))

trainx = trainx.astype('uint8')
testx = testx.astype('uint8')


seed = random.randint(0, 100000)
np.random.seed(seed)
np.random.shuffle(trainx)
np.random.seed(seed)
np.random.shuffle(trainy)


savemat('stl_train.mat', {'X': trainx, 'y': trainy})
savemat('stl_test.mat', {'X': testx, 'y': testy})
