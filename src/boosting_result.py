import pickle
import os
import Layer
import numpy as np
import pandas as pd
import config
from sklearn.linear_model import LogisticRegression
"""
single data
"""
train  = pd.read_csv('../data/training.csv', index_col=0)
test  = pd.read_csv('../data/test.csv', index_col=0)
test_ind = test.index
labels = train["signal"]

if not os.path.exists("submission"):
    os.mkdir("submission")

for train_file in os.listdir('train'):
    train_data = pickle.load(open("train/" + train_file))
    if train_file.endswith(".pkl"):
    	#check agreement correlation test
        preds = pickle.load(open("test/" + train_file))
        preds = pd.DataFrame({"Id": test_ind, "prediction": preds})
        preds.to_csv("submission/" + train_file + '.csv',index=False, sep=',')

train_list = [pickle.load(open("train/" + train_file)) for train_file in os.listdir('train') if train_file.endswith(".pkl")]
test_list = [pickle.load(open("test/" + test_file)) for test_file in os.listdir('test') if test_file.endswith(".pkl")]

for train_file in os.listdir('train'):
	if train_file.endswith(".pkl"):
		print pickle.load(open("train/" + train_file))
print np.array(labels).astype(np.float32).shape

model = LogisticRegression()
model.fit(np.array(train_list).T,np.array(labels).astype(np.float32))
preds = model.predict_proba(np.array(test_list).T)[:,1]
# train_list = [pickle.load(open(config.BLENDING_TRAIN_DIR + "/" + train_file)) for train_file in config.BLENDING_FILES]
# test_list = [pickle.load(open(config.BLENDING_TEST_DIR + "/" + test_file)) for test_file in config.BLENDING_FILES]
# #load train and test
# train  = pd.read_csv('../data/train_set.csv', index_col=0)
# test  = pd.read_csv('../data/test_set.csv', index_col=0)

# labels = train.cost.values

# test_ind = test.index

# layer = Layer.RegressionLayer()
# train_result_array = np.log1p(np.array(train_list))
# label_log = np.log1p(labels)

# preds = np.zeros(np.array(test_list)[0].shape)
# result , weights = Layer.blending_weight_optimize(np.array(train_list),labels,"slsqp")
# for index in xrange(len(weights)):
#     preds = preds + weights[index] * np.array(test_list)[index]
# # _,preds = layer.predict(np.array(train_list).T, labels,np.array(test_list).T,parameter=parameter)

preds = pd.DataFrame({"Id": test_ind, "prediction": preds})
preds = preds.set_index('Id')
preds.to_csv('stacking_submission.csv')