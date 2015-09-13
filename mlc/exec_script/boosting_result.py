import pickle
import os
import Layer
import numpy as np
import pandas as pd
import config
from sklearn.linear_model import LogisticRegression
import Layer
from sklearn.preprocessing import LabelEncoder
"""
single data
"""
train  = pd.read_csv('../data/train.csv')
test  = pd.read_csv('../data/test.csv')
test_ind = test.ID.astype(str)
encoder = LabelEncoder()
labels = train.target
labels = encoder.fit_transform(labels).astype(np.int32)


if not os.path.exists("submission"):
    os.mkdir("submission")

for train_file in os.listdir('train'):
    train_data = pickle.load(open("train/" + train_file))
    if train_file.endswith(".pkl"):
    	#check agreement correlation test
        preds = pickle.load(open("test/" + train_file))
        preds = pd.DataFrame({"Id": test_ind, "target": preds})
        preds.to_csv("submission/" + train_file + '.csv',index=False, sep=',')

train_list = [pickle.load(open("train/" + train_file)) for train_file in os.listdir('train') if train_file.endswith(".pkl")]
test_list = [pickle.load(open("test/" + test_file)) for test_file in os.listdir('test') if test_file.endswith(".pkl")]

train = np.array(train_list).T
test = np.array(test_list).T
labels = np.array(labels)
learning_model = Layer.ClassificationBinaryLayer()

parameter = {
	"model": "LasagneNeuralNetworkClassification",
	"layer_number":[100,100,100,100],
	"dropout_layer":[0.0,0.5,0.5,0.3]
}

if learning_model == "bagging_clf_binary":
	layer = Layer.ClassificationBinaryBaggingLayer()
elif learning_model == "bagging_clf_multi":
	layer = Layer.ClassificationBinaryBaggingLayer()
elif learning_model == "clf":
	layer = Layer.ClassificationBinaryLayer()
elif learning_model == "bagging_reg":
	layer = Layer.ClassificationBinaryBaggingLayer()
elif learning_model == "reg":
	layer = Layer.RegressionLayer()

preds = learning_model.predict_proba(train,labels, test,parameter=parameter)[:,1]
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