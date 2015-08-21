#coding:utf-8
import pickle
import os
import pandas as pd
import numpy as np
import evaluation_functions
import Layer
#load train and test
print "single model approach"

train  = pd.read_csv('../data/train_set.csv', index_col=0)
test  = pd.read_csv('../data/test_set.csv', index_col=0)
test_ind = test.index
labels = train.cost.values

for train_file in os.listdir('train'):
	train_data = pickle.load(open("train/" + train_file))
	if train_file.endswith(".pkl"):
		evaluate_value = evaluation_functions.evaluate_function(labels,train_data,'rmsle')
        print train_file,evaluate_value

#組み合わせた場合どうなるかを検証する
print "start Combination Modeling"
train_list = [pickle.load(open("train/" + train_file)) for train_file in os.listdir('train') if train_file.endswith(".pkl")]
test_list = [pickle.load(open("test/" + test_file)) for test_file in os.listdir('test') if test_file.endswith(".pkl")]

train_array = np.array(train_list).T
test_array = np.array(test_list).T
layer = Layer.Layer()

print train_array.shape,test_array.shape

parameter = {
    "alpha": 4.8e-06, 
    "eta": 0.02,
    "model": "XGBREGLINEAR", 
    "num_round": 10000, 
    "objective": "reg:linear",
    "colsample_bytree": 0.7,
    "max_depth": 9,
    "min_child_weight":8,
    "scale_pos_weight":1,
    "subsample":0.7
}

train_preds,preds = layer.predict(train_array, labels, test_array,parameter=parameter)
evaluate_value = evaluation_functions.evaluate_function(labels,train_preds,'rmsle')
print evaluate_value
#相関はどの程度あるのか