#coding:utf-8
import Layer
import feature_vector
import pandas as pd
import numpy as np 
from sklearn import preprocessing
import pickle
import json
import os
import Util
import sys


validation_indexs = None

"""
you should read the pickle file
"""
feature_pkl_file = sys.argv[1]
parameter_file_json = sys.argv[2]

filename = feature_pkl_file
train,labels,test = pickle.load(open(filename,"r"))

validation_time = 1

if os.path.exists("validation_list.pkl"):
	validation_indexs = pickle.load(open("validation_list.pkl","r"))
else:
	validation_indexs = Util.genIndexKFold(labels, validation_time)
	pickle.dump(validation_indexs,open("validation_list.pkl","w"))

feature_name,ext = os.path.splitext(os.path.basename(filename))
layer_zone = Layer.RegressionBaggingLayer()
#layer_zone.models_dump(['XGBREGLINEAR','XGBREGLOGISTIC','RIDGE'], train,labels, test, 'gini', '')

model_filename = parameter_file_json
name,ext = os.path.splitext(os.path.basename(model_filename))

with open(model_filename) as data_file:    
    parameter = json.load(data_file)
train_predict,test_predict = layer_zone.predict(train,labels, test,parameter=parameter)
print train_predict,test_predict

if not os.path.exists("train"):
	os.mkdir("train")
pickle.dump(train_predict, open("./train/" + name + "_" + feature_name + ".pkl","w"))

if not os.path.exists("test"):
	os.mkdir("test")
pickle.dump(test_predict, open("./test/" + name + "_" + feature_name + ".pkl","w"))

#Stackingの結果
# layer = Layer.Layer()
# test_pred,predict = layer.predict(train,labels, test, {'model':'RFREG','n_estimators':100,'max_features':0.05}, None)
# print predict
# #generate solution
# preds = pd.DataFrame({"Id": test_ind, "Hazard": predict})
# preds = preds.set_index('Id')
# preds.to_csv('xgboost_benchmark.csv')
