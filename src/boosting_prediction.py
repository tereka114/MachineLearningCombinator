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

import argparse

"""
you should read the pickle file
"""
def option_parse():
	parser = argparse.ArgumentParser(description='This script is that machine predict the result.you can choose bagging or signle')
	parser.add_argument('-i','--input_file',
		type=str
		)
	parser.add_argument('-p','--input_parameter',
		type=str
		)
	parser.add_argument('-l','--learning_model',
		type=str
		)
	args = parser.parse_args()
	return args

args = option_parse()

feature_pkl_file = args.input_file
parameter_file_json = args.input_parameter
learning_model = args.learning_model

filename = feature_pkl_file
train,labels,test = pickle.load(open(filename,"r"))

feature_name,ext = os.path.splitext(os.path.basename(filename))

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
#layer_zone.models_dump(['XGBREGLINEAR','XGBREGLOGISTIC','RIDGE'], train,labels, test, 'gini', '')

model_filename = parameter_file_json
name,ext = os.path.splitext(os.path.basename(model_filename))

with open(model_filename) as data_file:    
    parameter = json.load(data_file)
train_predict,test_predict = layer.predict_proba(train,labels, test,parameter=parameter)

if not os.path.exists("train"):
	os.mkdir("train")
pickle.dump(train_predict, open("./train/" + name + "_" + feature_name + "_" + learning_model + ".pkl","w"))

if not os.path.exists("test"):
	os.mkdir("test")
pickle.dump(test_predict, open("./test/" + name + "_" + feature_name + "_" + learning_model + ".pkl","w"))