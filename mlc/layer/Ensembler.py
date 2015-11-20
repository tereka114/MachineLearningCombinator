#coding:utf-8
import layer
import numpy as np
import pickle
import os
from ..utility.Util import create_directory,model_select
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.linear_model import LassoCV

class EnsambleLayer(object):
	def __init__(self):
		pass

class EnsambleLayerBinaryClassifier(EnsambleLayer):
	def __init__(self):
		pass

	def predict_proba(self,train_x,train_y,test_x,parameters):
		"""
		:param train_x: train_x
		"""
		ensemble_parameters = parameters["ensemble_parameters"]
		folder_name = parameters["ensanble_name"]
		train_result_array = []
		test_result_array = []
		for index,parameter in enumerate(ensemble_parameters):
			clf = layer.ClassificationBinaryLayer()
			model_parameter= parameter['model_parameter']

			filename = os.path.join(folder_name,str(index) + ".pkl")
			create_directory(folder_name)

			if os.path.exists(filename):
				train_predict_proba,test_predict_proba = pickle.load(open(filename,"r"))
			else:
				clf = None
				if not "type" in parameter:
					print "This parameter is not set model parameter"
					clf = layer.ClassificationBinaryBaggingLayer()
				elif parameter['type'] == 'bagging':
					clf = layer.ClassificationBinaryBaggingLayer()
				else:
					clf = layer.ClassificationBinaryLayer()

				train_predict_proba,test_predict_proba = clf.predict_proba(train_x,train_y,test_x,model_parameter)
				pickle.dump((train_predict_proba,test_predict_proba),open(filename,"w"))

			train_result_array.append(train_predict_proba)
			test_result_array.append(test_predict_proba)
		return np.array(train_result_array).T,np.array(test_result_array).T

class EnsambleLayerMultiClassifier(EnsambleLayer):
	def __init__(self):
		pass

	def predict_proba(self,train_x,train_y,test_x,parameters):
		ensemble_parameters = parameters["ensemble_parameters"]
		train_result_array = []
		test_result_array = []
		for parameter in ensemble_parameters:
			clf = layer.ClassificationLayer()
			model_parameter= parameter['model_parameter']
			#parameter
			clf = None
			if not "type" in parameter:
				print "This parameter is not set model parameter"
				clf = layer.ClassificationMultiBaggingLayer()
			elif parameter['type'] == 'bagging':
				clf = layer.ClassificationMultiBaggingLayer()
			else:
				clf = layer.ClassificationLayer()

			train_predict_proba,test_predict_proba = clf.predict_proba(train_x,train_y,test_x,model_parameter)

			train_result_array.append(train_predict_proba)
			test_result_array.append(test_predict_proba)
			print train_predict_proba[0]
		print np.hstack(train_result_array)[0]
		return np.hstack(train_result_array),np.hstack(test_result_array)

class BlendingLayer(object):
	def __init__(self):
		pass

	def preprocess_label(self):
		pass

class BlendingRegression(object):
	def __init__(self):
		pass

	def predict(self,trains_x,train_y,tests_x,parameters,times=10,isFile=True,foldername="blend-dir"):
		"""
		Ensamble many features and regression

		:params train_X: dictionary for training
		:params train_y: testing vector
		"""
		#parameter_get
		test_data_sample = tests_x.values()[0]

		if not os.path.exists(foldername):
			os.makedirs(foldername)

		skf = None
		kfold_file = foldername + "/kfold_index.pkl"
		if os.path.exists(kfold_file):
			skf = pickle.load(open(kfold_file,"r"))
		else:
			skf = KFold(n=len(train_y),n_folds=times,shuffle=True)
			pickle.dump(skf,open(kfold_file,"w"))

		blend_train = np.zeros((len(train_y),len(parameters)))
		blend_test = np.zeros((len(test_data_sample),len(parameters)))

		for j,parameter in enumerate(parameters):
			train_x = trains_x[parameter['data']]
			test_x = tests_x[parameter['data']]

			blend_test_tmp = np.zeros((len(test_data_sample),len(parameters)))

			#file path check
			for i, (train_index,valid_index) in enumerate(skf):
				clf = model_select(parameter['parameter'])

				train = train_x[train_index]
				train_valid_y = train_y[train_index]

				kfold_filepath = "./" + foldername + "/parameter_{}_kfold_{}.pkl".format(j,i)

				if os.path.exists(kfold_filepath):
					blend_train_prediction,blend_test_prediction = pickle.load(open(kfold_filepath,"r"))
					blend_train[train_index,j] = np.expm1(clf.predict(train))
					blend_test_tmp[:,i] = np.expm1(clf.predict(test_x))
				else:
					clf.fit(train,np.log1p(train_valid_y))
					blend_train_prediction = np.expm1(clf.predict(train))
					blend_test_prediction = np.expm1(clf.predict(test_x))
					pickle.dump((blend_train_prediction,blend_test_prediction),open(kfold_filepath,"w"))

				blend_train[train_index,j] = blend_train_prediction
				blend_test_tmp[:,i] = blend_test_prediction
			blend_test[:,j] = blend_test_tmp.mean(1)

		#Blending Model
		bclf = LassoCV(n_alphas=100, alphas=None, normalize=True, cv=5, fit_intercept=True, max_iter=10000, positive=True)
		bclf.fit(blend_train, train_y)
		y_test_predict = bclf.predict(blend_test)

		return y_test_predict

class EnsambleLayerRegression(EnsambleLayer):
	def __init__(self):
		pass
		# Skf
		# for knw
		# 
		# 
		# Bagging

		# Neural Network CV
		# XGB CV