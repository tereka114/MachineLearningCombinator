#coding:utf-8
import layer
import numpy as np
import pickle
import os
from ..utility.Util import create_directory,model_select
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.linear_model import LassoCV

class BlendingLayer(object):
	def __init__(self):
		pass

	def preprocess_label(self):
		pass

class BlendingRegression(BlendingLayer):
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

class BlendingBinaryClassifier(BlendingLayer):
	def __init__(self):
		pass

	def predict_proba(self,trains_x,train_y,tests_x,parameters,times=10,isFile=True,foldername="blend-dir"):
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

				#要変更
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

class StackingLayer(object):
	def __init__(self):
		pass

class EnsambleLayer(object):
	def __init__(self):
		pass

class EmsambleRegression(object):
	def __init__(self):
		pass

	def predict(self):
		pass