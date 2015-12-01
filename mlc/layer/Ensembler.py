#coding:utf-8
import layer
import numpy as np
import pickle
import os
from ..utility.Util import create_directory,model_select
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.linear_model import LassoCV
import random
import pandas as pd
from ..utility.evaluation_functions import evaluate_function
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
		skf = KFold(n=len(train_y),n_folds=times,shuffle=True,random_state=71)

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


class StackingRegressionLayer(object):
	def __init__(self):
		pass

	def stacking(self,train_x=None,train_y=None,train_ids=None,test_ids=None,test_x=None,parameter=None,
		epochs=1,n_folds=10,isTrain=True,id_name=None,scoreName=None,output_dir=None,pred_name=None,evaluation_name=""):
		kfold = KFold(n=len(train_y),n_folds=n_folds,shuffle=True,random_state=71)

		train_dir = '{}/train_row'.format(output_dir)
		test_dir = '{}/test_row'.format(output_dir)

		if not os.path.exists(train_dir):
			os.makedirs(train_dir)

		if not os.path.exists(test_dir):
			os.makedirs(test_dir)

		for epoch in xrange(epochs):
			print "processing iteration",epoch
			#random state
			seed = 71 + epoch * 100
			parameter['seed'] = seed
			if isTrain:
				prediction = np.zeros((len(train_y)))
				preds_epoch = pd.DataFrame()

				train_csv_name = '{}/{}.epoch{}.csv'.format(train_dir,pred_name,epoch)

				if os.path.exists(train_csv_name):
					print "already exists"
					continue

				for index,(train_index,valid_index) in enumerate(kfold):
					print "kfold",index
					train_x_fold,valid_x_fold = train_x[train_index],train_x[valid_index]
					train_y_fold,valid_y_fold = train_y[train_index],train_y[valid_index]

					index_shuffle = [i for i in range(train_x_fold.shape[0])]
					random.shuffle(index_shuffle)

					clf = model_select(parameter)
					clf.fit(train_x_fold[index_shuffle],np.log1p(train_y_fold[index_shuffle]))
					prediction[valid_index] = np.expm1(clf.predict(valid_x_fold))

				# add evaluation function
				print evaluate_function(train_y,prediction,"rmspe")

				preds_epoch[id_name] = train_ids
				preds_epoch[scoreName] = prediction
				preds_epoch.to_csv(train_csv_name, index=False)
				preds_epoch = preds_epoch.drop(id_name, axis=1)
			else:
				preds_epoch = pd.DataFrame()

				test_csv_name = '{}/{}.epoch{}.csv'.format(test_dir,pred_name,epoch)
				if os.path.exists(test_csv_name):
					print "already exists"
					continue

				index_shuffle = [i for i in range(train_x.shape[0])]
				random.shuffle(index_shuffle)

				clf = model_select(parameter)
				clf.fit(train_x[index_shuffle],np.log1p(train_y[index_shuffle]))
				prediction = np.expm1(clf.predict(test_x))

				preds_epoch[id_name] = test_ids
				preds_epoch[scoreName] = prediction
				preds_epoch.to_csv(test_csv_name, index=False)
				preds_epoch = preds_epoch.drop(id_name, axis=1)

class StackingBinaryClassification(object):
	def __init__(self):
		pass

	def stacking(self,train_x=None,train_y=None,train_ids=None,test_ids=None,test_x=None,parameter=None,
		epochs=1,n_folds=10,isTrain=True,id_name=None,scoreName=None,output_dir=None,pred_name=None,evaluation_name=""):

		kfold = KFold(n=len(train_y),n_folds=n_folds,shuffle=True,random_state=71)

		train_dir = '{}/train_row'.format(output_dir)
		test_dir = '{}/test_row'.format(output_dir)

		if not os.path.exists(train_dir):
			os.makedirs(train_dir)

		if not os.path.exists(test_dir):
			os.makedirs(test_dir)

		for epoch in xrange(epochs):
			print "processing iteration",epoch
			#random state
			seed = 71 + epoch * 100
			parameter['seed'] = seed
			if isTrain:
				prediction = np.zeros((len(train_y)))
				preds_epoch = pd.DataFrame()

				train_csv_name = '{}/{}.epoch{}.csv'.format(train_dir,pred_name,epoch)

				if os.path.exists(train_csv_name):
					print "already exists"
					continue

				for index,(train_index,valid_index) in enumerate(kfold):
					print "kfold",index
					train_x_fold,valid_x_fold = train_x[train_index],train_x[valid_index]
					train_y_fold,valid_y_fold = train_y[train_index],train_y[valid_index]

					index_shuffle = [i for i in range(train_x_fold.shape[0])]
					random.shuffle(index_shuffle)

					clf = model_select(parameter)
					clf.fit(train_x_fold[index_shuffle],train_y_fold[index_shuffle])

					if parameter['model'] == "XGBREGLOGISTIC":
					    prediction[valid_index] = clf.predict_proba(valid_x_fold)
					else:
						prediction[valid_index] = clf.predict_proba(valid_x_fold)[:,1]

				# add evaluation function
				print evaluate_function(train_y,prediction,evaluation_name)

				preds_epoch[id_name] = train_ids
				preds_epoch[scoreName] = prediction
				preds_epoch.to_csv(train_csv_name, index=False)
				preds_epoch = preds_epoch.drop(id_name, axis=1)
			else:
				preds_epoch = pd.DataFrame()

				test_csv_name = '{}/{}.epoch{}.csv'.format(test_dir,pred_name,epoch)
				if os.path.exists(test_csv_name):
					print "already exists"
					continue

				index_shuffle = [i for i in range(train_x.shape[0])]
				random.shuffle(index_shuffle)

				clf = model_select(parameter)
				clf.fit(train_x[index_shuffle],train_y[index_shuffle])
				if parameter['model'] == "XGBREGLOGISTIC":
					prediction = clf.predict(test_x)
				else:
					prediction = clf.predict(test_x)[:,1]
					
				preds_epoch[id_name] = test_ids
				preds_epoch[scoreName] = prediction
				preds_epoch.to_csv(test_csv_name, index=False)
				preds_epoch = preds_epoch.drop(id_name, axis=1)

class EnsambleLayer(object):
	def __init__(self):
		pass

class BaggingRegression(object):
	def __init__(self):
		pass

	def bagging(self,trains,tests,train_y,model_name=None):
		blend_train = trains.T
		bclf = LassoCV(n_alphas=100, alphas=None, normalize=True, cv=5, fit_intercept=True, max_iter=10000, positive=True)
		bclf.fit(blend_train, train_y)
		y_test_predict = bclf.predict(tests.T)
		train_predict = bclf.predict(trains.T)

		return train_predict,y_test_predict

class BaggingRegressionAverage(object):
	def __init__(self):
		pass

	def bagging(self,data):
		return data.mean(0)

class EnsambleAverageRegression(object):
	def __init__(self):
		pass

	def predict(self,trains,tests,train_y,filename,id_name=None,scoreName=None,output_dir=None,pred_name=None):
		trains_bagging_score = trains.mean(0)
		tests_bagging_score = tests.mean(0)

		train_dir = '{}/'.format(output_dir)
		test_dir = '{}/'.format(output_dir)

		if not os.path.exists(train_dir):
			os.makedirs(train_dir)

		if not os.path.exists(test_dir):
			os.makedirs(test_dir)

		preds_epoch = pd.DataFrame()
		