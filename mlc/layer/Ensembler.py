#coding:utf-8
import layer
import numpy as np
import pickle
import os
from ..utility.Util import create_directory

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

class EnsambleLayerRegression(EnsambleLayer):
	def __init__(self):
		pass

	def predict(self,train_x,train_y,test_x,parameters):
		ensemble_parameters = parameters["ensemble_parameters"]
		folder_name = parameters["ensanble_name"]

		train_result_array = []
		test_result_array = []
		train_predict_result = None
		test_predict_result = None

		for index,parameter in enumerate(ensemble_parameters):
			clf = None
			model_parameter= parameter['model_parameter']

			filename = os.path.join(folder_name,str(index) + ".pkl")
			create_directory(folder_name)

			if os.path.exists(filename):
				train_predict_proba,test_predict_proba = pickle.load(open(filename,"r"))
			else:
				if not "type" in parameter:
					print "This parameter is not set model parameter"
					clf = layer.RegressionBaggingLayer()
				elif parameter['type'] == 'bagging':
					clf = layer.RegressionBaggingLayer()
				else:
					clf = layer.RegressionBaggingLayer()

				train_predict_result,test_predict_result = clf.predict(train_x,train_y,test_x,model_parameter)
				pickle.dump((train_predict_result,test_predict_result),open(filename,"w"))

			train_result_array.append(train_predict_result)
			test_result_array.append(test_predict_result)
		return np.array(train_result_array).T,np.array(test_result_array).T