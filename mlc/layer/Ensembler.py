#coding:utf-8
import Layer
import numpy as np
import pickle
import os

class EmsambleLayer(object):
	def __init__(self):
		pass

	def predict(self,parameters):
		pass

class EmsambleLayerRegression(EmsambleLayer):
	def __init__(self):
		pass

class EmsambleLayerBinaryClassifier(EmsambleLayer):
	def __init__(self):
		pass

	def predict(self,parameters):
		train_prediction_array = []
		test_prediction_array = []
		for parameter in parameters:
			model_parameter = parameter["parameter"]
			feature_filename = parameter["feature_path"]
			model_id_file = parameter["id"] + ".pkl"
			train = None
			test = None
			
			if os.path.exists(model_id_file):
				print "load",model_id_file
				train,test = pickle.load(model_id_file)
			else:
				print "prediction",model_id_file
				train,labels,test = pickle.load(open(feature_filename,"r"))
				clf = Layer.ClassificationBinaryBaggingLayer()
				train,test = clf.predict_proba(train,labels,test,model_parameter)

				f = file(model_id_file, 'w')
				pickle.dump((train,test), f)
			train_prediction_array.append(train)
			test_prediction_array.append(test)
				#TODO:結果保存
		train_prediction_array = np.array(train_prediction_array).T
		test_prediction_array = np.array(test_prediction_array).T

		return train_prediction_array,test_prediction_array