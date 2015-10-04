#coding:utf-8
import mlc.parameter_tunes.optimize
from sklearn.datasets import load_digits,load_diabetes
from sklearn.cross_validation import train_test_split
import mlc.layer.layer
from mlc.utility.evaluation_functions import evaluate_function
import numpy as np

def multi_classification():
	digits = load_digits()
	x = digits.data
	y = digits.target

	sample_parameter = {
		'n_jobs': -1,
		'min_samples_leaf': 2.0,
		'n_estimators': 500,
		'max_features': 0.55,
		'criterion': 'mse',
		'min_samples_split': 4.0,
		'model': 'RFCLF',
		'max_depth': 4.0
	}

	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

	clf_layer = mlc.layer.layer.ClassificationLayer()
	print "single prediction"
	y_train_predict,y_test_predict = clf_layer.predict(x_train,y_train,x_test,sample_parameter)
	#print y_test_predict
	y_train_predict_proba,y_test_predict_proba = clf_layer.predict_proba(x_train,y_train,x_test,sample_parameter)
	#print y_test_predict_proba
	print evaluate_function(y_test,np.argmax(y_test_predict_proba,axis=1),'accuracy')

	print "multi ensamble prediction"

	multi_bagging_clf = mlc.layer.layer.ClassificationMultiBaggingLayer()
	y_train_predict_proba,y_test_predict_proba = multi_bagging_clf.predict_proba(x_train,y_train,x_test,sample_parameter,times=5)

	print evaluate_function(y_test,np.argmax(y_test_predict_proba,axis=1),'accuracy')

def binary_classification():
	digits = load_digits(n_class=2)
	x = digits.data
	y = digits.target

	sample_parameter = {
    	"colsample_bytree": 0.9,
    	"min_child_weight": 10,
    	"num_round": 300,
    	"subsample": 0.7, 
    	"eta": 0.2,
    	"max_depth": 4, 
    	"gamma": 0.6000000000000001,
    	"model": "XGBREGLOGISTIC",
    	"objective": "binary:logistic"
    }
	
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

	clf_layer = mlc.layer.layer.ClassificationBinaryLayer()
	print "single prediction"
	#y_train_predict,y_test_predict = clf_layer.predict(x_train,y_train,x_test,sample_parameter)
	#print y_test_predict
	y_train_predict_proba,y_test_predict_proba = clf_layer.predict_proba(x_train,y_train,x_test,sample_parameter)
	#print y_test_predict_proba
	print evaluate_function(y_test,y_test_predict_proba,'auc')

	print "multi ensamble prediction"

	multi_bagging_clf = mlc.layer.layer.ClassificationBinaryBaggingLayer()
	y_train_predict_proba,y_test_predict_proba = multi_bagging_clf.predict_proba(x_train,y_train,x_test,sample_parameter,times=5)

	print evaluate_function(y_test,y_test_predict_proba,'auc')

"""
todo:add regression
"""
def bagging_regression():
	digits = load_diabetes()
	x = digits.data
	y = digits.target

	sample_parameter = {
		'n_jobs': -1,
		'min_samples_leaf': 2.0,
		'n_estimators': 500,
		'max_features': 0.55,
		'criterion': 'mse',
		'min_samples_split': 4.0,
		'model': 'RFREG',
		'max_depth': 4.0
	}

	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

	clf_layer = mlc.layer.layer.RegressionLayer()
	print "single prediction"
	#y_train_predict,y_test_predict = clf_layer.predict(x_train,y_train,x_test,sample_parameter)
	#print y_test_predict
	y_train_predict_proba,y_test_predict_proba = clf_layer.predict(x_train,y_train,x_test,sample_parameter)
	#print y_test_predict_proba
	print evaluate_function(y_test,y_test_predict_proba,'mean_squared_error')

	print "multi ensamble prediction"

	multi_bagging_clf = mlc.layer.layer.RegressionBaggingLayer()
	y_train_predict_proba,y_test_predict_proba = multi_bagging_clf.predict(x_train,y_train,x_test,sample_parameter,times=5)

	print evaluate_function(y_test,y_test_predict_proba,'mean_squared_error')

if __name__ == '__main__':
	# print "start binary classification"
	# binary_classification()
	# print "end binary classification"

	print "start multi classification"
	multi_classification()
	print "end multi classification"

	bagging_regression()