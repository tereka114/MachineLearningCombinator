import numpy as np
from hyperopt import hp,fmin,tpe,Trials
from sklearn import cross_validation
from ..utility.config import parameter_dictionary
from ..utility.evaluation_functions import evaluate_function
from ..utility.Util import model_select
from ..utility.config import parameter_dictionary

def optimize_model_function(params,x,y,validation_indexs,evaluate_function_name):
	"""
	optimize parameter

	:param params:
	:param x:
	:param y:
	:param validation_indexs:
	:param evaluate_function_name:
	"""
	print params
	evals = np.zeros((len(validation_indexs)))
	cnt = 0
	for i,validation_index in enumerate(validation_indexs):
		x_train, x_test = x[validation_index[0]],x[validation_index[1]]
		y_train, y_test = y[validation_index[0]],y[validation_index[1]]

		clf = model_select(params)
		clf.fit(x_train, y_train)

		y_pred = None
		score = 0.0

		y_pred = clf.predict(x_test)
		if evaluate_function_name == "accuracy":
			y_pred = clf.predict(x_test)
			score = evaluate_function(y_test,y_pred,evaluate_function_name)
			score = -score
		elif evaluate_function_name == "logloss":
			y_pred = clf.predict_proba(x_test)
			score = evaluate_function(y_test,y_pred,evaluate_function_name)
			score = -score
		elif evaluate_function_name == "mean_squared_error":
			y_pred = clf.predict(x_test)
			score = evaluate_function(y_test,y_pred,evaluate_function_name)
		elif evaluate_function_name == "gini":
			y_pred = clf.predict(x_test)
			score = evaluate_function(y_test,y_pred,evaluate_function_name)
			score = -score
		elif evaluate_function_name == "rmsle":
			y_pred = np.expm1(clf.predict(x_test))
			score = evaluate_function(y_test,y_pred,evaluate_function_name)
			score = score
		cnt = cnt + 1
		#print cnt,params['model'],score
		print score
		evals[i] = score

	evaluate_score = np.mean(evals)
	print "final result",evaluate_score
	return evaluate_score

def optimize_model_parameter(x,y,model_name=None,loss_function="accuracy",parameter=None,max_evals=100):
	"""
	using hyperopt
	"""
	model_turning_params = None
	if model_name == None and parameter == None:
		print "you must set parameter or model_name"
		return None
	elif parameter != None:
		param = parameter
	elif model_name != None:
		param = parameter_dictionary[model_name]
	else:
		return None

	stop_flag = False
	validation_indexs = []
	cnt = 0

	for train_index,test_index in cross_validation.KFold(n=len(x),n_folds=10):
		validation_indexs.append((train_index,test_index))

	trials = Trials()
	function = lambda param : optimize_model_function(param, x, y, validation_indexs,loss_function)
	print param
	print "========================================================================"
	best_param = fmin(function,param,
		algo=tpe.suggest,max_evals=max_evals,trials=trials)
	print "========================================================================"
	print trials.trials[0]
	print trials.trials[1]
	return best_param

"""
calculate linear weight for minimize function
"""
def optimize_linear_weight(params,train_x,train_y,evaluate_function_name):
	result = np.zeros(train_y[0].shape)
	bagging_models = len(train_x)
	sum_number = 0.0

	for index in xrange(bagging_models):
		sum_number = sum_number + params[index]
	weight_params = [params[index] / sum_number for index in xrange(bagging_models)]

	for index in xrange(bagging_models):
		result = result + weight_params[index] * train_x[index]
	y_pred = result

	score = 0.0

	if evaluate_function_name == "accuracy":
		score = evaluate_function(train_y,y_pred,evaluate_function_name)
		score = -score
	elif evaluate_function_name == "logloss":
		score = evaluate_function(train_y,y_pred,evaluate_function_name)
		score = -score
	elif evaluate_function_name == "mean_squared_error":
		score = -evaluate_function(train_y,y_pred,evaluate_function_name)
	elif evaluate_function_name == "gini":
		score = -evaluate_function(train_y,y_pred,evaluate_function_name)
	print score
	return score
