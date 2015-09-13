import numpy as np
from hyperopt import hp,fmin,tpe
from sklearn import cross_validation
from ..utility.config import parameter_dictionary
from ..utility.evaluation_functions import evaluate_function
from ..utility.Util import model_select

"""
Optimization Program
"""
def optimize_model_function(params,x,y,validation_indexs,evaluate_function_name):
	print params
	evals = np.zeros((len(validation_indexs),len(validation_indexs[0])))
	cnt = 0
	for i,validation_index_list in enumerate(validation_indexs):
		for j,validation_index in enumerate(validation_index_list):
			x_train, x_test = x[validation_index[0]],x[validation_index[1]]
			y_train, y_test = np.log1p(y[validation_index[0]]),y[validation_index[1]]

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
			evals[i][j] = score
	evaluate_score = np.mean(evals)
	print "final result",evaluate_score
	return evaluate_score

def optimize_model_parameter(x,y,model_name=None,times=10,loss_function="accuracy",parameter=None,max_evals=100,total_time=None):
	model_turning_params = None
	if model_name == None and parameter == None:
		print "you must set parameter or model_name"
		return None
	elif parameter != None:
		model_turning_params = parameter
	elif model_name != None:
		config.parameter_dictionary[model_name]
	else:
		return None

	if total_time == None:
		total_time = 99999

	stop_flag = False
	validation_indexs = []
	cnt = 0
	for time in xrange(times):
		if stop_flag:
			break
		validation_index_list = []

		for train_index,test_index in cross_validation.KFold(n=len(x),n_folds=5):
			if stop_flag:
				break
			validation_index_list.append((train_index,test_index))
			cnt = cnt + 1

			if total_time < cnt:
				stop_flag = True

		validation_indexs.append(validation_index_list)
		config.parameter_dictionary[model_name]

	function = lambda param : optimize_model_function(param, x, y, validation_indexs,loss_function)
	print parameter
	print "========================================================================"
	best_param = fmin(function,model_turning_params,
		algo=tpe.suggest,max_evals=max_evals)
	print "========================================================================"
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
