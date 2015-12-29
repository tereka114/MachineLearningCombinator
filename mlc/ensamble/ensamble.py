import numpy as np
from sklearn.cross_validation import KFold
from hyperopt import hp, fmin, tpe, Trials
import os
import pickle
from ..utility.Util import model_select
from ..utility.evaluation_functions import evaluate_function

class EnsamblePredictor(object):
	def __init__(self):
		pass

	def exists(self,filename):
		if os.path.exists(filename):
			return True
		return False

	def _ensamble_process(self,p1,p2,weight):
		return p1 * (1 - weight) + p2 * weight

	def ensamble_param_hyper_opt(self,valid1,valid2,parameter,evaluate_function_name,test_y_array):
		"""
		:parameter:
		"""
		weight = parameter['best_w']
		evaluation_tmp = np.zeros((len(valid1),len(valid1[0])))
		for time,(valid_time1,valid_time2) in enumerate(zip(valid1,valid2)):
			for index,(valid_kf1,valid_kf2,test_y) in enumerate(zip(valid_time1,valid_time2,test_y_array)):
				ensamble_predict = self._ensamble_process(valid_kf1,valid_kf2,weight)
				tmp_score = evaluate_function(test_y,ensamble_predict,evaluate_function_name)
				evaluation_tmp[time][index] = tmp_score
		return np.mean(evaluation_tmp)

	def ensamble_fit(self,x,y,n_folds,times,parameters,evaluate_function_name,out_tmp_dir="ensamble_tmp"):
		kf = KFold(len(y),n_folds,shuffle=True,random_state=71)
		evaluation_cv = []
		valid_all_prediction = []

		if not self.exists(out_tmp_dir):
			os.makedirs(out_tmp_dir)

		#build test_y
		test_y_array = [y[test_index] for train_index,test_index in kf]

		for parameter_index,parameter in enumerate(parameters):
			evaluation_tmp = np.zeros((times,n_folds))

			time_prediction = []

			for time in xrange(times):
				kf_prediction = []
				for index,(train_index,test_index) in enumerate(kf):
					train_x,train_y = x[train_index],y[train_index]
					test_x,test_y = x[test_index],y[test_index]

					filename = "{}_{}_{}.pkl".format(parameter_index,time,index)
					filepath = os.path.join(out_tmp_dir,filename)

					valid_prediction = None
					if self.exists(filepath):
						valid_prediction = pickle.load(open(filepath,"r"))
					else:
						clf = model_select(parameter)
						clf.fit(train_x,train_y)
						if parameter['model'] == "XGBREGLOGISTIC":
							valid_prediction = self.clf.predict_proba(train_x)
						else:
							valid_prediction = clf.predict(test_x)
						pickle.dump(valid_prediction, open(filepath,"w"))

					kf_prediction.append(valid_prediction)
					tmp_score = evaluate_function(test_y,valid_prediction,evaluate_function_name)
					evaluation_tmp[time][index] = tmp_score
				time_prediction.append(kf_prediction)
			valid_all_prediction.append(time_prediction)
			evaluation_cv.append(np.mean(evaluation_tmp))

		evaluation_cv = np.array(evaluation_cv)
		sorted_model = np.argsort(evaluation_cv)[::-1]

		print evaluation_cv,sorted_model
		for mid in sorted_model:
			print parameters[mid]

		valid_evaluate_prediction = None
		best_score = evaluation_cv[sorted_model[0]]

		weighted_array = []

		for index,mid in enumerate(sorted_model):
			if index == 0:
				valid_evaluate_prediction = valid_all_prediction[mid]
			else:
				# set hyperopt turning
				bestw_parameter = {
					"best_w": hp.quniform("best_w",0.01,1,0.01)
				}

				trials = Trials()
				max_evals = 200
				bestw_function = lambda bestw_parameter: self.ensamble_param_hyper_opt(valid_evaluate_prediction,valid_all_prediction[mid],bestw_parameter,evaluate_function_name,test_y_array)
				best_w_result = fmin(bestw_function,bestw_parameter,algo=tpe.suggest, max_evals=max_evals, trials=trials)
				best_w = best_w_result['best_w']
				print best_w_result
				#print best_w,trials.results

				evaluation_tmp = np.zeros((len(valid_evaluate_prediction),len(valid_evaluate_prediction[0])))
				tmp_prediction_result = []

				for time,(valid_time1,valid_time2) in enumerate(zip(valid_evaluate_prediction,valid_all_prediction[mid])):
					tmp_prediction_result_per_time = []

					for index,(valid_kf1,valid_kf2,test_y) in enumerate(zip(valid_time1,valid_time2,test_y_array)):
						ensamble_predict = self._ensamble_process(valid_kf1,valid_kf2,best_w)
						tmp_score = evaluate_function(test_y,ensamble_predict,evaluate_function_name)
						evaluation_tmp[time][index] = tmp_score
						tmp_prediction_result_per_time.append(ensamble_predict)
					tmp_prediction_result.append(tmp_prediction_result_per_time)
				score = np.mean(evaluation_tmp)
				print evaluation_tmp

				if best_score < score:
					valid_evaluate_prediction = tmp_prediction_result
					weighted_array.append(best_w)
					best_score = score
				else:
					weighted_array.append(0)
				print score,best_score
		#print out score
		self.parameters = [parameters[mid] for mid in sorted_model]
		print best_score
		print weighted_array
		return self.parameters,weighted_array

	def fit(self,train_x,train_y,parameters=None):
		if not self.parameters == None:
			parameters = self.parameters

		classifier_list = []
		for parameter in parameters:
			clf = model_select(parameter)
			clf.fit(train_x,train_y)
			classifier_list.append(clf)

		self.classifier_list = classifier_list

	def predict(self,test_x):
		test_predictions = np.zeros((len(self.classifier_list),len(test_x)))
		for index,clf in enumerate(self.classifier_list):
			if self.parameters[index]['model'] == "XGBREGLOGISTIC":
				test_predictions[index] = clf.predict_proba(test_x)
			else:
				test_predictions[index] = clf.predict(test_x)

		prediction = None
		for index,(test_prediction,weight) in enumerate(zip(test_predictions,self.weighted_array)):
			if index == 0:
				prediction = test_prediction
				continue
			prediction = self._ensamble_process(prediction, test_prediction, weight)
		return prediction