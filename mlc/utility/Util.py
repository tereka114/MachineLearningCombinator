from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR,SVC
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso, LassoLars, ElasticNet
import numpy as np
from ..model.XGBoost import XGBoostRegressor,XGBoostClassifier
from ..model.ChainerNeuralNetwork import ChainerNeuralNetworkModel,ChainerNeuralNetwork
from ..model.LasagneNeuralNetwork import NeuralNetwork
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import os

def model_select(parameter):
	model_name = parameter['model']
	if model_name == "LOGISTIC":
		return LogisticRegression(C=parameter['C'])
	elif model_name == "SVR":
		return SVR(C=parameter['C'],gamma=parameter['gamma'],kernel=parameter['kernel'])
	elif model_name == "EXTREECLF":
		return ExtraTreesClassifier(n_estimators=int(parameter['n_estimators']),
			max_depth=int(parameter['max_depth']),
			min_samples_leaf=int(parameter['min_samples_leaf']),
			min_samples_split=int(parameter['min_samples_split']),
			random_state=int(parameter['seed']))
	elif model_name == "RFREG":
		return RandomForestRegressor(
			n_estimators=int(parameter['n_estimators']),
			max_features=parameter['max_features'],
			max_depth=int(parameter['max_depth']),
			min_samples_leaf=int(parameter["min_samples_leaf"]),
			min_samples_split=int(parameter['min_samples_split']),
			random_state=int(parameter['seed'])
			)
	elif model_name == "RFCLF":
		return RandomForestClassifier(
			n_estimators=int(parameter['n_estimators']),
			max_features=parameter['max_features'],
			max_depth=int(parameter['max_depth']),
			min_samples_leaf=int(parameter["min_samples_leaf"]),
			min_samples_split=int(parameter['min_samples_split']),
			random_state=int(parameter['seed'])
			)
	elif model_name == "SVC":
		return SVC(C=parameter["C"],
			kernel=parameter["kernel"],
			degree=int(parameter["degree"]))
	elif model_name == 'XGBREGLINEAR':
		params = {}
		params['objective'] = parameter['objective']
		params['eta'] = parameter['eta']
		params['max_depth'] = int(parameter['max_depth'])
		params['subsample'] = parameter['subsample']
		params['colsample_bytree'] = parameter['colsample_bytree']
		params['min_child_weight'] = int(parameter['min_child_weight'])
		params['seed'] = parameter['seed']
		params['silent'] = 1
		return XGBoostRegressor(int(parameter['num_round']),**params)
	elif model_name == "XGBREGTREE":
		params = {}
		params['objective'] = parameter['objective']
		params['booster'] = parameter['booster']
		params['eta'] = parameter['eta']
		params['max_depth'] = int(parameter['max_depth'])
		params['subsample'] = parameter['subsample']
		params['colsample_bytree'] = parameter['colsample_bytree']
		params['min_child_weight'] = int(parameter['min_child_weight'])
		params['seed'] = parameter['seed']
		params['silent'] = 1
		return XGBoostRegressor(int(parameter['num_round']),**params)
	elif model_name == 'XGBREGLOGISTIC':
		params = {}
		params['objective'] = parameter['objective']
		params['eta'] = parameter['eta']
		params['min_child_weight'] = int(parameter['min_child_weight'])
		params['max_depth'] = int(parameter['max_depth'])
		params['subsample'] = parameter['subsample']
		params['colsample_bytree'] = parameter['colsample_bytree']
		params['eval_metric'] = "auc"
		params['silent'] = 1
		return XGBoostClassifier(int(parameter['num_round']),**params)
	elif model_name == "XGBMULCLASSIFIER":
		params = {}
		params['objective'] = parameter['objective']
		params['eta'] = parameter['eta']
		params['min_child_weight'] = int(parameter['min_child_weight'])
		params['max_depth'] = int(parameter['max_depth'])
		params['subsample'] = parameter['subsample']
		params['colsample_bytree'] = parameter['colsample_bytree']
		params['eval_metric'] = "mlogloss"
		params['silent'] = 1
		params['num_class'] = parameter['num_class']
		return XGBoostClassifier(int(parameter['num_round']),**params)
	elif model_name == 'LASSO':
		return Lasso(alpha=parameter['alpha'], normalize=True)
	elif model_name == 'RIDGE':
		return Ridge(alpha=parameter["alpha"], normalize=True)
	elif model_name == 'EXTREEREG':
		return ExtraTreesRegressor(n_estimators=int(parameter['n_estimators']),
			max_depth=int(parameter['max_depth']),
			min_samples_leaf=int(parameter['min_samples_leaf']),
			min_samples_split=int(parameter['min_samples_split']),
			random_state=int(parameter['seed']))
	elif model_name == 'DECISIONTREEREG':
		return DecisionTreeRegressor(n_estimators=int(parameter['n_estimators']),
			max_depth=int(parameter['max_depth']),
			min_samples_leaf=int(parameter['min_samples_leaf']),
			min_samples_split=int(parameter['min_samples_split']),
			random_state=int(parameter['seed']))
	elif model_name == 'ChainerNeuralNetworkRegression':
		model = ChainerNeuralNetworkModel(problem_type='regression',layer1=int(parameter['layer1']),layer2=int(parameter['layer2']),layer3=int(parameter['layer3']),n_in=int(parameter['n_in']),n_out=int(parameter['n_out']),dropout1=int(parameter['dropout1']),dropout2=int(parameter['dropout2']),dropout3=int(parameter['dropout3']))
		return ChainerNeuralNetwork(problem_type='regression',batch_size=100, cuda=parameter['cuda'], epoch=parameter['epoch'],model=model,seed=2015,evaluate_function_name=parameter['evaluate_function'],convert=parameter['convert'])
	elif model_name == 'GBR':
		return GradientBoostingRegressor(n_estimators=100,
			max_depth=10,
			max_features=0.7)
	elif model_name == 'LasagneNeuralNetworkRegression':
		return NeuralNetwork(epochs=int(parameter['epochs']))
	elif model_name == 'LasagneNeuralNetworkClassification':
		return NeuralNetwork(epochs=int(parameter['epochs']),problem_type="classification",dropout_layer=parameter['dropout_layer'],layer_number=parameter['layer_number'])
	elif model_name == 'KNN':
		return KNeighborsClassifier(n_neighbors=int(parameter['n_neighbors']))

def genIndexKFold(x,times):
	skf = KFold(n=len(x),n_folds=times,shuffle=True)
	return skf

def genIndexStratifiedKFold(y,times):
	skf = StratifiedKFold(y,n_folds=times,shuffle=True)
	return skf

def cross_inspect_feature():
	pass

def wrapper_learning_selector():
	pass

def stack_feature(clf,train,labels,test):
	clf.fit(train,labels)

	pred_train = clf.predict(train).reshape((len(train)),1)
	pred_test = clf.predict(test).reshape((len(test)),1)

	train_s = np.hstack((train,pred_train))
	test_s = np.hstack((test,pred_test))

	return train_s,test_s

def create_directory(path):
	if not os.path.exists(path):
		os.mkdir(path)
