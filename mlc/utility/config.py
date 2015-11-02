#coding:utf-8
from hyperopt import hp
import numpy as np

parameter_dictionary = {}
modelRandomSeed = 2015
debug_mode = False

if debug_mode:
	min_n_estimators = 2
	max_n_estimators = 10
	step_n_estimators = 1
	xgb_min_num_round = 1
	xgb_max_num_round = 10
	xgb_num_round_step = 1
	n_jobs = 4
else:
	min_n_estimators = 10
	max_n_estimators = 500
	step_n_estimators = 5
	n_jobs = -1
	xgb_min_num_round = 250
	xgb_max_num_round = 2000
	xgb_num_round_step = 10

BLENDING_TRAIN_DIR = "train"
BLENDING_TEST_DIR = "test"

BLENDING_FILES = [
	"XGBREGLINEAR_std_feature_vector_pca_f_std_t_one_of_k_f_label_t.pkl",
	"feature_vector2_std_RFREG_feature_vector_pca_f_std_t_one_of_k_f_label_t.pkl"
]

LOSS_FUNCTION = "RMSLE"

"""
Classification Model Parameters
"""
parameter_svc_classifier = {
}

parameter_extratrees_classifier = {
	'model':'EXTREEREG',
	'max_depth':hp.quniform("max_depth",3.0,20.0,1.0),
	'min_sample_leaf':hp.quniform("min_sample_leaf",1.0,10.0,1.0),
	'min_samples_split':hp.quniform("min_samples_split",1.0,10.0,1.0),
	'n_estimators':500,	
}

parameter_randomforest_classifier = {
	'model':'RFCLF',
	'n_estimators':500,
	'max_features':hp.quniform("max_features",0.5,1.0,0.05),
	'criterion':'mse',
	'min_samples_leaf':hp.quniform("min_samples_leaf",1.0,5.0,1.0),
	'min_samples_split':hp.quniform("min_samples_split",1.0,5.0,1.0),
	'max_depth':hp.quniform("max_depth",3.0,50.0,1.0),
	'n_jobs':n_jobs,
}

parameter_logisticclassifier_classifier = {
	'model':'logistic_classifier',
	'C': hp.loguniform("C", np.log(0.001), np.log(1000))
}

"""
Regression Model Parameters
"""
parameter_svr_regression = {
	'model':'SVR',
	'C': hp.loguniform("C", np.log(1), np.log(100)),
	'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
	'kernel': 'rbf',
	'random_state' : modelRandomSeed
}

parameter_randomforest_regression = {
	'model':'RFREG',
	'n_estimators':100,
	'max_features':hp.quniform("max_features",0.5,1.0,0.05),
	'criterion':'mse',
	'min_samples_leaf':hp.quniform("min_samples_leaf",1.0,5.0,1.0),
	'min_samples_split':hp.quniform("min_samples_split",1.0,5.0,1.0),
	'max_depth':hp.quniform("max_depth",3.0,50.0,1.0),
	'n_jobs':n_jobs,
	'random_state' : modelRandomSeed
}

parameter_extratree_reg = {
	'model':'EXTREEREG',
	'max_depth':hp.quniform("max_depth",3.0,50.0,1.0),
	'min_sample_leaf':hp.quniform("min_sample_leaf",1.0,10.0,1.0),
	'min_samples_split':hp.quniform("min_samples_split",1.0,10.0,1.0),
	'n_estimators':100,
	'random_state' : modelRandomSeed
}

parameter_xgboost_linear_regression = {
	'model':'XGBREGLINEAR',
	'objective': 'reg:linear',
	'booster' : 'gblinear',
	'num_round' : hp.quniform('n_estimators', 500, 2000, 1),
	'eta' : hp.quniform('eta', 0.01, 1, 0.01),
	'min_child_weight':hp.quniform('min_child_weight',1,20,1),
	'max_depth': hp.quniform('max_depth',4,20,1),
	'subsample': hp.quniform('subsample',0.5,1,0.1),
	"colsample_bytree": hp.quniform('colsample_bytree',0.5,1,0.05),
	"alpha": hp.quniform("alpha",0,0.5,0.005),
	"lambda": hp.quniform('lambda',0,5,0.05),
	"lambda_bias":hp.quniform('lambda_bias',0,3,0.1),
	'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
	'seed' : modelRandomSeed
}

parameter_xgboost_tree_regression = {
	'model':'XGBREGTREE',
	'objective': 'reg:linear',
	'booster': 'gbtree',
	'num_round' : hp.quniform('n_estimators', 500, 2000, 1),
	'eta' : hp.quniform('eta', 0.01, 1, 0.01),
	'min_child_weight':hp.quniform('min_child_weight',1,20,1),
	'max_depth': hp.quniform('max_depth',4,20,1),
	'subsample': hp.quniform('subsample',0.5,1,0.1),
	"colsample_bytree": hp.quniform('colsample_bytree',0.5,1,0.05),
	"alpha": hp.quniform("alpha",0,0.5,0.005),
	"lambda": hp.quniform('lambda',0,5,0.05),
	"lambda_bias":hp.quniform('lambda_bias',0,3,0.1),
	'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
	'seed' : modelRandomSeed
}

parameter_xgboost_logistic_regression = {
	'model':'XGBREGLOGISTIC',
	'objective': "binary:logistic",
	'eta' : hp.choice('eta', [0.025, 0.1, 0.3, 0.5, 1.0]),
	'min_child_weight':hp.quniform('min_child_weight',1,20,1),
	'max_depth': hp.quniform('max_depth',4,20,1),
	'subsample': hp.quniform('subsample',0.5,01,0.05),
	"colsample_bytree": hp.quniform('colsample_bytree',0.5,1,0.05),
	'num_round' : hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
	'seed' : modelRandomSeed
}

parameter_skl_ridge = {
	'model':'RIDGE',
	'alpha': hp.loguniform("alpha", np.log(0.01), np.log(20)),
	'random_state' : modelRandomSeed
}

parameter_skl_lasso = {
    'model':'LASSO',
    'alpha': hp.loguniform("alpha", np.log(0.00001), np.log(0.1)),
    'random_state' : modelRandomSeed
}

parameter_chainer_regression = {
	'model':'ChainerNeuralNetworkRegression',
	'layer1':hp.quniform('layer1',100,784,20),
	'layer2':hp.quniform('layer2',100,784,20)
}

parameter_dictionary['SVC'] = parameter_svc_classifier
parameter_dictionary['extraTree'] = parameter_extratrees_classifier
parameter_dictionary['RFCLF'] = parameter_randomforest_classifier
parameter_dictionary['LGCLF'] = parameter_logisticclassifier_classifier
parameter_dictionary['SVR'] = parameter_svr_regression
parameter_dictionary['RFREG'] = parameter_randomforest_regression
parameter_dictionary['XGBREGLINEAR'] = parameter_xgboost_linear_regression
parameter_dictionary['XGBREGLOGISTIC'] = parameter_xgboost_logistic_regression
parameter_dictionary['RIDGE'] = parameter_skl_ridge
parameter_dictionary['LASSO'] = parameter_skl_lasso
parameter_dictionary['EXTREEREG'] = parameter_extratree_reg
parameter_dictionary['ChainerNeuralNetworkRegression'] = parameter_chainer_regression