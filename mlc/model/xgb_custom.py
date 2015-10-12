from ..utility.evaluation_functions import *

def rmspe_evaluation(preds,dtrain):
	y_true = dtrain.get_label()
	exp_y_true = np.expm1(y_true)
	exp_y_preds = np.expm1(preds)
	return "error-RMSPE",evaluate_function(exp_y_true,exp_y_preds,"rmspe")

def select_evaluation(function_name):
	if function_name == "rmspe":
		return rmspe_evaluation