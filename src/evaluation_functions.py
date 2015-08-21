import numpy as np
from sklearn.metrics import accuracy_score,log_loss,mean_squared_error

"""
evaluate function for created models
"""
def weighted_kappa():
	pass

def accuracy(y_true,y_pred):
	return accuracy_score(y_true,y_pred)

def logloss(y_true,y_pred):
	return log_loss(y_true, y_pred)

def mean_squared_error_func(y_true,y_pred):
	return mean_squared_error(y_true, y_pred)

def Gini(expected, predicted):
    assert expected.shape[0] == predicted.shape[0], 'unequal number of rows'

    _all = np.asarray(np.c_[
        expected,
        predicted,
        np.arange(expected.shape[0])], dtype=np.float)

    _EXPECTED = 0
    _PREDICTED = 1
    _INDEX = 2

    # sort by predicted descending, then by index ascending
    sort_order = np.lexsort((_all[:, _INDEX], -1 * _all[:, _PREDICTED]))
    _all = _all[sort_order]

    total_losses = _all[:, _EXPECTED].sum()
    gini_sum = _all[:, _EXPECTED].cumsum().sum() / total_losses
    gini_sum -= (expected.shape[0] + 1.0) / 2.0
    return gini_sum / expected.shape[0]

def gini_normalized(expected, predicted):
    return Gini(expected, predicted) / Gini(expected, expected)

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

def evaluate_function(y_true,y_pred,eval_func):
    if eval_func == "accuracy":
		return accuracy(y_pred, y_true)
    elif eval_func == "logloss":
		return logloss(y_true, y_pred)
    elif eval_func == "mean_squared_error":
		return mean_squared_error_func(y_true, y_pred)
    elif eval_func == "gini":
		return gini_normalized(y_true, y_pred)
    elif eval_func == "rmsle":
        return rmsle(y_true,y_pred)