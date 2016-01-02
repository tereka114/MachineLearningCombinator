import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_curve, auc, roc_auc_score
"""
evaluate function for created models
"""


def weighted_kappa():
    pass


def calc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def logloss(y_true, y_pred):
    return log_loss(y_true, y_pred)


def mean_squared_error_func(y_true, y_pred):
    """
    calculate mean squared error

    :params y_true: true_data as np.array
    :params y_pred: true_data as np.array
    """
    return mean_squared_error(y_true, y_pred)

def reast_mean_squared_error_func(y_true, y_pred):
    """
    calculate mean squared error

    :params y_true: true_data as np.array
    :params y_pred: true_data as np.array
    """
    return mean_squared_error(y_true, y_pred) ** 0.5

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1. / (y[ind]**2)
    return w


def ToZero(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y > 0
    w[ind] = y[ind]
    return w


def rmspe(y, yhat):
    w = ToWeight(y)
    yhat = ToZero(yhat)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return rmspe

# def rmspe(y_true,y_pred):
#     """
#     calculate (rmspe)
#     """
#     #print y_true
#     #print y_pred
#     x1 = y_true + 0.000001
#     x2 = y_true - y_pred
#     x3 = x2 / x1
#     #print x3
#     x4 = x3 * x3
#     #print x4
#     x5 = np.mean(x4)
#     #print x5
#     return np.sqrt(x5)


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
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array
    :param data: array-like
    :param window_size: size
    :return: the sequence of windows
    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending
    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    target_distribution = np.arange(
        1, total_events + 1, dtype='float') / total_events
    subarray_distribution = np.cumsum(np.bincount(
        subindices, minlength=total_events), dtype='float')
    subarray_distribution /= 1.0 * subarray_distribution[-1]
    return np.mean((target_distribution - subarray_distribution) ** 2)


def compute_cvm(predictions, masses, n_neighbours=200, step=50):
    """
    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.
    :param predictions: array-like, predictions
    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
    :param n_neighbours: count of neighbours for event to define mass bin
    :param step: step through sorted mass-array to define next center of bin
    :return: average cvm value
    """
    predictions = np.array(predictions)
    masses = np.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[np.argsort(masses)]

    # Second, replace probabilities with order of probability among other
    # events
    predictions = np.argsort(np.argsort(
        predictions, kind='mergesort'), kind='mergesort')

    # Now, each window forms a group, and we can compute contribution of each
    # group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return np.mean(cvms)


def __roc_curve_splitted(data_zero, data_one, sample_weights_zero, sample_weights_one):
    """
    Compute roc curve
    :param data_zero: 0-labeled data
    :param data_one:  1-labeled data
    :param sample_weights_zero: weights for 0-labeled data
    :param sample_weights_one:  weights for 1-labeled data
    :return: roc curve
    """
    labels = [0] * len(data_zero) + [1] * len(data_one)
    weights = np.concatenate([sample_weights_zero, sample_weights_one])
    data_all = np.concatenate([data_zero, data_one])
    fpr, tpr, _ = roc_curve(labels, data_all, sample_weight=weights)
    return fpr, tpr


def compute_ks(data_prediction, mc_prediction, weights_data, weights_mc):
    """
    Compute Kolmogorov-Smirnov (ks) distance between real data predictions cdf and Monte Carlo one.
    :param data_prediction: array-like, real data predictions
    :param mc_prediction: array-like, Monte Carlo data predictions
    :param weights_data: array-like, real data weights
    :param weights_mc: array-like, Monte Carlo weights
    :return: ks value
    """
    assert len(data_prediction) == len(
        weights_data), 'Data length and weight one must be the same'
    assert len(mc_prediction) == len(
        weights_mc), 'Data length and weight one must be the same'

    data_prediction, mc_prediction = np.array(
        data_prediction), np.array(mc_prediction)
    weights_data, weights_mc = np.array(weights_data), np.array(weights_mc)

    assert np.all(data_prediction >= 0.) and np.all(
        data_prediction <= 1.), 'Data predictions are out of range [0, 1]'
    assert np.all(mc_prediction >= 0.) and np.all(
        mc_prediction <= 1.), 'MC predictions are out of range [0, 1]'

    weights_data /= np.sum(weights_data)
    weights_mc /= np.sum(weights_mc)

    fpr, tpr = __roc_curve_splitted(
        data_prediction, mc_prediction, weights_data, weights_mc)

    Dnm = np.max(np.abs(fpr - tpr))
    return Dnm


def roc_auc_truncated(labels, predictions, tpr_thresholds=(0.2, 0.4, 0.6, 0.8),
                      roc_weights=(4, 3, 2, 1, 0)):
    """
    Compute weighted area under ROC curve.
    :param labels: array-like, true labels
    :param predictions: array-like, predictions
    :param tpr_thresholds: array-like, true positive rate thresholds delimiting the ROC segments
    :param roc_weights: array-like, weights for true positive rate segments
    :return: weighted AUC
    """
    assert np.all(predictions >= 0.) and np.all(
        predictions <= 1.), 'Data predictions are out of range [0, 1]'
    assert len(tpr_thresholds) + \
        1 == len(roc_weights), 'Incompatible lengths of thresholds and weights'
    fpr, tpr, _ = roc_curve(labels, predictions)
    area = 0.
    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
    for index in range(1, len(tpr_thresholds)):
        tpr_cut = np.minimum(tpr, tpr_thresholds[index])
        tpr_previous = np.minimum(tpr, tpr_thresholds[index - 1])
        area += roc_weights[index - 1] * \
            (auc(fpr, tpr_cut, reorder=True) -
             auc(fpr, tpr_previous, reorder=True))
    tpr_thresholds = np.array(tpr_thresholds)
    # roc auc normalization to be 1 for an ideal classifier
    area /= np.sum((tpr_thresholds[1:] -
                    tpr_thresholds[:-1]) * np.array(roc_weights))
    return area


def evaluate_function(y_true, y_pred, eval_func):
    if eval_func == "accuracy":
        return accuracy(y_true, y_pred)
    elif eval_func == "logloss":
        return logloss(y_true, y_pred)
    elif eval_func == "mean_squared_error":
        return mean_squared_error_func(y_true, y_pred)
    elif eval_func == "rmse":
        return reast_mean_squared_error_func(y_true, y_pred)
    elif eval_func == "gini":
        return gini_normalized(y_true, y_pred)
    elif eval_func == "rmsle":
        return rmsle(y_true, y_pred)
    elif eval_func == "area_auc":
        return roc_auc_truncated(y_true, y_pred)
    elif eval_func == "auc":
        return calc_auc(y_true, y_pred)
    elif eval_func == "rmspe":
        return rmspe(y_true, y_pred)

        
