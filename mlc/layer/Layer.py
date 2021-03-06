# coding:utf-8
from ..utility.Util import model_select, genIndexKFold, genIndexStratifiedKFold
import numpy as np
from ..utility.config import parameter_dictionary
from hyperopt import hp, fmin, tpe
from sklearn.svm import SVR
from ..utility.evaluation_functions import evaluate_function
from scipy.optimize import minimize
import logging

# logger = logging.getLogger("applog")
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M',
#                     filename='./tmp/app.log',
#                     filemode='a')

# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# logging.getLogger('').addHandler(console)


class Layer(object):

    def __init__(self):
        pass

    def fit(self, parameter, train_x, train_y):
        clf = model_select(parameter)
        clf.fit(train_x, train_y)
        self.clf = clf

    def predict(self, train_x, train_y, test_x, parameter):
        self.fit(parameter, train_x, train_y)
        return self.clf.predict(train_x), self.clf.predict(test_x)

    def predict_proba(self, train_x, train_y, test_x, parameter):
        self.fit(parameter, train_x, train_y)
        return self.clf.predict_proba(train_x), self.clf.predict_proba(test_x)


class RegressionLayer(Layer):

    def __init__(self):
        pass

    def predict(self, train_x, train_y, test_x, parameter):
        # self.fit(parameter,train_x,train_y)
        # return self.clf.predict(train_x),self.clf.predict(test_x)
        self.fit(parameter, train_x, np.log1p(train_y))
        train_predict = np.expm1(self.clf.predict(train_x))
        test_predict = np.expm1(self.clf.predict(test_x))
        return train_predict, test_predict


class ClassificationLayer(Layer):

    def __init__(self):
        pass


class ClassificationBinaryLayer(Layer):

    def __init__(self):
        pass

    def predict_proba(self, train_x, train_y, test_x, parameter):
        self.fit(parameter, train_x, train_y)
        if parameter['model'] == "XGBREGLOGISTIC":
            train_predict = self.clf.predict_proba(train_x)
            test_predict = self.clf.predict_proba(test_x)
        else:
            train_predict = self.clf.predict_proba(train_x)[:, 1]
            test_predict = self.clf.predict_proba(test_x)[:, 1]
        return train_predict, test_predict


class BaggingLayer(Layer):

    def __init__(self):
        self.model_parameter_list = []


class RegressionBaggingLayer(BaggingLayer):

    def __init__(self):
        pass

    def predict(self, train_x, train_y, test_x, parameter, times=1, validation_indexs=None, type='regression'):
        print parameter['model'] + " predict staring"

        train_preds = np.zeros((times, len(train_x)))
        test_preds = np.zeros((times, len(test_x)))
        for time in xrange(times):
            logging.info("time {}".format(str(time)))
            validation_indexs = genIndexKFold(train_x, 5)
            test_pred = np.zeros((len(validation_indexs), len(test_x)))
            train_pred = np.zeros((len(train_x)))

            for i, (train_ind, test_ind) in enumerate(validation_indexs):
                clf = model_select(parameter)
                logging.info("start time:{} Fold:{}".format(str(time), str(i)))
                print "start time:{} Fold:{}".format(str(time), str(i))
                X_train = train_x[train_ind]
                Y_train = np.log1p(train_y[train_ind])
                X_test = train_x[test_ind]
                Y_test = train_y[test_ind]

                clf.fit(X_train, Y_train)
                test_pred[i][:] = np.expm1(clf.predict(test_x))
                train_pred[test_ind] = np.expm1(clf.predict(X_test))
                evaluation = evaluate_function(
                    Y_test, train_pred[test_ind], 'rmsle')
                logging.info("time:{} Fold:{} evaluation:{}".format(
                    str(time), str(i), str(evaluation)))
            train_preds[time] = train_pred
            test_preds[time] = np.mean(test_pred, axis=0)
            print train_preds, test_preds

        return np.mean(train_preds, axis=0), np.mean(test_preds, axis=0)


class ClassificationMultiBaggingLayer(BaggingLayer):

    def __init__(self):
        pass

    def predict_proba(self, train_x, train_y, test_x, parameter, times=1):
        print parameter['model'] + " predict staring"
        clf_number = len(set(train_y))

        folding_indexs = 10
        train_pred_list = np.zeros((times, len(train_x), clf_number))
        test_pred = np.zeros((times * folding_indexs, len(test_x), clf_number))

        for i in xrange(times):
            validation_indexs = genIndexStratifiedKFold(
                train_y, folding_indexs)
            train_pred = np.zeros((len(train_x), clf_number))

            for j, (train_ind, test_ind) in enumerate(validation_indexs):
                clf = model_select(parameter)
                X_train = train_x[train_ind]
                Y_train = train_y[train_ind]
                X_test = train_x[test_ind]
                Y_test = train_y[test_ind]

                clf.fit(X_train, Y_train)
                test_pred[i * 10 + j] = clf.predict_proba(test_x)
                train_pred[test_ind] = clf.predict_proba(X_test)
            train_pred_list[i] = train_pred
        return np.mean(train_pred_list, axis=0), np.mean(test_pred, axis=0)


class ClassificationBinaryBaggingLayer(BaggingLayer):

    def __init__(self):
        pass

    def predict_proba(self, train_x, train_y, test_x, parameter, times=1):
        print parameter['model'] + " predict staring"

        folding_indexs = 5
        train_pred_list = np.zeros((times, len(train_x)))
        test_pred = np.zeros((times * folding_indexs, len(test_x)))

        for i in xrange(times):
            validation_indexs = genIndexStratifiedKFold(
                train_y, folding_indexs)
            print validation_indexs
            train_pred = np.zeros((len(train_x)))

            for j, (train_ind, test_ind) in enumerate(validation_indexs):
                clf = model_select(parameter)
                X_train = train_x[train_ind]
                Y_train = train_y[train_ind]
                X_test = train_x[test_ind]
                Y_test = train_y[test_ind]

                clf.fit(X_train, Y_train)
                if parameter['model'] == "XGBREGLOGISTIC":
                    test_pred[i * 10 + j] = clf.predict_proba(test_x)
                    train_pred[test_ind] = clf.predict_proba(X_test)
                else:
                    test_pred[i * 10 + j] = clf.predict_proba(test_x)[:, 1]
                    train_pred[test_ind] = clf.predict_proba(X_test)[:, 1]
                print evaluate_function(Y_test, train_pred[test_ind], 'auc')
            train_pred_list[i] = train_pred
            print train_pred_list[i]

        return np.mean(train_pred_list, axis=0), np.mean(test_pred, axis=0)

    def predict(self, train_x, train_y, test_x, parameter, times=5, validation_indexs=None, type='regression'):
        print parameter['model'] + " predict staring"

        train_preds = np.zeros((times, len(train_x)))
        test_preds = np.zeros((times, len(test_x)))
        for time in xrange(times):
            validation_indexs = genIndexKFold(train_x, 10)
            test_pred = np.zeros((len(validation_indexs), len(test_x)))
            train_pred = np.zeros((len(train_x)))

            for i, (train_ind, test_ind) in enumerate(validation_indexs):
                clf = model_select(parameter)
                print "Fold", i
                X_train = train_x[train_ind]
                Y_train = np.log1p(train_y[train_ind])
                X_test = train_x[test_ind]
                Y_test = train_y[test_ind]

                clf.fit(X_train, Y_train)
                test_pred[i][:] = np.expm1(clf.predict(test_x))
                train_pred[test_ind] = np.expm1(clf.predict(X_test))
                print evaluation_functions.evaluate_function(Y_test, train_pred[test_ind], 'rmsle')
            train_preds[time] = train_pred
            test_preds[time] = np.mean(test_pred, axis=0)

        return np.mean(train_preds, axis=0), np.mean(test_preds, axis=0)


class RegressionFramework(object):

    def __init__(self):
        self.layer = Layer()

    def setStackingParameter(self, stacking_list, training_x, training_y, loss_function='gini'):
        for i in xrange(len(stacking_list)):
            self.layer.setModel(time=1, model_name=stacking_list[
                                i], train_x=training_x, train_y=training_y, loss_function=loss_function)
            self.layer.dumpParameter()

    def startFrameWork(self, train_x, train_y, test_x, stacking_list, loss_function='gini'):
        self.setStackingParameter(
            stacking_list, train_x, train_y, loss_function=loss_function)
        training_pred_x, testing_pred_x = self.layer.predict_all(
            train_x, train_y, test_x, validation_indexs=None, type=loss_function)
        minimumLayer = MinimumRankingAverage()
        minimumLayer.setFit(training_pred_x, train_y, max_evals=200)
        return minimumLayer.predict(testing_pred_x)


class MinimumRankingAverage(object):

    def __init__(self):
        pass

    def fit(self, x_trains, y_train, max_evals=200, loss_function='gini'):
        weight_number = len(x_trains)
        parameter_dict = {}
        for i in xrange(weight_number):
            parameter_dict[i] = hp.quniform(str(i), 0.01, 1.0, 0.02)

        print "======================================================="
        function = lambda params: optimize_linear_weight(
            params, x_trains, y_train, loss_function)
        print parameter_dict, x_trains, y_train
        weight_params = fmin(function, parameter_dict,
                             algo=tpe.suggest, max_evals=max_evals)
        print "======================================================="
        self.weight_params = np.array(
            [weight_params[str(i)] for i in xrange(weight_number)])

    def predict(self, x_test):
        params = self.weight_params
        bagging_models = len(self.weight_params)
        print len(self.weight_params)
        sum_number = 0.0

        for index in xrange(len(self.weight_params)):
            sum_number = sum_number + params[index]
        weight_params = [params[index] /
                         sum_number for index in xrange(bagging_models)]

        result = np.zeros(x_test[0].shape)
        for index in xrange(bagging_models):
            print weight_params[index], x_test[index]
            result = result + weight_params[index] * x_test[index]
            print result
        return result


class AverageLayer(object):

    def __init__(self):
        pass

    def predict(self, x_test):
        result = np.zeros(x_test[0].shape)
        for index in xrange(len(x_test)):
            result = result + x_test[index]
        return result / len(x_test)


class Boosting(object):

    def __init__(self, params):
        self.clf = None

    def fit(self, x_trains, y_train):
        x_trains_transpose = x_trains.T
        self.clf.fit(x_trains_transpose, y_train)

    def predict(self, x_tests):
        return self.clf.predict(x_tests.T)


class SVRRegressorBoosting(Boosting):

    def __init__(self):
        self.clf = SVR()


class ManyRegressorBoosting(object):

    def __init__(self, params):
        self.clfs = []
        for param in params:
            clf = model_select(param)
        self.clfs.append(clf)

    def fit(self, x_trains, y_train):
        x_trains_transpose = x_trains.T
        for clf in self.clfs:
            clf.fit(x_trains_transpose, y_train)

    def predict(self, x_trains, y_train):
        x_trains_transpose = x_trains.T
        N = len(x_trains_transpose)
        prediction = np.zeros((N))
        for clf in self.clfs:
            prediction = prediction + clf.predict(x_trains_transpose)
        return prediction / len(self.clfs)


def blending_weight_optimize(predictions, labels, function_name):
    def loss_func(weights):
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction
        return evaluation_functions.evaluate_function(labels, final_prediction, 'rmsle')

    if function_name == "cobyla":
        starting_values = [1.0] * len(predictions)
        # cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        res = minimize(loss_func, starting_values, method='COBYLA')
    elif function_name == "slsqp":
        starting_values = [1.0] * len(predictions)
        cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        bounds = [(0, 1)] * len(predictions)
        res = minimize(loss_func, starting_values, method='SLSQP',
                       bounds=bounds, constraints=cons)
    elif function_name == "tnc":
        starting_values = [1.0] * len(predictions)
        res = optimize.minimize(
            loss_func, starting_values, method='TNC', tol=1e-10)
    elif function_name == "":
        pass
    elif function_name == "":
        pass
    print res['x']
    return res['fun'], res['x']


class SLSQPResolver(object):

    def __init__(self):
        pass

    def loss_func(self, weights):
        final_prediction = 0
        predictions = self.predictions
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction
        return evaluation_functions.evaluate_function(self.labels, final_prediction, 'rsmle')

    def predict(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels
        starting_values = [0.5] * len(predictions)
        cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        # our weights are bound between 0 and 1
        bounds = [(0, 1)] * len(predictions)
        pack = (starting_values, predictions, labels)
        res = minimize(self.loss_func, (starting_values, predictions,
                                        labels), method='SLSQP', bounds=bounds, constraints=cons)

        return res['fun']


class RegressionDecisionLayer(Layer):

    def __init__(self):
        pass

    def fit(self, training_x, training_y, stacking=True):
        pass
