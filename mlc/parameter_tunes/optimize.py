# coding:utf-8
import numpy as np
import time
import os

from hyperopt import hp, fmin, tpe, Trials
from sklearn import cross_validation
import pickle
import pandas as pd
import logging
import csv
import time
from ..utility.config import parameter_dictionary
from ..utility.evaluation_functions import evaluate_function
from ..utility.Util import model_select
from ..utility.config import parameter_dictionary
from ..utility.file_util import dictionary_in_list_convert_to_csv


def optimize_model_function_validation(params, x_train, y_train, x_test, y_test, evaluate_function_name):
    """
    optimize parameter
    train_test parameter turning

    :param params:
    :param x:
    :param y:
    ::
    :param evaluate_function_name:
    """
    print params
    cnt = 0

    clf = model_select(params)
    print len(y_train), len(y_test)

    train_y_pred = None
    y_pred = None
    score = 0.0

    # y_pred = clf.predict(x_test)
    if evaluate_function_name == "accuracy":
        clf.fit(x_train, y_train)
        train_y_pred = clf.predict(x_train)
        y_pred = clf.predict(x_test)
        train_score = evaluate_function(
            y_train, train_y_pred, evaluate_function_name)
        train_score = -train_score
        score = evaluate_function(y_test, y_pred, evaluate_function_name)
        score = -score
    elif evaluate_function_name == "logloss":
        clf.fit(x_train, y_train)
        train_y_pred = clf.predict(x_train)
        y_pred = clf.predict_proba(x_test)
        train_score = evaluate_function(
            y_train, train_y_pred, evaluate_function_name)
        score = evaluate_function(y_test, y_pred, evaluate_function_name)
        score = -score
        train_score = -train_score
    elif evaluate_function_name == "mean_squared_error":
        train_y_pred = clf.predict(x_train)
        y_pred = clf.predict(x_test)
        train_score = evaluate_function(
            y_train, train_y_pred, evaluate_function_name)
        score = evaluate_function(y_test, y_pred, evaluate_function_name)
    elif evaluate_function_name == "gini":
        y_pred = clf.predict(x_test)
        train_y_pred = clf.predict(x_train)
        train_score = evaluate_function(
            y_train, train_y_pred, evaluate_function_name)
        score = evaluate_function(y_test, y_pred, evaluate_function_name)
        score = -score
        train_score = -train_score
    elif evaluate_function_name == "rmsle":
        y_pred = np.expm1(clf.predict(x_test))
        train_y_pred = clf.predict(x_train)

        train_score = evaluate_function(
            y_train, train_y_pred, evaluate_function_name)
        score = evaluate_function(y_test, y_pred, evaluate_function_name)
    elif evaluate_function_name == "auc":
        if params['model'] == "XGBREGLOGISTIC":
            y_pred = clf.predict_proba(x_test)
        else:
            y_pred = clf.predict_proba(x_test)[:, 1]

        train_score = evaluate_function(
            y_train, train_y_pred, evaluate_function_name)
        score = evaluate_function(y_test, y_pred, evaluate_function_name)
        score = -score
        train_score = -train_score
    elif evaluate_function_name == "rmspe":
        clf.fit(x_train, np.log1p(y_train))
        train_y_pred = np.expm1(clf.predict(x_train))
        y_pred = np.expm1(clf.predict(x_test))

        train_score = evaluate_function(
            y_train, train_y_pred, evaluate_function_name)
        score = evaluate_function(y_test, y_pred, evaluate_function_name)
        score = score
    cnt = cnt + 1
    # print cnt,params['model'],score
    print train_score, score
    return score


def optimize_model_function(params, x, y, validation_indexs, evaluate_function_name):
    """
    optimize parameter
    k-fold classification

    :param params:
    :param x:
    :param y:
    :param validation_indexs:
    :param evaluate_function_name:
    """
    print params
    evals = np.zeros((len(validation_indexs)))
    cnt = 0
    for i, validation_index in enumerate(validation_indexs):
        score = optimize_model_function_validation(params, x[validation_index[0]], y[validation_index[0]], x[
                                                   validation_index[1]], y[validation_index[1]], evaluate_function_name)
        cnt = cnt + 1
        # print cnt,params['model'],score
        print score
        evals[i] = score

    evaluate_score = np.mean(evals)
    print "final result", evaluate_score
    return evaluate_score


def optimize_model_function_split(params, x_train, x_test, y_train, y_test, evaluate_function):
    print params
    evals = np.zeros((len(x_train)))
    cnt = 0
    for index in xrange(len(x_train)):
        score = optimize_model_function_validation(params, x_train[index], y_train[
                                                   index], x_test[index], y_test[index], evaluate_function)
        cnt = cnt + 1
        # print cnt,params['model'],score
        print score
        evals[index] = score

    evaluate_score = np.mean(evals)
    print "final result", evaluate_score
    return evaluate_score


def optimize_model_parameter_split(x, y, model_name=None, loss_function="accuracy", parameter=None, max_evals=100, n_folds=5, isWrite=True, times=1, problem_pattern="classification"):
    """
    hyperopt model turning
    """
    if model_name == None and parameter == None:
        print "you must set parameter or model_name"
        return None
    elif parameter != None:
        param = parameter
    elif model_name != None:
        param = parameter_dictionary[model_name]
    else:
        return None

    x_trains = []
    x_tests = []
    y_trains = []
    y_tests = []

    for time in xrange(times):
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
            x, y, test_size=0.0125)
        x_trains.append(x_train)
        x_tests.append(x_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    trials = Trials()
    function = lambda param: optimize_model_function_split(
        param, x_trains, x_tests, y_trains, y_tests, loss_function)
    print param
    print "========================================================================"
    best_param = fmin(function, param,
                      algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print "========================================================================"
    print "write result to csv files"

    # write the csv file
    if isWrite:
        datas = []
        for trial_data in trials.trials:
            print trial_data
            trial_parameter_dictionary = {}
            trial_parameter_dictionary['model'] = model_name
            trial_parameter_dictionary['tid'] = trial_data['misc']['tid']
            for key, value in trial_data['misc']['vals'].items():
                print key, value[0]
                trial_parameter_dictionary[key] = value[0]
            trial_parameter_dictionary['loss'] = trial_data['result']['loss']
            trial_parameter_dictionary[
                'status'] = trial_data['result']['status']
            datas.append(trial_parameter_dictionary)
        filename = str(time.time()) + ".csv"
        dictionary_in_list_convert_to_csv(datas, filename)

    print trials.statuses()
    return best_param


def optimize_model_parameter_validation(x, y, model_name=None, loss_function="accuracy", parameter=None, max_evals=100, n_folds=5, isWrite=True, problem_pattern="classification"):
    """
    hyperopt model turning
    """
    if model_name == None and parameter == None:
        print "you must set parameter or model_name"
        return None
    elif parameter != None:
        param = parameter
    elif model_name != None:
        param = parameter_dictionary[model_name]
    else:
        return None

    validation_indexs = []

    if problem_pattern == "classification":
        for train_index, test_index in cross_validation.StratifiedKFold(y, n_folds=n_folds):
            validation_indexs.append((train_index, test_index))
    else:
        for train_index, test_index in cross_validation.KFold(len(y), n_folds=n_folds):
            validation_indexs.append((train_index, test_index))

    trials = Trials()
    function = lambda param: optimize_model_function(
        param, x, y, validation_indexs, loss_function)
    print param
    print "========================================================================"
    best_param = fmin(function, param,
                      algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print "========================================================================"
    print "write result to csv files"

    # write the csv file
    if isWrite:
        datas = []
        for trial_data in trials.trials:
            print trial_data
            trial_parameter_dictionary = {}
            trial_parameter_dictionary['model'] = model_name
            trial_parameter_dictionary['tid'] = trial_data['misc']['tid']
            for key, value in trial_data['misc']['vals'].items():
                print key, value[0]
                trial_parameter_dictionary[key] = value[0]
            trial_parameter_dictionary['loss'] = trial_data['result']['loss']
            trial_parameter_dictionary[
                'status'] = trial_data['result']['status']
            datas.append(trial_parameter_dictionary)
        filename = str(time.time()) + ".csv"
        dictionary_in_list_convert_to_csv(datas, filename)

    print trials.statuses()
    return best_param

    def model_evaluation(clf, x, y, evaluate_function_name, labeled_type, label_convert_type="normal"):
        if evaluate_function_name == "accuracy":
            y_pred = clf.predict(x)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
            score = -score
        elif evaluate_function_name == "logloss":
            y_pred = clf.predict(x)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
            train_score = -score
        elif evaluate_function_name == "mean_squared_error":
            y_pred = clf.predict(x)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
        elif evaluate_function_name == "gini":
            y_pred = clf.predict(x)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
            score = -score
            train_score = -train_score
        elif evaluate_function_name == "rmsle":
            y_pred = clf.predict(x)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
        elif evaluate_function_name == "auc":
            if params['model'] == "XGBREGLOGISTIC":
                y_pred = clf.predict_proba(x_test)
            else:
                y_pred = clf.predict_proba(x_test)[:, 1]

            train_score = evaluate_function(
                y_train, train_y_pred, evaluate_function_name)
            score = evaluate_function(y_test, y_pred, evaluate_function_name)
            score = -score
            train_score = -train_score
        elif evaluate_function_name == "rmspe":
            y_pred = clf.predict(x)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
            score = score
        return score


class Optimization(object):

    def __init__(self):
        pass

    def cross_validation_optimize(self, x, y, feature_name, runs=1, kfolds=5, problem_type="regression", isOverWrite=False):
        for run in xrange(runs):
            random_seed = 2015 + 1000 * (run + 1)

            kfold_indexs = None
            if problem_type == "regression":
                kfold_indexs = cross_validation.KFold(
                    n=len(x), n_folds=kfolds, shuffle=True, random_state=random_seed)
            elif problem_type == "classification" or problem_type == "binary_classification":
                kfold_indexs = cross_validation.StratifiedKFold(
                    labels=y, n_folds=kfolds, shuffle=True, random_state=random_seed)

            if not os.path.exists("./CrossValidationIndexs"):
                os.makedirs("./CrossValidationIndexs")

            data_path = "./CrossValidationIndexs/Runs{}.pkl".format(run)

            if isOverWrite or not os.path.exists(data_path):
                logging.info("create kfold data")
                pickle.dump(kfold_indexs, open(data_path, 'w'))
            else:
                logging.info(
                    "this data exist in these pathes. Don't need creation")

    def optimize(self, x=None, y=None, test_x=None,parameter=None, max_evals=10, runs=1, kfolds=5, 
        feature_name="feature_name", evaluate_function_name="rmsle", problem_type="regression", isWriteCsv=False, 
        isBagging=False, id_column_name="ids", ids=None, prediction_column_name="prediction", isOverWrite=False,label_convert_type="normal"):
        self.cross_validation_optimize(
            x, y, feature_name, runs, kfolds, problem_type, isOverWrite)

        model = parameter['model']
        trials = Trials()

        self.trial_counter = 0
        if not os.path.exists("./log"):
            os.makedirs("./log")

        if not os.path.exists("./result"):
            os.makedirs("./result")

        self.result_files = "result/result_{}_{}.csv".format(
            feature_name, model)
        self.log_file_path = "log/log_{:0f}.log".format(time.time())
        print self.log_file_path

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=self.log_file_path,
                            filemode="w")

        writer = open(self.result_files, "wb")
        writer = csv.writer(writer)

        # create header
        header = ["trial"]
        parameter_header = [k for k, v in parameter.items()]
        header += parameter_header
        header += ["trial_mean", "trial_std"]

        writer.writerow(header)
        self.writer = writer
        self.parameter_header = parameter_header
        self.header = header

        # execute hyperopt function for optimization parameters
        function = lambda parameter: self.hyperopt_optimization(
            x, y, test_x,parameter, runs, kfolds, feature_name, evaluate_function_name, problem_type, id_column_name=id_column_name, ids=ids, prediction_column_name=prediction_column_name, isCVSave=False, saveCVName="CV",label_convert_type=label_convert_type)
        best_parameter = fmin(
            function, parameter, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        return best_parameter

    def hyperopt_optimization(self, x=None, y=None, test_x=None,parameter=None, runs=1, kfolds=5, feature_name="feature_name", evaluate_function_name="evaluation", problem_type="regression", isBagging=False, id_column_name="ids", ids=None, prediction_column_name="prediction", isCVSave=False, saveCVName="CV",label_convert_type="normal"):
        # create csv log data
        parameter_log_data = [self.trial_counter]
        parameter_log_data.extend([parameter[parameter_column]
                                   for parameter_column in self.parameter_header])

        logging.info("{} time optimization start".format(
            str(self.trial_counter)))

        scores_mean, scores_valid = self.hyperopt_wrapper_function(
            x, y, test_x,parameter, runs, kfolds, feature_name, evaluate_function_name, problem_type, self.trial_counter, id_column_name=id_column_name, ids=ids, prediction_column_name=prediction_column_name, isCVSave=False, saveCVName="CV",label_convert_type=label_convert_type)

        parameter_log_data.append(scores_mean)
        parameter_log_data.append(scores_valid)

        self.writer.writerow(parameter_log_data)
        # count up
        self.trial_counter += 1

        return scores_mean

    def labeled_preprocess(self, y, convert_type="normal"):
        if convert_type == "log":
            return np.log1p(y)
        return y

    def labeled_after_process(self, y, convert_type="normal"):
        if convert_type == "log":
            return np.expm1(y)
        return y

    def model_evaluation(self, clf, x, y, evaluate_function_name, label_convert_type="normal"):
        if evaluate_function_name == "accuracy":
            y_pred = clf.predict(x)
            y_pred = self.labeled_after_process(y_pred, label_convert_type)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
            score = -score
        elif evaluate_function_name == "logloss":
            y_pred = clf.predict(x)
            y_pred = self.labeled_after_process(y_pred, label_convert_type)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
            train_score = -score
        elif evaluate_function_name == "mean_squared_error":
            y_pred = clf.predict(x)
            y_pred = self.labeled_after_process(y_pred, label_convert_type)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
        elif evaluate_function_name == "rmse":
            y_pred = clf.predict(x)
            y_pred = self.labeled_after_process(y_pred, label_convert_type)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
        elif evaluate_function_name == "gini":
            y_pred = clf.predict(x)
            y_pred = self.labeled_after_process(y_pred, label_convert_type)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
            score = -score
            train_score = -train_score
        elif evaluate_function_name == "rmsle":
            y_pred = clf.predict(x)
            y_pred = self.labeled_after_process(y_pred, label_convert_type)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
        elif evaluate_function_name == "auc":
            if params['model'] == "XGBREGLOGISTIC":
                y_pred = clf.predict_proba(x_test)
            else:
                y_pred = clf.predict_proba(x_test)[:, 1]

            train_score = evaluate_function(
                y_train, train_y_pred, evaluate_function_name)
            score = evaluate_function(y_test, y_pred, evaluate_function_name)
            score = -score
            train_score = -train_score
        elif evaluate_function_name == "rmspe":
            y_pred = clf.predict(x)
            y_pred = self.labeled_after_process(y_pred, label_convert_type)
            score = evaluate_function(
                y, y_pred, evaluate_function_name)
            score = score
        return score

    def hyperopt_wrapper_function(self, x, y, test_x,parameter, runs, kfolds, feature_name,
                                  evaluate_function_name, problem_type, trial_counter, isWriteCsv=False, isBagging=False,
                                  id_column_name="ids", ids=None, prediction_column_name="prediction", isCVSave=False, saveCVName="CV", label_convert_type="normal"):
        """
        hyper_opt_optimize
        """
        model_name = parameter["model"]
        scores = np.zeros((runs, kfolds))

        # check
        # if x == None or y == None:
        #     logging.info("you should set the train vector and test vector")
        #     return None

        # calculate score
        for run in xrange(runs):
            data_path = "./CrossValidationIndexs/Runs{}.pkl".format(
                run, feature_name)
            kfolds_index = pickle.load(open(data_path))
            kfold = 0
            for train_index, valid_index in kfolds_index:
                logging.info("runs:{} kfold:{} model:{} train index length:{} test index length:{}".format(
                    str(run), str(kfold), model_name, len(train_index), len(valid_index)))

                kfold_train_x, kfold_train_y = x[train_index], y[train_index]
                kfold_valid_x, kfold_valid_y = x[valid_index], y[valid_index]

                kfold_train_y = self.labeled_preprocess(kfold_train_y, label_convert_type)
                score = None

                if not isBagging:
                    clf = model_select(parameter)
                    clf.fit(kfold_train_x, kfold_train_y)
                    # evaluation
                    score = self.model_evaluation(
                        clf, kfold_valid_x, kfold_valid_y, evaluate_function_name,label_convert_type)
                    if isCVSave:
                        pass
                elif isBagging:
                    """
                    Todo: you should implement bagging system
                    """
                    pass

                scores[run][kfold] = score
                print "score:{}".format(score)
                logging.info("score:{}".format(score))
                kfold += 1

        scores_mean = np.mean(scores)
        scores_valid = np.std(scores)

        logging.info("finish this cv mean_score:{}   std_score:{}".format(
            str(scores_mean), str(scores_valid)))
        print "mean_score:{}   std_score:{}".format(str(scores_mean), str(scores_valid))
        logging.info("start all data prediction")

        """
        if you want to write the parameter.
        you should 
        """

        clf = model_select(parameter)
        y = self.labeled_preprocess(y, label_convert_type)
        clf.fit(x, y)

        single_prediction_savedir = "./submission"
        if not os.path.exists(single_prediction_savedir):
            os.makedirs(single_prediction_savedir)

        submission_file = "submission_{}_{}_{}.csv".format(
            feature_name, model_name, str(trial_counter))

        submission_file_path = os.path.join(
            single_prediction_savedir, submission_file)

        prediction = None

        if problem_type == "regression":
            prediction = clf.predict(test_x)
            prediction = self.labeled_after_process(prediction, label_convert_type)
            output = pd.DataFrame(
                {id_column_name: ids, prediction_column_name: prediction})
            output.to_csv(submission_file_path, index=False)

        elif problem_type == "classification":
            prediction = clf.predict_proba(x)

        logging.info("{} time optimizer finish all data prediction".format(
            str(self.trial_counter)))

        return scores_mean, scores_valid
