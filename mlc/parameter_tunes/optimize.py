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
from sklearn.grid_search import ParameterGrid

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

        print parameter

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
            print "log"
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

class OptimizeCrossValidation(object):
    def __init__(self):
        pass

    def optimize(self,x,y,kfolds,base_parameter,parameter_grid_dict,times,evaluate_function_name,problem_type,output_files):
        scores = []
        kfold = cross_validation.KFold(
            n=len(x), n_folds=kfolds, shuffle=True, random_state=71)

        for params in ParameterGrid(parameter_grid_dict):

            base_parameter['seed'] = 71
            for key,value in params.items():
                base_parameter[key] = value

            total_train_evaluation = 0
            total_test_evaluation = 0

            count = 1
            for time in xrange(times):
                for index,(train_index,test_index) in enumerate(kfold):
                    x_train,x_valid = x[train_index],x[test_index]
                    y_train,y_valid = y[train_index],y[test_index]

                    clf = model_select(base_parameter)
                    clf.fit(x_train,y_train)

                    train_prediction,valid_prediction = None,None

                    if problem_type == "classification":
                        train_prediction = clf.predict_proba(x_train)
                        valid_prediction = clf.predict_proba(x_valid)
                    else:
                        train_prediction = clf.predict(x_train)
                        valid_prediction = clf.predict(x_valid)

                    train_score = evaluate_function(y_train, train_prediction,evaluate_function_name)
                    test_score = evaluate_function(y_valid, valid_prediction,evaluate_function_name)

                    total_train_evaluation += train_score
                    total_test_evaluation += test_score
                    train_score_avg = total_train_evaluation / (time + 1)
                    test_score_avg = total_test_evaluation / (time + 1)

                    score_dict = {"data index":index,
                        "train score":train_score,
                        "test score":test_score,
                        "train score average":train_score_avg,
                        "test score average":test_score_avg
                    }

                    for key,value in base_parameter.items():
                        score_dict[key] = value
                    scores.append(score_dict)

                    count += 1

                    base_parameter['seed'] = base_parameter['seed'] + 100
        df = pd.DataFrame(scores)
        df.to_csv(output_files)

class OptimizeEpochsValidation(object):
    def __init__(self):
        pass

    def optimize_epochs(self,x,y,base_parameter,kfolds,epochs,evaluate_function_name,problem_type="classification",stopping_epochs=[],output_files=""):
        kfold = cross_validation.KFold(
            n=len(x), n_folds=kfolds, shuffle=True, random_state=71)

        scores_list = []

        for index,(train_index,test_index) in enumerate(kfold):
            x_train,x_valid = x[train_index],x[test_index]
            y_train,y_valid = y[train_index],y[test_index]

            #if you select xgboost model
            clf = model_select(base_parameter)
            if base_parameter['model'].startswith("XG"):
                clf.fit(x_train,y_train)
                for parameter_epoch in stopping_epochs:
                    train_prediction,valid_prediction = None,None
                    if problem_type == "classification":
                        train_prediction = clf.predict_proba(x_train,parameter_epoch)
                        valid_prediction = clf.predict_proba(x_valid,parameter_epoch)
                    else:
                        train_prediction = clf.predict(x_train,parameter_epoch)
                        valid_prediction = clf.predict(x_valid,parameter_epoch)                        
                    train_score = evaluate_function(y_train, train_prediction,evaluate_function_name)
                    test_score = evaluate_function(y_valid, valid_prediction,evaluate_function_name)

                    score_dict = {"data_index":index,"train_score":train_score,"test_score":test_score,"stop epochs":parameter_epoch}
                    for key,value in base_parameter.items():
                        score_dict[key] = value
                    scores_list.append(score_dict)
            elif base_parameter['model'].startswith('Neural'):

                pass
        df = pd.DataFrame(scores_list)
        # if not os.path.exists(output_files):
        #     os.makedirs(output_files)
        df.to_csv(output_files)