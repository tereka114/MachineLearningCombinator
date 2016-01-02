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
from ..utility.Util import model_select,model_prediction_by_problem
from ..utility.config import parameter_dictionary
from ..utility.file_util import dictionary_in_list_convert_to_csv,mkdir
from sklearn.grid_search import ParameterGrid

class Optimization(object):
    def __init__(self):
        pass

    def cross_validation_optimize(self, x, y,runs=1, kfolds=5, problem_type="regression"):
        kfold_list = []
        for run in xrange(runs):
            random_seed = 2015 + 1000 * (run + 1)

            kfold_indexs = None
            if problem_type == "regression":
                kfold_indexs = cross_validation.KFold(
                    n=len(x), n_folds=kfolds, shuffle=True, random_state=random_seed)
            elif problem_type == "classification" or problem_type == "binary_classification":
                kfold_indexs = cross_validation.StratifiedKFold(
                    y, n_folds=kfolds, shuffle=True, random_state=random_seed)
            kfold_list.append(kfold_indexs)
        return kfold_list

    def optimize(self, x=None, y=None, test_x=None,parameter=None, max_evals=10, runs=1, kfolds=5, 
        feature_name="feature_name", evaluate_function_name="rmsle", problem_type="regression", isWriteCsv=False, 
        isBagging=False, id_column_name="ids", ids=None, prediction_column_name="prediction", isOverWrite=False,label_convert_type="normal"):
        kf_list = self.cross_validation_optimize(
            x, y,runs, kfolds, problem_type)

        model = parameter['model']
        trials = Trials()

        self.trial_counter = 0

        mkdir("./log")
        mkdir("./result")

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
            x, y, test_x,parameter, runs, kfolds, kf_list,feature_name, evaluate_function_name, problem_type, id_column_name=id_column_name, ids=ids, prediction_column_name=prediction_column_name, isCVSave=False, saveCVName="CV",label_convert_type=label_convert_type)
        best_parameter = fmin(
            function, parameter, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        return best_parameter

    def hyperopt_optimization(self, x=None, y=None, test_x=None,parameter=None, runs=1, kfolds=5, kfold_list=None ,feature_name="feature_name", evaluate_function_name="evaluation", problem_type="regression", isBagging=False, id_column_name="ids", ids=None, prediction_column_name="prediction", isCVSave=False, saveCVName="CV",label_convert_type="normal"):
        # create csv log data
        parameter_log_data = [self.trial_counter]
        parameter_log_data.extend([parameter[parameter_column]
                                   for parameter_column in self.parameter_header])

        logging.info("{} time optimization start".format(
            str(self.trial_counter)))

        print parameter

        scores_mean, scores_valid = self.hyperopt_wrapper_function(
            x, y, test_x,parameter, runs, kfolds, kfold_list,feature_name, evaluate_function_name, problem_type, self.trial_counter, id_column_name=id_column_name, ids=ids, prediction_column_name=prediction_column_name, isCVSave=False, saveCVName="CV",label_convert_type=label_convert_type)

        parameter_log_data.append(scores_mean)
        parameter_log_data.append(scores_valid)

        self.writer.writerow(parameter_log_data)
        # count up
        self.trial_counter += 1

        return scores_mean

    def hyperopt_wrapper_function(self, x, y, test_x,parameter, runs, kfolds, kfold_list,feature_name,
                                  evaluate_function_name, problem_type, trial_counter, isWriteCsv=False, isBagging=False,
                                  id_column_name="ids", ids=None, prediction_column_name="prediction", isCVSave=False, saveCVName="CV", label_convert_type="normal"):
        """
        hyper_opt_optimize
        """
        model_name = parameter["model"]
        scores = np.zeros((runs, kfolds))

        for run in xrange(runs):
            for kfold,(train_index, valid_index) in enumerate(kfold_list[run]):
                logging.info("runs:{} kfold:{} model:{} train index length:{} test index length:{}".format(
                    str(run), str(kfold), model_name, len(train_index), len(valid_index)))

                save_path = "./{}/Runs{}/KFold{}".format(feature_name,run,kfold)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                kfold_train_x, kfold_train_y = x[train_index], y[train_index]
                kfold_valid_x, kfold_valid_y = x[valid_index], y[valid_index]

                kfold_train_y = self.labeled_preprocess(kfold_train_y, label_convert_type)
                score = None

                clf = model_select(parameter)
                clf.fit(kfold_train_x, self.labeled_preprocess(kfold_train_y,label_convert_type))

                valid_predict = model_prediction_by_problem(clf,kfold_valid_x,model_name,problem_type)
                valid_predict = self.labeled_after_process(valid_predict, label_convert_type)
                # evaluation
                score = evaluate_function(kfold_valid_y,valid_predict,evaluate_function_name)

                scores[run][kfold] = score
                print "score:{}".format(score)
                logging.info("score:{}".format(score))
                kfold += 1

                if problem_type == "regression" or problem_type == "binary_classification":
                    prediction = clf.predict(test_x)
                    prediction = self.labeled_after_process(prediction, label_convert_type)
                    output = pd.DataFrame(
                        {id_column_name: valid_index, prediction_column_name: valid_predict})
                    valid_csv = "{}/{}_valid_{}.csv".format(save_path,model_name,trial_counter)
                    output.to_csv(valid_csv, index=False)
                    #output.to_csv(submission_file_path, index=False)

        scores_mean = np.mean(scores)
        scores_valid = np.std(scores)

        logging.info("finish this cv mean_score:{}   std_score:{}".format(
            str(scores_mean), str(scores_valid)))
        print "mean_score:{}   std_score:{}".format(str(scores_mean), str(scores_valid))
        logging.info("start all data prediction")

        clf = model_select(parameter)
        y = self.labeled_preprocess(y, label_convert_type)
        clf.fit(x, y)

        single_prediction_savedir = "./submission/{}/{}".format(feature_name,model_name)
        if not os.path.exists(single_prediction_savedir):
            os.makedirs(single_prediction_savedir)

        submission_file = "submission_{}.csv".format(str(trial_counter))

        submission_file_path = os.path.join(
            single_prediction_savedir, submission_file)

        prediction = None

        if problem_type == "regression" or problem_type == "binary_classification":
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


    def labeled_preprocess(self, y, convert_type="normal"):
        if convert_type == "log":
            print "log"
            return np.log1p(y)
        return y

    def labeled_after_process(self, y, convert_type="normal"):
        if convert_type == "log":
            return np.expm1(y)
        return y

class OptimizeCrossValidation(object):
    def __init__(self):
        pass

    def optimize(self,x=None,y=None,x_test=None,kfolds=None,base_parameter=None,parameter_grid_dict=None,
        times=None,evaluate_function_name=None,problem_type=None,output_files=None,bagging_times=None,response_function=None):
        scores = []
        kfold = cross_validation.KFold(
            n=len(x), n_folds=kfolds, shuffle=True, random_state=71)

        for params in ParameterGrid(parameter_grid_dict):
            base_parameter['seed'] = 71
            model_name = base_parameter['model']
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

                    train_prediction = model_prediction_by_problem(clf,x_train,model_name,problem_type)
                    valid_prediction = model_prediction_by_problem(clf,x_valid,model_name,problem_type)

                    if not response_function == None:
                        train_prediction = response_function(train_prediction)
                        valid_prediction = response_function(valid_prediction)

                    train_score = evaluate_function(y_train, train_prediction,evaluate_function_name)
                    test_score = evaluate_function(y_valid, valid_prediction,evaluate_function_name)

                    total_train_evaluation += train_score
                    total_test_evaluation += test_score
                    train_score_avg = total_train_evaluation / (time + 1)
                    test_score_avg = total_test_evaluation / (time + 1)

                    score_dict = {
                        "data index":index,
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
            base_parameter['seed'] = 71

            if not x_test == None: 
                bagging_result = np.zeros((bagging_times,len(y_train)))
                for time in xrange(bagging_times):
                    clf = model_select(base_parameter)
                    clf.fit(x,y_train)
                    test_prediction = model_prediction_by_problem(clf,x_test,model_name,problem_type)

                    if not response_function == None:
                        test_prediction = response_function(train_prediction)
                    bagging_result[time] = test_prediction
                pd.DataFrame()

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
                    elif problem_type == "regression":
                        train_prediction = clf.predict(x_train,parameter_epoch)
                        valid_prediction = clf.predict(x_valid,parameter_epoch)
                    else:
                        if base_parameter['model'] == "XGBREGLOGISTIC":
                            train_prediction = self.clf.predict_proba(x_train,parameter_epoch)
                            valid_prediction = self.clf.predict_proba(x_valid,parameter_epoch)
                        else:
                            train_prediction = self.clf.predict_proba(x_train,parameter_epoch)[:, 1]
                            valid_prediction = self.clf.predict_proba(x_valid,parameter_epoch)[:, 1]                     
                    train_score = evaluate_function(y_train, train_prediction,evaluate_function_name)
                    test_score = evaluate_function(y_valid, valid_prediction,evaluate_function_name)

                    score_dict = {"data_index":index,"train_score":train_score,"test_score":test_score,"stop epochs":parameter_epoch}
                    for key,value in base_parameter.items():
                        score_dict[key] = value
                    scores_list.append(score_dict)
        df = pd.DataFrame(scores_list)
        # if not os.path.exists(output_files):
        #     os.makedirs(output_files)
        df.to_csv(output_files)