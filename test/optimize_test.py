import unittest
from sklearn.datasets import load_digits, load_diabetes
from sklearn.cross_validation import train_test_split
from mlc.parameter_tunes.optimize import Optimization,OptimizeEpochsValidation,OptimizeCrossValidation
import numpy as np
import mlc.utility.config

class OptimizeTest(unittest.TestCase):
    def make_data_set(self):
        digits = load_diabetes()
        x = digits.data
        y = digits.target
        return x, y

    def make_classification_data_set(self):
        digits = load_digits()
        x = digits.data
        y = digits.target

        return x, y

    def test_optimize_regression_time250(self):
        x, y = self.make_data_set()
        optimizer = Optimization()
        parameter = mlc.utility.config.parameter_dictionary['XGBREGTREE']

        optimizer.optimize(x, y, x,parameter, 5, 1, 5, "test_feature", "rmse", "regression", isWriteCsv=False, isBagging=False, id_column_name="ids", ids=None, prediction_column_name="prediction", isOverWrite=False,label_convert_type="log")

    # def test_optimize_epochs_validation_epochs(self):
    #     x, y = self.make_data_set()
    #     optimizer = OptimizeEpochsValidation()

    #     xgboost_regression = {
    #         'model':'XGBREGTREE',
    #         'objective': "reg:linear",
    #         'booster': 'gbtree',
    #         'num_round' : 1000,
    #         'eta' : 0.01,
    #         'min_child_weight':10,
    #         'max_depth': 10,
    #         'subsample': 0.8,
    #         "colsample_bytree": 0.7,
    #         'seed' : 71
    #     }

    #     optimizer.optimize_epochs(x,y,xgboost_regression,5,1000,"rmsle",problem_type="regression",stopping_epochs=[100,200,500,1000],output_files="./output_dir/output_1.csv")

    # def test_optimize_parameter_validation(self):
    #     x,y = self.make_classification_data_set()

    #     optimizer = OptimizeCrossValidation()

    #     xgboost_regression = {
    #         'model':'XGBMULCLASSIFIER',
    #         'objective': "multi:softprob",
    #         'booster': 'gbtree',
    #         'num_round' : 1000,
    #         'eta' : 0.01,
    #         'min_child_weight':10,
    #         'max_depth': 10,
    #         'subsample': 0.8,
    #         "colsample_bytree": 0.7,
    #         'seed' : 71,
    #         "num_class":10,
    #         'eval_metric': "mlogloss"
    #     }

    #     grid_param = {
    #         'min_child_weight': [1,4,5,10],
    #         'max_depth':[1,2,3],
    #         'colsample_bytree':[0.5,0.75,1.0]
    #     }

    #     optimizer.optimize(x,y,5,xgboost_regression,grid_param,1,"logloss","classification","./output_dir/output_2.csv")

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(OptimizeTest))
    return suite
