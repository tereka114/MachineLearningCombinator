import unittest
from sklearn.datasets import load_digits, load_diabetes
from sklearn.cross_validation import train_test_split
from mlc.parameter_tunes.optimize import Optimization
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

    def test_optimize_regression_time1(self):
        x, y = self.make_data_set()
        optimizer = Optimization()
        parameter = mlc.utility.config.parameter_dictionary['RFREG']

        optimizer.optimize(x, y, x,parameter, 1, 1, 5, "test_feature", "rmse", "regression", isWriteCsv=False, isBagging=False, id_column_name="ids", ids=None, prediction_column_name="prediction", isOverWrite=False)

    def test_optimize_regression_time250(self):
        x, y = self.make_data_set()
        optimizer = Optimization()
        parameter = mlc.utility.config.parameter_dictionary['RFREG']

        optimizer.optimize(x, y, x,parameter, 250, 1, 5, "test_feature", "rmse", "regression", isWriteCsv=False, isBagging=False, id_column_name="ids", ids=None, prediction_column_name="prediction", isOverWrite=False)

    def test_lasagne_classification(self):
        x, y = self.make_classification_data_set()

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(OptimizeTest))
    return suite
