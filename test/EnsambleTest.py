import unittest
from sklearn.datasets import load_digits, load_diabetes
from sklearn.cross_validation import train_test_split
import mlc.layer.ensembler
import os

class EnsambleTest(unittest.TestCase):
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

	def test_blending_regression(self):
		blending_regression = mlc.layer.ensembler.BlendingRegression()
		data,label = self.make_data_set()
		data_dict = {"data":data}

		blending_regression_parameters = self.blending_regression_parameter()
		print blending_regression.predict(data_dict,label,data_dict,blending_regression_parameters,times=12)

		os.removedirs("blend-dir")

	def test_stacking_regression_layer(self):
		stacking_regression = mlc.layer.ensembler.StackingRegressionLayer()
		data,label = self.make_data_set()

		parameter = {
						"colsample_bytree": 0.7,
						"min_child_weight": 1,
						"num_round": 10,
						"subsample": 0.9,
						"eta": 0.02,
						"max_depth": 10,
						"model": "XGBREGLINEAR",
						"objective": "reg:linear",
						"seed":771
					}

		stacking_regression.stacking(train_x=data,train_y=label,train_ids=range(len(data)),test_ids=range(len(data)),test_x=data,parameter=parameter,
		epochs=3,n_folds=10,isTrain=True,id_name="Id",scoreName="predict",output_dir="datas/stacking-1",pred_name="Score",evaluation_name="")

		stacking_regression.stacking(train_x=data,train_y=label,train_ids=range(len(data)),test_ids=range(len(data)),test_x=data,parameter=parameter,
		epochs=3,n_folds=10,isTrain=False,id_name="Id",scoreName="predict",output_dir="datas/stacking-1",pred_name="Score",evaluation_name="")

	def test_blending_binary_classification(self):
		pass

	def test_blending_multi_classification(self):
		pass

	def blending_regression_parameter(self):
		return [
				{
					"parameter":{
						"colsample_bytree": 0.7,
						"min_child_weight": 1,
						"num_round": 3000,
						"subsample": 0.9,
						"eta": 0.02,
						"max_depth": 10,
						"model": "XGBREGLINEAR",
						"objective": "reg:linear",
						"seed":771
					},
					"data":"data"
				},
				{
					"parameter":{
						"colsample_bytree": 0.7,
						"min_child_weight": 1,
						"num_round": 3000,
						"subsample": 0.9,
						"eta": 0.02,
						"max_depth": 10,
						"model": "XGBREGLINEAR",
						"objective": "reg:linear",
						"seed":71
					},
					"data":"data"
				},
				{
					"parameter":{
						"colsample_bytree": 0.7,
						"min_child_weight": 1,
						"num_round": 3000,
						"subsample": 0.9,
						"eta": 0.02,
						"max_depth": 10,
						"model": "XGBREGLINEAR",
						"objective": "count:poisson",
						"seed":771
					},
					"data":"data"
				}
			]

def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(EnsambleTest))
    return suite