#coding:utf-8
import unittest
import mlc.model.LasagneNeuralNetwork
from sklearn.datasets import load_digits,load_diabetes
from sklearn.cross_validation import train_test_split
import numpy as np

class ModelTest(unittest.TestCase):
	def make_data_set(self):
		digits = load_diabetes()
		x = digits.data
		y = digits.target
		return x,y

	def make_classification_data_set(self):
		digits = load_digits()
		x = digits.data
		y = digits.target

		return digits

	def test_lasagne_regression(self):
		x,y = self.make_data_set()
		print len(x),len(y)
		neural_network = mlc.model.LasagneNeuralNetwork.NeuralNetwork(problem_type="regression",batch_size=10,epochs=400,layer_number=[100,100,100],dropout_layer=[0.0,0.0,0.0])
		neural_network.fit(x,np.log(y + 0.001),valid=True,evaluate_function="mean_squared_loss")

	def test_lasagne_classification(self):


	def test_xgboost_regression(self):
		x,y = self.make_data_set()

def suite():
  suite = unittest.TestSuite()
  suite.addTests(unittest.makeSuite(ModelTest))
  return suite