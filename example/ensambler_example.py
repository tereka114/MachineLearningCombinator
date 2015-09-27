#coding:utf-8
import mlc.layer.ensembler
import json
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

def ensambler_multi():
	digits = load_digits()
	x = digits.data
	y = digits.target

	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

	parameter_file = "./parameter/ensamble_sample.json"

	with open(parameter_file) as data_file:
		    parameter = json.load(data_file)
	clf = mlc.layer.ensembler.EnsambleLayerMultiClassifier()
	train_predict,test_predict = clf.predict_proba(x_train,y_train,x_test,parameter)

	print train_predict.shape,test_predict.shape

def ensambler_binary():
	digits = digits = load_digits(n_class=2)
	x = digits.data
	y = digits.target

	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

	parameter_file = "./parameter/ensamble_sample.json"

	with open(parameter_file) as data_file:
		    parameter = json.load(data_file)
	clf = mlc.layer.ensembler.EnsambleLayerBinaryClassifier()
	train_predict,test_predict = clf.predict_proba(x_train,y_train,x_test,parameter)
	
	print train_predict.shape,test_predict.shape

if __name__ == '__main__':
	ensambler_multi()
	ensambler_binary()