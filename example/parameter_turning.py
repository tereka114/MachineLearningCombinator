#coding:utf-8
import mlc.parameter_tunes.optimize
from sklearn.datasets import load_digits

digits = load_digits(n_class=2)
x = digits.data
y = digits.target
parameter = mlc.parameter_tunes.optimize.optimize_model_parameter(x,y,"XGBREGLOGISTIC",max_evals=3,loss_function='auc')

print parameter