from ..utility.Util import model_select
import seaborn
import matplotlib.pyplot as plt
import numpy as np

def feature_importance_rf_plot(x, y, filepath=None, header=None,param=None, visualize=False):
	"""
	:params :
 	"""
 	clf = model_select(param)
 	clf.fit(x, y)
 	print clf.feature_importances_
 	importances = clf.feature_importances_
 	indices = np.argsort(importances)[::-1]

 	plt.figure()
 	plt.title("Feature importances")
 	plt.bar(range(10), importances[indices],
 	       color="r", align="center")
 	plt.xticks(range(10), indices)
 	plt.xlim([-1, 10])
 	plt.show()

def feature_importance_xg_plot(param, filepath, visualize=False):
    pass

def feature_correlation(x,filepath=None, visualize=False):
    """
    :param x:
    """
    seaborn.pairplot(x)
    if visualize:
    	seaborn.plt.show()
    if not filepath == None:
    	plt.savefig(filepath)