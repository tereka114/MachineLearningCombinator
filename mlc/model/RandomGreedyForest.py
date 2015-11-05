#coding:utf-8
import subprocess

class RandomGreedyForest(object):
	def __init__(self,**params):
	    self.clf = None
	    self.params = params

	def fit(self,x,y):
		pass