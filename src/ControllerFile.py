#coding:utf-8
import os
import subprocess

"""
現状想定しているのは1ファイル時のみ
"""

"""
特徴ベクトルを構築する
"""
subprocess.call("python feature_vector.py", shell=True)

"""
Stacking Algorithm
"""
feature_dir = "feature_vector"
parameter_dir = "parameter"
for feature_file_name in os.listdir(feature_dir):
	feature_file = os.path.join(feature_dir,feature_file_name)
	if feature_file_name.endswith(".pkl"):
		for parameter_file_name in os.listdir(parameter_dir):
			parameter_file = os.path.join(parameter_dir,parameter_file_name)
			if parameter_file.endswith(".json"):
				print parameter_file,feature_file
				print "python boosting_prediction.py %s %s" % (feature_file,parameter_file)
				subprocess.call("python boosting_prediction.py %s %s" % (feature_file,parameter_file),shell=True)
"""
Blending Algorithm
"""
#subprocess.call("python boosting_result.py",shell=True)