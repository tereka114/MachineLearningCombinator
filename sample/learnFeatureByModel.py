import os
import subprocess

feature_dir = "feature_vector"
parameter_file = "parameter/LasagneNeuralNetwork.json"
for feature_file_name in os.listdir(feature_dir):
	feature_file = os.path.join(feature_dir,feature_file_name)
	subprocess.call("python boosting_prediction.py %s %s" % (feature_file,parameter_file),shell=True)