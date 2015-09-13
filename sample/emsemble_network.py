import Ensembler
import json

parameter_file_lists = ["parameter/DebugSingleModel.json","",""]
final_predict_parameter = ""

for index,parameter_file in enumerate(parameter_file_lists):
	with open(parameter_file) as data_file:    
	    parameter = json.load(data_file)
	ensenble_parameters = parameter["ensenble_parameters"]
	clf = Ensembler.EmsambleLayerBinaryClassifier()
	train_features,test_features = clf.predict(ensenble_parameters)