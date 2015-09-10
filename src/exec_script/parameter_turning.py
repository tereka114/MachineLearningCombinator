import config
import optimize
import json
import os
import sys
import pickle

feature_file = sys.argv[1]
model_name = sys.argv[2]

train_x,train_y,test = pickle.load(open(feature_file,"r"))

feature_name,ext = os.path.splitext(os.path.basename(feature_file))
loss_function = "rmsle"
parameter_dict = config.parameter_dictionary[model_name]

turning_param = optimize.optimize_model_parameter(train_x,train_y,model_name,times=1,loss_function=loss_function,parameter=parameter_dict,max_evals=200,total_time=None)

for k,v in parameter_dict.items():
	if not k in turning_param:
		turning_param[k] = parameter_dict[k]

with open("./parameter/" + feature_name + "_" + model_name + '.json', 'w') as f:
		    json.dump(turning_param, f, sort_keys=True, indent=4)