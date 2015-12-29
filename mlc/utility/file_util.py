import csv
import os
def dictionary_in_list_convert_to_csv(dictionary_list,filename):
	"""
	:params dictionary_list:
	:params filename:
	"""
	keys = dictionary_list[0].keys()
	with open(filename, 'wb') as output_file:
	    dict_writer = csv.DictWriter(output_file, keys)
	    dict_writer.writeheader()
	    dict_writer.writerows(dictionary_list)

def mkdir(filepath):
	if not os.path.exists(filepath):
		os.makedirs(filepath)