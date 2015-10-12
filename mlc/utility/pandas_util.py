import pandas as pd
from numpy.random import *
import numpy as np
from sklearn import preprocessing

def add_count_data(df,count_columns_name,additional_columns_name=None):
	"""
	:df: input dataframe
	:count_columns_name: column name which you want to group
	:additional_columns_name: additional columns name(counted column)
	"""
	#df = pd.concat([df,pd.DataFrame(df.count(axis=0),columns=['F']).T],axis=1)
	cnt = pd.DataFrame(df[count_columns_name].value_counts(),columns=[additional_columns_name])
	cnt[count_columns_name] = cnt.index
	return pd.merge(df,cnt,on=count_columns_name,how='left')

def category_vector_one_of_k(df,convert_list,varbose=True,varbose_index=100000):
	"""
	convert label values to one of k columns

	:df: dataframe
	:convert list: columns you want to convert one of k expression
	"""
	for convert_column in convert_list:
		if varbose == True:
			print convert_column
		for index,value in enumerate(df[convert_column].values):
			if varbose and (index % varbose_index == 0):
				print index
			column = convert_column + ":" + str(value)
			if not column in df.columns:
				df[convert_column + ":" + str(value)] = 0
			df.ix[index,convert_column + ":" + str(value)] = 1
		del df[convert_column]
	return df

def category_vector_label_train_test(train_df,test_df,convert_list):
	for convert_column in convert_list:
		le = preprocessing.LabelEncoder()

		train_values = [str(value) for value in train_df[convert_column].values]
		test_values = [str(value) for value in test_df[convert_column].values]

		label_list = train_values + test_values
		le.fit(label_list)

		train_df[convert_column] = le.transform(train_values)
		test_df[convert_column] = le.transform(test_values)

	return train_df,test_df

def category_vector_label(df,convert_list):
	"""
	:df: pandas datafram 
	"""
	for convert_column in convert_list:
		le = preprocessing.LabelEncoder()
		data_values = [str(value) for value in df[convert_column].values]
		le.fit(data_values)
		df[convert_column] = le.transform(data_values)
	return df

def extract_na_columns(df):
	"""
	check na for DataFrame

	:df: dataframe
	"""
	na_count_data = len(df) - df.count()

def pandas_column_infomation(df,min_column_flag=True):
	"""
	output of pandas dataframe

	:param df: pandas dataframe
	:param min_column_flag:
	"""
	for column in df.columns:
		column_value_list = df[column].unique()
		if len(column_value_list) >= 1:
			print column,type(column_value_list[0]),len(column_value_list),column_value_list[0]
		else:
			print column,type(column_value_list[0]),len(column_value_list)

def pandas_extract_min_pattern_columns(df,min_number=10):
	min_column_list = []
	for column in df.columns:
		nunique = df[column].nunique()
		if nunique <= min_number:
			min_column_list.append(column)
	return min_column_list

def pandas_remove_ununique_count(df):
	"""
	remove ununique column from dataframe.

	:param df: dataframe
	"""
	for column in df.columns:
		if df[column].values.nunique() < 2:
			df = df.drop(column)
	return df

def pandas_extract_category_column(df):
	"""
	get the column
	:param df: dataframe 
	"""
	str_columns = []
	for column in df.columns:
		for value in df[column].values:
			if type(value) == str or type(value) == bool:
				str_columns.append(column)
				break
	return str_columns

def pandas_extranct_numeric_column(df):
	"""
	get the numeric columns
	:param df: dataframe
	"""
	not_str_columns = []
	for column in df.columns:
		flag = True
		for value in df[column].values:
			if type(value) == str or type(value) == bool:
				flag = False
				break
		if flag:
			not_str_columns.append(column)
	return not_str_columns

def pandas_convert_time(df,columns,date_type=None,convert_function=None):
	"""
	convert time to month day second, so on
	:param df: dataframe

	example:
		spring data: 07JUL06:00:00:00

	"""
	add_columns_name = []

	if date_type == None and convert_function == None:
		print "you should decide how to convert"

	if "spring_data" == date_type:
		for column in columns:
			print column
			year_column_name = "{}:YEAR".format(column)
			month_column_name = "{}:MONTH".format(column)
			day_column_name = "{}:DAY".format(column)

			add_columns_name.append(year_column_name)
			add_columns_name.append(month_column_name)
			add_columns_name.append(day_column_name)

			for value_index,time in enumerate(list(df[column].values)):
				if value_index % 100000 == 0:
					print value_index

				year = "UN"
				month = "UNK"
				day = "UNK"

				if len(str(time)) == len("07JUL06:00:00:00"):
					year = time[0:2]
					month = time[2:5]
					day = time[5:7]

				data_pair_list = [(year_column_name,year),(month_column_name,month),(day_column_name,day)]

				for index,column_data in enumerate(data_pair_list):
					column_name = column_data[0]
					data = column_data[1]

					if not column_name in df.columns:
						df[column_name] = str(0)
					df.ix[index,column_name] = data
			del df[column]
	elif "date" == date_type:
		for column in columns:
			year_column_name = "{}:YEAR".format(column)
			month_column_name = "{}:MONTH".format(column)
			day_column_name = "{}:DAY".format(column)

			add_columns_name.append(year_column_name)
			add_columns_name.append(month_column_name)
			add_columns_name.append(day_column_name)

			for value_index,time in enumerate(list(df[column].values)):
				if value_index % 100000 == 0:
					print value_index

				year = "na"
				month = "na"
				day = "na"

				if len(str(time)) == len("2015-09-17"):
					year = time[0:4]
					month = time[5:7]
					day = time[8:-1]

				data_pair_list = [(year_column_name,year),(month_column_name,month),(day_column_name,day)]

				for index,column_data in enumerate(data_pair_list):
					column_name = column_data[0]
					data = column_data[1]

					if not column_name in df.columns:
						df[column_name] = str(0)
					df.ix[index,column_name] = data
			del df[column]
	return df,add_columns_name