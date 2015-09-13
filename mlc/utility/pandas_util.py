import pandas as pd
from numpy.random import *
import numpy as np

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

if __name__ == '__main__':
	df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
		'foo', 'bar', 'foo', 'foo'],
		'B' : ['one', 'one', 'two', 'three',
		'two', 'two', 'one', 'three'],
		'C' : randn(8), 'D' : randn(8)})
	print add_count_data(df,count_columns_name='A',additional_columns_name='count')