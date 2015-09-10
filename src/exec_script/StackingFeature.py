import numpy as np

class StackingFeature(object):
	def __init__(self):
		pass

	def concatArray(self,train,test,train_joint_array,test_joint_array):
		temp_train_jointed_array = train_joint_array.reshape(1,len(train_joint_array))
		temp_test_jointed_array = test_joint_array.reshape(1,len(test_joint_array))
		result_train = np.hstack((train,temp_train_jointed_array))
		result_test = np.hstack((test,temp_test_jointed_array))
		return result_train,result_test

	def mean(self,train,test):
		train_joint_array = np.mean(train,axis=1)
		test_joint_array = np.mean(test,axis=1)
		return self.concatArray(train, test, train_joint_array, test_joint_array)

	def std(self,train,test):
		train_joint_array = np.std(train,axis=1)
		test_joint_array = np.std(test,axis=1)
		return self.concatArray(train, test, train_joint_array, test_joint_array)

	def convert(self,train,test,mean_flag=True,std_flag=True,tsne_flag=True):
		train_array = train
		test_array = test
		
		if mean_flag:
			train_array,test_array = self.mean(train, test)

		if std_flag:
			train_array,test_array = self.std(train, test)

		return train_array,test_array