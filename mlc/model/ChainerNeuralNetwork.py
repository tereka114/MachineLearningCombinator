#coding:utf-8
import chainer
from chainer import optimizers
from chainer import cuda
import chainer.functions as Function
import numpy as np
from sklearn.cross_validation import train_test_split
from ..utility.evaluation_functions import evaluate_function

class NeuralNetwork(object):
	def __init__(self):
		pass

class ChainerNeuralNet(object):
	def __init__(self,batch_size = 100,cuda=False,varbose=True,epoch=100,problem_type='classifier',model=None,layer_param=[]):
		self.batch_size = batch_size
		self.cuda = cuda
		self.varbose = varbose
		self.model = model
		self.epochs = epoch
		self.problem_type = problem_type
		self.loss_function_type = "softmax_cross_entropy"
		self.layer_param = layer_param

	def create_model(self,input_size):
		pass

	def forward(self,x_batch,y_batch,train=True):
		pass

	def loss_function(self,y_predict,y_true):
		if self.loss_function_type == "mean_squared_loss":
			return Function.mean_squared_error(y_predict, y_true)
		elif self.loss_function_type == "softmax_cross_entropy":
			return Function.softmax_cross_entropy(y_predict,y_true)
		elif self.loss_function_type == "":
			pass
		return None

	def fit(self,train_x,train_y,validation=True):
		if self.problem_type == "regression":
			self.loss_function_type = "mean_squared_loss"
			train_x_copy = train_x.astype(np.float32).copy()
			train_y_copy = np.log(train_y.astype(np.float32).copy()).reshape(len(train_y), 1)
			self.create_model(len(train_x[0]),1)

		optimizer = optimizers.Adam()
		optimizer.setup(self.model.collect_parameters())

		split_train_x,valid_x,split_train_y,valid_y = train_test_split(train_x_copy,train_y_copy,test_size=0.1)
		
		N = len(train_x_copy)
		valid_loss_memory = None

		for epoch in xrange(self.epochs):
			if self.varbose:
				print epoch + 1
			sum_loss = 0.0
			validation_sum_loss = 0.0
			validate_count = 0

			perm = np.random.permutation(N)
			for i in xrange(0,N,self.batch_size):
				x_batch = train_x_copy[perm[i:i+self.batch_size]]
				y_batch = train_y_copy[perm[i:i+self.batch_size]]

				if self.cuda:
					x_batch = cuda.to_gpu(x_batch)
					y_batch = cuda.to_gpu(y_batch)

				optimizer.zero_grads()
				y,loss = self.forward(x_batch, y_batch)
				loss.backward()
				optimizer.update()

				if self.varbose:
					sum_loss += float(cuda.to_cpu(loss.data)) * self.batch_size

			if validation:
				validation_sum_loss += self.validate(valid_x,valid_y) 
				validate_count += 1

			"""
			early stopping system
			"""
			if epoch >= 40 and epoch % 10 == 0:
				print validation_sum_loss,valid_loss_memory
				validation_loss = validation_sum_loss / validate_count
				if valid_loss_memory == None:
					valid_loss_memory = validation_loss / validate_count
				else:
					if valid_loss_memory < validation_loss:
						valid_loss_memory = validation_loss
						break
					else:
						valid_loss_memory = validation_loss
				print valid_loss_memory,validation_loss

			if self.varbose:
				print sum_loss / N


	def validate(self,valid_x,valid_y):
		sum_loss = 0.0
		cnt = 0
		for i in xrange(0,len(valid_x),self.batch_size):
			x_batch = valid_x[i:i+self.batch_size]
			y_batch = valid_y[i:i+self.batch_size]

			if self.cuda:
				x_batch = cuda.to_gpu(x_batch)
				y_batch = cuda.to_gpu(y_batch)

			y,loss = self.forward(x_batch, y_batch)
			sum_loss += evaluation_functions.evaluate_function(y_batch,cuda.to_cpu(y.data),'rslme')
			cnt += 1
		return sum_loss / cnt

	def predict(self,x_data):
		y = None
		x_data_copy = x_data.astype(np.float32).copy()
		if self.cuda:
			N = len(x_data)
			y = np.zeros((N,1))
			for i in xrange(0,N,1):
				x_batch = x_data_copy[i:i+self.batch_size]

				if self.cuda:
					x_batch = cuda.to_gpu(x_batch)

				y_batch = cuda.to_cpu(self.forward(x_batch, None, False).data)
				y[i:i+self.batch_size] = y_batch
		else:
			y = self.forward(x_data.astype(np.float32), None,False)
		return np.exp(y).reshape((len(x_data)))

	def predict_proba(self,x_data):
		y = self.forward(x_data, None, False)
		y = Function.softmax(y)
		return y.data

class Chainer3LayerNeuralNetwork(ChainerNeuralNet):
	def __init__(self,batch_size = 100,cuda=False,varbose=True,epoch=100,problem_type='classifier',layer1=784,layer2=784):
		super(Chainer3LayerNeuralNetwork,self).__init__(batch_size=batch_size,cuda=cuda,varbose=varbose,epoch=epoch,problem_type=problem_type,layer_param=None)
		self.layer1 = layer1
		self.layer2 = layer2

	def create_model(self,input_size,output_size):
		self.model = chainer.FunctionSet(
			l1 = Function.Linear(input_size,self.layer1),
			l2 = Function.Linear(self.layer1,self.layer2),
			l3 = Function.Linear(self.layer2,output_size)
			)
		if self.cuda:
			print "change cuda mode"
			cuda.init()
			self.model.to_gpu()

	def forward(self,x_batch,y_batch=None,train=True):
		x = chainer.Variable(x_batch)
		t = None
		if train:
			t = chainer.Variable(y_batch)
		h1 = Function.dropout(Function.relu(self.model.l1(x)),  train=train)
		h2 = Function.dropout(Function.relu(self.model.l2(h1)), train=train)
		y = self.model.l3(h2)
		if train:
			return y,self.loss_function(y, t)
		else:
			return y

class ChainerConvolutionalNeuralNetwork(ChainerNeuralNet):
	def __init__(self):
		pass

	def create_model(self,input_size,output_size):
		self.model = chainer.FunctionSet(conv1=Function.Convolution2D(3,32,11,stride=4),
                    bn1   = Function.BatchNormalization(32),
                    conv2=Function.Convolution2D(32,64,3,pad=1),
                    bn2   = Function.BatchNormalization( 64),
                    conv3=Function.Convolution2D(64,64,3,pad=1),
                    fl4=Function.Linear(1024, 256),
                    fl5=Function.Linear(256, 10))

	def fit(self,train_x,train_y,transpose=True):
		self.create_model(len(train_x[0]),10)
		optimizer = optimizers.Adam()
		optimizer.setup(self.model.collect_parameters())
		
		N = len(train_x)

		for epoch in xrange(self.epochs):
			if self.varbose:
				print epoch + 1
			sum_loss = 0.0
			perm = np.random.permutation(N)
			for i in xrange(0,N,self.batch_size):
				x_batch = train_x[perm[i:i+self.batch_size]]
				y_batch = train_y[perm[i:i+self.batch_size]]

				if self.cuda:
					x_batch = cuda.to_gpu(x_batch)
					y_batch = cuda.to_gpu(y_batch)

				optimizer.zero_grads()
				y,loss = self.forward(x_batch, y_batch)
				loss.backward()
				optimizer.update()

				if self.varbose:
					sum_loss += float(cuda.to_cpu(loss.data)) * self.batch_size
			if self.varbose:
				print sum_loss / N


	def predict(self,x_data):
		y = self.forward(x_data, None, False)
		y = Function.softmax(y)
		return np.argmax(y.data,axis=1)

	def predict_proba(self,x_data):
		y = self.forward(x_data, None, False)
		y = Function.softmax(y)
		return y.data

