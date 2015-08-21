import numpy as np
import theano
import theano.tensor as T
from sklearn.cross_validation import train_test_split
import lasagne

def root_mean_squared_loss_function(a,b):
	return (T.log(1.0 + a) - T.log(1.0 + b)) ** 2

class NeuralNetwork(object):
	def __init__(self):
		self.problem_type = "regression"
		pass

	"""
	you should set the construction of model
	"""
	def setModel(self,input_dim):
		n_classes = 1

		neural_network = lasagne.layers.InputLayer(
		    shape=(None, input_dim),
		)
		neural_network = lasagne.layers.DenseLayer(
		    lasagne.layers.DropoutLayer(neural_network,p=0.3),
		    num_units=2400,
		    nonlinearity=lasagne.nonlinearities.sigmoid,
		)
		neural_network = lasagne.layers.DenseLayer(
		    lasagne.layers.DropoutLayer(neural_network,p=0.3),
		    num_units=2000,
		    nonlinearity=lasagne.nonlinearities.sigmoid,
		)
		# neural_network = lasagne.layers.DenseLayer(
		#     lasagne.layers.DropoutLayer(neural_network,p=0.3),
		#     num_units=512,
		#     nonlinearity=lasagne.nonlinearities.sigmoid,
		# )
		neural_network = lasagne.layers.DenseLayer(
		    lasagne.layers.DropoutLayer(neural_network,p=0.3),
		    num_units=n_classes,
		    nonlinearity=lasagne.nonlinearities.rectify,
		)
		self.neural_network = neural_network

		#pred = lasagne.layers.get_output(self.neural_network, input_var, deterministic=True)

	def fit(self,train_x,train_y):
		if self.problem_type == "regression":
			self.loss_function_type = "mean_squared_loss"
			train_x_copy = train_x.astype(np.float32).copy()
			train_y_copy = train_y.astype(np.float32).copy().reshape(len(train_y), 1)
		elif self.loss_function_type == "classification":
			pass

		split_train_x,valid_x,split_train_y,valid_y = train_test_split(train_x_copy,train_y_copy,test_size=0.1)

		print split_train_x.shape,valid_x.shape
		N,input_dim = split_train_x.shape
		input_var = T.matrix('x')
		self.setModel(input_dim)
		target_var = T.matrix('y')
		batch_size = 128 
		
		prediction = lasagne.layers.get_output(self.neural_network,input_var)
		loss = lasagne.objectives.squared_error(prediction, target_var)
		loss = loss.mean()

		params = lasagne.layers.get_all_params(self.neural_network, trainable=True)
		updates = lasagne.updates.adam(loss, params)

		pred = lasagne.layers.get_output(self.neural_network, input_var, deterministic=True)
		test_acc = T.mean(((pred) - (target_var)) ** 2,dtype=theano.config.floatX)
		test = theano.function([input_var, target_var], [test_acc])
		# updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01,
  #                                               momentum=0.9)
		train_fn = theano.function([input_var, target_var], loss, updates=updates)
		#test_acc = T.mean((T.log(pred + 1.0) - T.log(target_var + 1.0)) ** 2,dtype=theano.config.floatX)
		prediction_fc = theano.function([input_var], pred)
		self.prediction_fc = prediction_fc

		nlist = np.arange(N)
		for epoch in xrange(100):
		    np.random.shuffle(nlist)
		    train_loss_sum = 0.0

		    for j in xrange(N / batch_size):
		        ns = nlist[batch_size*j:batch_size*(j+1)]
			print split_train_x[ns]
		        train_loss = train_fn(split_train_x[ns], split_train_y[ns])
		        train_loss_sum += train_loss
			print train_loss
		    train_loss,train_acc_loss = test(split_train_x,split_train_y)
		    est_acc_loss = test(valid_x,valid_y)
		    print("%d: train_loss=%.4f, train_acc_loss=%.4f test_acc_loss=%.4f" % (epoch+1, train_loss, train_acc_loss,test_acc_loss))

	def predict(self,x):
		y = self.prediction_fc(x)
		return y.reshape(len(y))
