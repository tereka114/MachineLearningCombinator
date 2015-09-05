import numpy as np
import theano
import theano.tensor as T
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import lasagne
import sys
import os
import time


# def load_dataset():
#     # We first define some helper functions for supporting both Python 2 and 3.
#     if sys.version_info[0] == 2:
#         from urllib import urlretrieve
#         import cPickle as pickle

#         def pickle_load(f, encoding):
#             return pickle.load(f)
#     else:
#         from urllib.request import urlretrieve
#         import pickle

#         def pickle_load(f, encoding):
#             return pickle.load(f, encoding=encoding)

#     # We'll now download the MNIST dataset if it is not yet available.
#     url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
#     filename = 'mnist.pkl.gz'
#     if not os.path.exists(filename):
#         print("Downloading MNIST dataset...")
#         urlretrieve(url, filename)

#     # We'll then load and unpickle the file.
#     import gzip
#     with gzip.open(filename, 'rb') as f:
#         data = pickle_load(f, encoding='latin-1')

#     # The MNIST dataset we have here consists of six numpy arrays:
#     # Inputs and targets for the training set, validation set and test set.
#     X_train, y_train = data[0]
#     X_val, y_val = data[1]
#     X_test, y_test = data[2]

#     # The inputs come as vectors, we reshape them to monochrome 2D images,
#     # according to the shape convention: (examples, channels, rows, columns)
#     X_train = X_train.reshape((-1, 784))
#     X_val = X_val.reshape((-1, 784))
#     X_test = X_test.reshape((-1, 784))

#     # The targets are int64, we cast them to int8 for GPU compatibility.
#     y_train = y_train.astype(np.uint8)
#     y_val = y_val.astype(np.uint8)
#     y_test = y_test.astype(np.uint8)

#     # We just return all the arrays in order, as expected in main().
#     # (It doesn't matter how we do this as long as we can read them again.)
#     return X_train, y_train, X_val, y_val, X_test, y_test

# def test():
# 	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
# 	nn = NeuralNetwork(problem_type="classification")
# 	nn.fit(X_train,y_train)

def root_mean_squared_loss_function(a,b):
	return (T.log(1.0 + a) - T.log(1.0 + b)) ** 2

class NeuralNetwork(object):
	def __init__(self,problem_type="regression",batch_size=128,epochs=400,layer_number=[],dropout_layer=[]):
		self.problem_type = problem_type
		self.batch_size = batch_size
		self.epochs = epochs
		self.layer_number = layer_number
		self.dropout_number = dropout_layer
		assert len(self.layer_number) == len(self.dropout_number),"you should correct number between hidden layers and dropout numbers"

	"""
	you should set the construction of model
	"""
	def setModel(self,input_dim,n_classes,input_var):
		neural_network = lasagne.layers.InputLayer(
		    shape=(None, input_dim),input_var=input_var
		)

		# construct the hidden layers
		for layer,dropout_number in zip(self.layer_number,self.dropout_number):
			neural_network = lasagne.layers.DenseLayer(
			    lasagne.layers.DropoutLayer(neural_network,p=dropout_number),
			    num_units=layer,
			    nonlinearity=lasagne.nonlinearities.leaky_rectify,
			)

		if self.problem_type == "classification":
			neural_network = lasagne.layers.DenseLayer(
			    neural_network,
			    num_units=n_classes,
			    nonlinearity=lasagne.nonlinearities.softmax,
			)
		elif self.problem_type == "regression":
			neural_network = lasagne.layers.DenseLayer(
			    neural_network,
			    num_units=n_classes
			)

		self.neural_network = neural_network

	def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
	    assert len(inputs) == len(targets)
	    if shuffle:
	        indices = np.arange(len(inputs))
	        np.random.shuffle(indices)
	    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
	        if shuffle:
	            excerpt = indices[start_idx:start_idx + batchsize]
	        else:
	            excerpt = slice(start_idx, start_idx + batchsize)
	        yield inputs[excerpt], targets[excerpt]

	def fit(self,train_x,train_y,valid=False,evaluate_function=None):
		input_var = T.matrix('inputs')
		print train_x
		print train_y

		if self.problem_type == "regression":
			print ("regression model Lasagne Neural Network")
			self.loss_function_type = "mean_squared_loss"
			train_x_copy = train_x.astype(np.float32).copy()
			train_y_copy = train_y.astype(np.float32).copy().reshape(len(train_y), 1)

			if valid is True:
				split_train_x,valid_x,split_train_y,valid_y = train_test_split(train_x_copy,train_y_copy,test_size=0.1)
			else:
				split_train_x,split_train_y = train_x_copy,train_y_copy

			N,input_dim = split_train_x.shape

			target_var = T.matrix('y')
			self.setModel(input_dim,1,input_var)
		elif self.problem_type == "classification":
			n_classes = 2
			self.loss_function_type = "cross_entropy_loss"
			train_x_copy = train_x.astype(np.float32).copy()
			train_y_copy = train_y.copy().astype(np.uint8)

			if valid is True:
				print ("start split train and valid")
				split_train_x,valid_x,split_train_y,valid_y = train_test_split(train_x_copy,train_y_copy,test_size=0.1)
			else:
				split_train_x,split_train_y = train_x_copy,train_y_copy

			N,input_dim = split_train_x.shape

			target_var = T.ivector('targets')
			self.setModel(input_dim,n_classes,input_var)

		batch_size = self.batch_size
		prediction = lasagne.layers.get_output(self.neural_network)

		loss = None

		if self.loss_function_type == "mean_squared_loss":
			print ("mean_squared_loss")
			loss = lasagne.objectives.squared_error(prediction, target_var)
		elif self.loss_function_type == "cross_entropy_loss":
			print ("cross entropy loss")
			loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

		loss = loss.mean()

		params = lasagne.layers.get_all_params(self.neural_network, trainable=True)
		#updates = lasagne.updates.sgd(loss,params,0.01)
		updates = lasagne.updates.adam(loss, params)

		# updates = lasagne.updates.nesterov_momentum(
  #          loss, params, learning_rate=0.01, momentum=0.9)

		test_prediction = lasagne.layers.get_output(self.neural_network, deterministic=True)

		test_loss = None
		if self.loss_function_type == "mean_squared_loss":
			test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
		elif self.loss_function_type == "cross_entropy_loss":
			test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)

		test_loss = test_loss.mean()
		# As a bonus, also create an expression for the classification accuracy:
		test_acc = None
		if self.loss_function_type == "cross_entropy_loss":
			test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
		                 	dtype=theano.config.floatX)

		# Compile a function performing a training step on a mini-batch (by giving
		# the updates dictionary) and returning the corresponding training loss:
		train_fn = theano.function([input_var, target_var], loss, updates=updates)
		val_fn = None
		# Compile a second function computing the validation loss and accuracy:
		if self.problem_type == "classification":
			val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
		else:
			val_fn = theano.function([input_var, target_var], test_loss)

		self.prediction_fc = theano.function([input_var],test_prediction)

		nlist = np.arange(N)
		print("Starting training...")
		num_epochs = self.epochs
		for epoch in xrange(num_epochs):
		        # In each epoch, we do a full pass over the training data:
		        train_err = 0
		        train_batches = 0
		        start_time = time.time()
		        for batch in self.iterate_minibatches(split_train_x, split_train_y, 500, shuffle=True):
		            inputs, targets = batch
		            train_err += train_fn(inputs, targets)
		            train_batches += 1

		        # And a full pass over the validation data:
		        if valid:
		        	val_err = 0
		        	val_acc = 0
		        	val_batches = 0
		        	for batch in self.iterate_minibatches(valid_x, valid_y, 500, shuffle=False):
		        	    inputs, targets = batch
		        	    if self.problem_type == "classification":
		        	    	err, acc = val_fn(inputs, targets)
		        	    	val_acc += acc
		        	    else:
		        	    	err = val_fn(inputs, targets)
		        	    val_err += err
		        	    val_batches += 1

		        # Then we print the results for this epoch:
		        print("Epoch {} of {} took {:.3f}s".format(
		            epoch + 1, num_epochs, time.time() - start_time))
		        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		        if valid:
		        	print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		        	if self.problem_type == "classification":
		        		print("  validation accuracy:\t\t{:.2f} %".format(
		            		val_acc / val_batches * 100))
		        		valid_preds = self.prediction_fc(valid_x)[:, 1]
		        		roc =  metrics.roc_auc_score(valid_y, valid_preds)
		        		print("  roc score:\t\t{:.6f}".format(roc))



	def predict(self,x):
		y = self.prediction_fc(x.astype(np.float32))
		return y.reshape(len(y))

	def predict_proba(self,x):
		y = self.prediction_fc(x)
		return y

if __name__ == '__main__':
	test()