import numpy as np
import theano
import theano.tensor as T
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import lasagne
import sys
import os
import time


def root_mean_squared_loss_function(a, b):
    return (T.log(1.0 + a) - T.log(1.0 + b)) ** 2


def loss_function_based_theano(input, target):
    pass


class NeuralNetwork(object):

    def __init__(self, problem_type="regression", batch_size=128, epochs=400, layer_number=[], dropout_layer=[]):
        self.problem_type = problem_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.layer_number = layer_number
        self.dropout_number = dropout_layer
        assert len(self.layer_number) == len(
            self.dropout_number), "you should correct number between hidden layers and dropout numbers"

    def getParam(self):
        param = {
            "problem_type": self.problem_type,
            "batchsize": self.batch_size,
            "self.epochs": self.epochs
        }
        return param

    """
	you should set the construction of model
	"""

    def setModel(self, input_dim, n_classes, input_var):
        neural_network = lasagne.layers.InputLayer(
            shape=(None, input_dim), input_var=input_var
        )

        # construct the hidden layers
        for layer, dropout_number in zip(self.layer_number, self.dropout_number):
            neural_network = lasagne.layers.DenseLayer(
                lasagne.layers.DropoutLayer(neural_network, p=dropout_number),
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

    def select_update_function(self, loss, params, update_function_type):
        if update_function_type == "adam":
            updates = lasagne.updates.adam(loss, params)
        elif update_function_type == "sgd":
            updates = lasagne.updates.sgd(loss, params, 0.01)
        elif update_function_type == "nesterov_momentum":
            updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=0.01, momentum=0.9)
        return updates

    def select_loss_function(self, prediction, target_var, function_type):
        if self.loss_function_type == "mean_squared_loss":
            print ("mean_squared_loss")
            loss = lasagne.objectives.squared_error(prediction, target_var)
        elif self.loss_function_type == "cross_entropy_loss":
            print ("cross entropy loss")
            loss = lasagne.objectives.categorical_crossentropy(
                prediction, target_var)
        return loss

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

    def fit(self, train_x, train_y, valid=True, evaluate_function=None):
        """
        :params train_x:
        :params train_y:
        :params valid: 
        """
        input_var = T.matrix('inputs')
        train_x_copy = train_x.astype(np.float32).copy()

        if self.problem_type == "regression":
            n_classes = 1
            print ("regression model Lasagne Neural Network")
            self.loss_function_type = "mean_squared_loss"
            train_y_copy = train_y.astype(
                np.float32).copy().reshape(len(train_y), 1)

            target_var = T.matrix('y')
        elif self.problem_type == "classification":
            n_classes = len(set(train_y))
            self.n_classes = n_classes
            print ("classification model Lasagne Neural Network")
            self.loss_function_type = "cross_entropy_loss"
            train_y_copy = train_y.copy().astype(np.uint8)
            target_var = T.ivector('targets')

        if valid is True:
            print ("start split train and valid")
            split_train_x, valid_x, split_train_y, valid_y = train_test_split(
                train_x_copy, train_y_copy, test_size=0.01)
        else:
            split_train_x, split_train_y = train_x_copy, train_y_copy

        N, input_dim = split_train_x.shape
        self.setModel(input_dim, n_classes, input_var)
        prediction = lasagne.layers.get_output(self.neural_network)

        loss = None

        if self.loss_function_type == "mean_squared_loss":
            print ("mean_squared_loss")
            loss = lasagne.objectives.squared_error(prediction, target_var)
        elif self.loss_function_type == "cross_entropy_loss":
            print ("cross entropy loss")
            loss = lasagne.objectives.categorical_crossentropy(
                prediction, target_var)

        loss = loss.mean()

        # define update functions
        params = lasagne.layers.get_all_params(
            self.neural_network, trainable=True)

        update_function_type = "nesterov_momentum"
        updates = self.select_update_function(
            loss, params, update_function_type)

        test_prediction = lasagne.layers.get_output(
            self.neural_network, deterministic=True)
        train_fn = theano.function(
            [input_var, target_var], loss, updates=updates)

        test_loss = self.select_loss_function(
            test_prediction, target_var, self.loss_function_type)

        # define test function
        val_fn = None
        test_acc = None
        # Compile a second function computing the validation loss and accuracy:
        if self.problem_type == "classification":
            test_loss = test_loss.mean()
            if self.loss_function_type == "cross_entropy_loss":
                test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                                  dtype=theano.config.floatX)
            val_fn = theano.function([input_var, target_var], [
                                     test_loss, test_acc])
        else:
            val_fn = theano.function([input_var, target_var], test_loss)

        self.prediction_fc = theano.function([input_var], test_prediction)

        print("Starting training...")
        num_epochs = self.epochs
        for epoch in xrange(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(split_train_x, split_train_y, self.batch_size, shuffle=True):
                inputs, targets = batch
                # print self.prediction_fc(inputs)
                # print inputs,targets
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            if valid:
                val_err = 0
                val_acc = 0
                val_batches = 0
                for batch in self.iterate_minibatches(valid_x, valid_y, 1, shuffle=False):
                    inputs, targets = batch
                    if self.problem_type == "classification":
                        err, acc = val_fn(inputs, targets)
                        val_acc += acc
                    else:
                        err = val_fn(inputs, targets)
                    val_err += err
                    val_batches += 1
                #print("  valid loss:\t\t{:.6f}".format(val_err / val_batches))

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(
                train_err / train_batches))

    def predict(self, x):
        if self.problem_type == "regression":
            y = self.prediction_fc(x.astype(np.float32))
            return y.reshape(len(y))
        elif self.problem_type == "classification":
            y = self.prediction_fc(x.astype(np.float32))
            return np.argmax(y,axis=1)

    def predict_proba(self, x):
        y = self.prediction_fc(x)
        return y