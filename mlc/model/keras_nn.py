import keras_recipe
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils
import numpy as np

class KerasNeuralNetwork(object):
	def __init__(self,epochs,batch_size,loss_function="categorical_crossentropy",validation=0.0):
		self.epochs = epochs
		self.batch_size = batch_size
		self.loss_function = loss_function
		self.validation = validation

	def fit(self,x,y):
		self.model.fit(x, y, nb_epoch=self.epochs, batch_size=self.batch_size, verbose=1, show_accuracy=True, validation_split=self.validation)

	def predict(self,x):
		return self.model.predict(x)

	def predict_proba(self,x):
		return self.model.predict_proba(x)

class Keras3LayerNeuralNetwork(KerasNeuralNetwork):
	def __init__(self,input_dim=0,layer1=0,layer2=0,layer3=0,nb_classes=0,dropout1=0.0,dropout2=0.0,dropout3=0.0,epochs=10,batch_size=128,loss_function="categorical_crossentropy"):
		self.model = keras_recipe.built_nn3layer(input_dim,layer1=layer1,layer2=layer2,layer3=layer3,nb_classes=nb_classes,dropout1=dropout1,dropout2=dropout2,dropout3=dropout3)
		self.model.compile(loss=loss_function, optimizer='adam')
		super(Keras3LayerNeuralNetwork,self).__init__(epochs, batch_size)

class Keras3LayerNeuralNetworkRegression(KerasNeuralNetwork):
	def __init__(self,input_dim=0,layer1=0,layer2=0,layer3=0,nb_classes=0,dropout1=0.0,dropout2=0.0,dropout3=0.0,epochs=10,batch_size=128,loss_function="mae"):
		self.model = keras_recipe.built_nn3layer_regression(input_dim,layer1=layer1,layer2=layer2,layer3=layer3,nb_classes=nb_classes,dropout1=dropout1,dropout2=dropout2,dropout3=dropout3)
		self.model.compile(loss=loss_function, optimizer='adadelta')
		super(Keras3LayerNeuralNetworkRegression,self).__init__(epochs, batch_size)

class Keras4LayerNeuralNetwork(KerasNeuralNetwork):
	def __init__(self,input_dim=0,layer1=0,layer2=0,layer3=0,layer4=0,nb_classes=0,dropout1=0.0,dropout2=0.0,dropout3=0.0,dropout4=0.0,epochs=10,batch_size=128,loss_function="categorical_crossentropy"):
		self.model = keras_recipe.built_nn4layer(input_dim,layer1=layer1,layer2=layer2,layer3=layer3,layer4=layer4,nb_classes=nb_classes,dropout1=dropout1,dropout2=dropout2,dropout3=dropout3,dropout4=dropout4)
		self.model.compile(loss=loss_function, optimizer='adam')
		super(Keras4LayerNeuralNetwork,self).__init__(epochs, batch_size)

class Keras4LayerNeuralNetworkRegression(KerasNeuralNetwork):
	def __init__(self,input_dim=0,layer1=0,layer2=0,layer3=0,layer4=0,nb_classes=0,dropout1=0.0,dropout2=0.0,dropout3=0.0,dropout4=0.0,epochs=10,batch_size=128,loss_function="mae"):
		self.model = keras_recipe.built_nn3layer(input_dim,layer1=layer1,layer2=layer2,layer3=layer3,nb_classes=nb_classes,dropout1=dropout1,dropout2=dropout2,dropout3=dropout3)
		self.model.compile(loss=loss_function, optimizer='adam')
		super(Keras4LayerNeuralNetworkRegression,self).__init__(epochs, batch_size)

class ImageClassificationNeuralNetwork(KerasNeuralNetwork):
	def __init__(self,epochs, batch_size):
		super(ImageClassificationNeuralNetwork,self).__init__(epochs, batch_size)

	def fit(self,x,y,doRTA):
		if doRTA == False:
			self.model.fit({"input":x,"output":y},nb_epoch=self.epochs,batch_size=self.batch_size)
		else:
			datagen = ImageDataGenerator(
			        featurewise_center=True,  # set input mean to 0 over the dataset
			        samplewise_center=False,  # set each sample mean to 0
			        featurewise_std_normalization=True,  # divide inputs by std of the dataset
			        samplewise_std_normalization=False,  # divide each input by its std
			        zca_whitening=False,
			        rotation_range=20,
			        width_shift_range=0.2, 
			        height_shift_range=0.2,
			        horizontal_flip=True, 
			        vertical_flip=False)
			datagen.fit(x)

			for e in range(self.epochs):
			    print('-'*40)
			    print('Epoch', e)
			    print('-'*40)
			    print('Training...')
			    # batch train with realtime data augmentation
			    progbar = generic_utils.Progbar(x.shape[0])
			    for X_batch, Y_batch in datagen.flow(x, y):
			        loss = self.model.train_on_batch({"input":X_batch,"output":Y_batch})
			        progbar.add(X_batch.shape[0], values=[('train loss', loss[0])])

	def predict_proba(self,x):
		return self.model.predict({"input":x},batch_size=self.batch_size)['output']

class ResidualNeuralNetwork(ImageClassificationNeuralNetwork):
	def __init__(self,shape,epochs, batch_size,n_class=447):
		self.model = keras_recipe.create_50_layer(shape,n_class)
		self.model.compile('adadelta', {'output':'categorical_crossentropy'})
		super(ResidualNeuralNetwork,self).__init__(epochs, batch_size)

class DAGCNNs(ImageClassificationNeuralNetwork):
	def __init__(self,shape=None,epochs=60,batch_size=32,n_conv=5,n_class=447,n_filter=64):
		self.model = keras_recipe.built_dagcnns(shape,n_filter=n_filter,n_class=n_class,n_conv=n_conv)
		self.model.compile('adadelta', {'output':'categorical_crossentropy'})
		super(DAGCNNs,self).__init__(epochs, batch_size)