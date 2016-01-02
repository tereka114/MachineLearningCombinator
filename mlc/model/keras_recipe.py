from keras.models import Sequential,Graph
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils.visualize_util import plot
import sys
sys.setrecursionlimit(10000)

def cinput_shape(graph):
    shape = list(graph.output_shape)
    shape.pop(0)
    return shape

def built_nn3layer(input_dim=0,layer1=0,layer2=0,layer3=0,nb_classes=0,dropout1=0.0,dropout2=0.0,dropout3=0.0,init = 'glorot_normal'):
	"""
	built neural network 3 layer
	"""
	model = Sequential()
	model.add(Dense(layer1, input_dim=input_dim, init=init))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dropout1))

	model.add(Dense(layer2))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dropout2))

	model.add(Dense(layer3))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dropout3))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	return model

def built_nn3layer_regression(input_dim=0,layer1=0,layer2=0,layer3=0,nb_classes=0,dropout1=0.0,dropout2=0.0,dropout3=0.0,init = 'glorot_normal'):
	"""
	built neural network 3 layer
	"""
	model = Sequential()
	model.add(Dense(layer1, input_dim=input_dim, init=init))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dropout1))

	model.add(Dense(layer2, init=init))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dropout2))

	model.add(Dense(layer3, init=init))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dropout3))

	model.add(Dense(1, init=init))
	return model

def built_nn4layer(input_dim,layer1=0,layer2=0,layer3=0,layer4=0,nb_classes=0,dropout1=0.0,dropout2=0.0,dropout3=0.0,dropout4=0.0):
	model = Sequential()
	model.add(Dense(layer1,input_dim=input_dim))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dropout1))

	model.add(Dense(layer2))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dropout2))

	model.add(Dense(layer3))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dropout3))

	model.add(Dense(layer4))
	model.add(PReLU())
	model.add(BatchNormalization())
	model.add(Dropout(dropout4))

	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	return model

def built_rnn_predicter(input_dim):
	pass

def built_residual_network():
	pass

def built_dagcnns(input_shape,n_filter=64,n_class=100,n_conv=5):
	g = Graph()
	input_name = "input"
	g.add_input(input_name, input_shape)

	layer_names = []
	#this is all layer function

	for i in xrange(n_conv-1):
		conv_name = "conv{}".format(i)
		activate_name = "activate{}".format(i)
		average_pool = "average_pool{}".format(i)
		max_pool = "max_pool{}".format(i)
		normalize_fc = "normalize_fc{}".format(i)
		normalize_pool = "normalize_pool{}".format(i)
		flatten_name = "flatten{}".format(i)
		fc_output = "full_connect{}".format(i)

		g.add_node(Convolution2D(n_filter,3,3),name=conv_name,input=input_name)
		g.add_node(PReLU(),name=activate_name, input=conv_name)

		g.add_node(AveragePooling2D(pool_size=(2,2)),name=average_pool,input=activate_name)
		g.add_node(BatchNormalization(),name=normalize_fc,input=average_pool)
		g.add_node(Flatten(), name=flatten_name, input=normalize_fc)
		g.add_node(Dense(n_class),name=fc_output,input=flatten_name)

		g.add_node(BatchNormalization(),normalize_pool,activate_name)
		g.add_node(MaxPooling2D(pool_size=(2,2)),max_pool,normalize_pool)

		input_name = max_pool
		layer_names.append(fc_output)

	# final_stage
	conv_name = "conv{}".format(n_conv)
	activate_name = "activate{}".format(n_conv)
	average_pool = "average_pool{}".format(n_conv)
	max_pool = "max_pool{}".format(n_conv)
	normalize_fc = "normalize_fc{}".format(n_conv)
	flatten_name = "flatten{}".format(n_conv)
	fc_output = "full_connect{}".format(n_conv)

	g.add_node(Convolution2D(n_filter,3,3),name=conv_name,input=input_name)
	g.add_node(PReLU(),name=activate_name, input=conv_name)

	g.add_node(AveragePooling2D(pool_size=(2,2)),name=average_pool,input=activate_name)
	g.add_node(BatchNormalization(),name=normalize_fc,input=average_pool)
	g.add_node(Flatten(), name=flatten_name, input=normalize_fc)
	g.add_node(Dense(n_class),name=fc_output,input=flatten_name)

	layer_names.append(fc_output)

	g.add_node(Activation("softmax"),name="softmax_layer",inputs=layer_names,merge_mode="sum")
	g.add_output("output","softmax_layer")
	return g

def activation():
    return PReLU()

def ConvB(input_shape, nb_filter, nb_row, nb_col, subsample=(1, 1)):
    g = Graph()
    g.add_input("input", input_shape)
    g.add_node(Convolution2D(nb_filter,nb_row,nb_col,subsample=subsample),"conv1", "input")
    g.add_node(BatchNormalization(),"bn", "conv1")
    g.add_node(activation(),"activ", "bn")
    g.add_output("output", "activ")
    return g

def Zero(input_shape, pad=(1,1)):
    g = Graph()
    g.add_input("input", input_shape)
    g.add_node(ZeroPadding2D(pad),"zero","input")
    g.add_output("output", "zero")
    return g

def convo1(input_shape):
    g = Graph()
    g.add_input("input", input_shape)
    g.add_node(ConvB(input_shape,64,7,7, subsample=(2, 2)),"conv1", "input")
    g.add_node(MaxPooling2D((3, 3),strides=(2, 2)), "maxpool","conv1")
    g.add_output("output", "maxpool")
    return g

def avgfc(input_shape, nb_outputs):
    pooling_size = list(input_shape)
    pooling_size.pop(0)
    g = Graph()
    g.add_input("input", input_shape)
    g.add_node(AveragePooling2D(pooling_size),"avgpool","input")
    g.add_node(Flatten(), "flatten", "avgpool")
    g.add_node(Dense(nb_outputs,activation="softmax"),"fc","flatten")
    g.add_output("output", "fc")
    return g

def time_block3(input_shape, nb_filter1, nb_filter2, nb_blocks, has_edge):
    convs = []
    last_shape = input_shape
    for i in range(nb_blocks):
        is_edge = i == 0 and has_edge
        c = block3(last_shape,nb_filter1, nb_filter2, is_edge)
        last_shape = cinput_shape(c)
        convs.append(c)

    g = Graph()
    g.add_input("input", input_shape)
    last_name = "input"
    for i in range(len(convs)):
        name = "conv" + str(i)
        g.add_node(convs[i],name,last_name)
        last_name = name
    g.add_output("output",last_name)
    return g

def block3(input_shape, nb_filter1, nb_filter2, is_edge):

    zerop = (1,1)
    subsample = (1,1)

    if is_edge:
        zerop = (2,2)
        subsample = (2,2)

    g = Graph()
    g.add_input("input",input_shape)

    zero = Zero(input_shape,zerop)
    conv1 = ConvB(cinput_shape(zero),nb_filter1,1,1,subsample=subsample)
    conv2 = ConvB(cinput_shape(conv1), nb_filter1, 3, 3)
    conv3 = ConvB(cinput_shape(conv2), nb_filter2, 1,1)
    shortcut = ConvB(input_shape,nb_filter2,1,1,subsample=subsample)

    g.add_node(zero,"zero","input")
    g.add_node(conv1,"conv1","zero")
    g.add_node(conv2,"conv2","conv1")
    g.add_node(conv3,"conv3","conv2")
    g.add_node(shortcut,"shortcut","input")
    g.add_output("output", inputs=["conv3", "shortcut"], merge_mode="sum")

    return g

def get_model(input_shape, nb_conv2, nb_conv3, nb_conv4, nb_conv5, nb_outputs):
    conv1 = convo1(input_shape)

    conv2 = time_block3(cinput_shape(conv1), 64, 256, nb_conv2, False)
    conv3 = time_block3(cinput_shape(conv2), 128, 512, nb_conv3, True)
    conv4 = time_block3(cinput_shape(conv3), 256, 1024, nb_conv4, True)
    conv5= time_block3(cinput_shape(conv4), 512, 2048, nb_conv5, True)
    last = avgfc(cinput_shape(conv5), nb_outputs)

    g = Graph()
    g.add_input("input",input_shape)
    g.add_node(conv1, "conv1", "input")

    g.add_node(conv2, "conv2", "conv1")
    g.add_node(conv3, "conv3", "conv2")
    g.add_node(conv4, "conv4", "conv3")
    g.add_node(conv5, "conv5", "conv4")
    g.add_node(last, "avgfc", "conv5")

    g.add_output("output", "avgfc")

    return g

def residual_block(input_shape):
	g = Graph()
	g.add_input("input",input_shape)

def create_50_layer(input_shape, nb_outputs):
    print "create model"
    return get_model(input_shape, 3, 4, 6, 3, nb_outputs)

def create_101_layer(input_shape, nb_outputs):
    print "Create model. This could take some time..."
    return get_model(input_shape, 3, 4, 23, 3, nb_outputs)

def create_150_layer(input_shape, nb_outputs):
    print "Create model. This could take some time..."
    return get_model(input_shape, 3, 8, 36, 3, nb_outputs)


def debug_layer(model):
	pass
	# if hasattr(model, 'outputs'):
	# 	for name, l in model.outputs.items():

if __name__ == '__main__':
	#model = built_nn3layer(784,layer1=300,layer2=200,layer3=100,nb_classes=10,dropout1=0.1,dropout2=0.2,dropout3=0.3)
	# plot(model, to_file='model_nn.png')
	model = built_dagcnns((3,256,256),n_filter=64,n_class=447,n_conv=5)
	print "start_compile"
	model.compile('sgd', {'output':'categorical_crossentropy'})
	# for name, l in model.outputs.items():
	# 	child_layers = l.previous

	# 	print child_layers
	plot(model, to_file='model.png')
	# model = create_50_layer((3,255,255), 3)
	# plot(model, to_file='model_rs.png')
