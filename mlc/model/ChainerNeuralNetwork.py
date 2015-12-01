# coding:utf-8
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
import numpy as np
from sklearn.cross_validation import train_test_split
from ..utility.evaluation_functions import evaluate_function


class NeuralNetwork(object):
    def __init__(self):
        pass

class ChainerNeuralNetworkModel(chainer.Chain):
    def __init__(self, problem_type='regression',n_in=0,n_out=1,layer1=10,layer2=20,layer3=10,dropout1=0.2,dropout2=0.3,dropout3=0.1):
        self.problem_type = problem_type
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        super(ChainerNeuralNetworkModel,self).__init__(
            l1=L.Linear(n_in, layer1),
            l2=L.Linear(layer1, layer2),
            l3=L.Linear(layer2, layer3),
            l4=L.Linear(layer3, n_out)
            )

    def __call__(self, x, t):
        h1 = F.dropout(F.relu(self.l1(x)),ratio=self.dropout1)
        h2 = F.dropout(F.relu(self.l2(h1)),ratio=self.dropout2)
        h3 = F.dropout(F.relu(self.l3(h2)),ratio=self.dropout3)        
        h4 = self.l4(h3)

        if self.problem_type == 'classifier':
            self.loss = F.softmax_cross_entropy(h4, t)
            return self.loss
        elif self.problem_type == 'regression':
            self.loss = F.mean_squared_error(h4, t)
            return self.loss

    def predict(self,x):
        h1 = F.dropout(F.relu(self.l1(x)),ratio=self.dropout1,train=False)
        h2 = F.dropout(F.relu(self.l2(h1)),ratio=self.dropout2,train=False)
        h3 = F.dropout(F.relu(self.l3(h2)),ratio=self.dropout3,train=False)
        h4 = self.l4(h3)

        return h4

class ChainerNeuralNetwork(object):
    def __init__(self, batch_size=100, cuda=False, epoch=100, problem_type='classifier',model=None,seed=2015,evaluate_function_name=None,convert=None):
        self.batchsize = batch_size
        self.cuda = cuda
        self.epochs = epoch
        self.problem_type = problem_type
        self.model = model
        self.seed = seed
        self.evaluate_function_name = evaluate_function_name
        self.convert_type = None

    def convert(self,data):
        if self.convert == "log":
            return np.log1p(data)
        else:
            return data

    def reconvert(self,data):
        if self.convert == "log":
            return np.expm1(data)
        return data

    def fit(self,x_train,y_train):
        N = len(x_train)
        batchsize = self.batchsize

        np.random.seed(self.seed)

        if self.cuda:
            cuda.get_device(0).use()
            self.model.to_gpu()
            xp = cuda.cupy
        else:
            xp = np
        self.xp = xp

        data = np.array(x_train,dtype=np.float32)

        if self.problem_type == 'classifier':
            target = y_train
        elif self.problem_type == 'regression':
            target = np.array(self.convert(y_train),dtype=np.float32).reshape((len(data),1))

        optimizer = optimizers.Adam()
        optimizer.setup(self.model)

        for epoch in xrange(self.epochs):
            print "epoch:",epoch
            perm = np.random.permutation(N)
            sum_loss = 0.0
            for i in xrange(0,N,batchsize):
                x = chainer.Variable(xp.asarray(data[perm[i:i + batchsize]]),volatile="off")
                t = chainer.Variable(xp.asarray(target[perm[i:i + batchsize]],dtype=np.float32),volatile="off")

                optimizer.update(self.model, x, t)

                if evaluate_function != None:
                    prediction = self.predict(x.data)
                    loss = evaluate_function(target[perm[i:i + batchsize]],prediction,self.evaluate_function_name)

                    sum_loss += loss * len(t)
            print "train_loss:{}".format(sum_loss / N)

    def predict(self,data):
        x = chainer.Variable(self.xp.asarray(data,dtype=np.float32),volatile="on")
        y = self.model.predict(x)
        if self.problem_type == 'regression':
            return self.reconvert(y.data.get().reshape((len(data))))
        else:
            return np.argmax()

    def predict_proba(x):
        pass
