# -*- coding: utf-8 -*-
 
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import xgb_custom

class XGBoostWrapper(object):
    def __init__(self,num_boost_round=5, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
    
    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self

class XGBoostClassifier(XGBoostWrapper):
    def __init__(self, num_boost_round=10, **params):
        super(XGBoostClassifier,self).__init__(num_boost_round,**params)
 
    def fit(self, X, y,num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))

        early_stopping = False
        if early_stopping == True:
            xg_train,xg_validate,xg_train_y,xg_validate_y = train_test_split(X,y,test_size=0.2)

            print self.params

            if self.params["objective"] == "binary:logistic":
                print "binary:logistic"
                dtrain = xgb.DMatrix(xg_train, label=xg_train_y)
                dvalid = xgb.DMatrix(xg_validate, label=xg_validate_y)
            else:
                dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in xg_train_y])
                dvalid = xgb.DMatrix(X, label=[self.label2num[label] for label in xg_validate_y])
            #evallist  = [(dtrain,'train')]

            watchlist = [(dtrain,'train'),(dvalid,'val')]
            self.clf = xgb.train(self.params, dtrain, num_boost_round,watchlist,early_stopping_rounds=80)
        else:
            xg_train,xg_train_y = X,y
            if self.params["objective"] == "binary:logistic":
                print "binary:logistic"
                dtrain = xgb.DMatrix(xg_train, label=xg_train_y)
                watchlist = [(dtrain,'train')]
                self.clf = xgb.train(self.params, dtrain, num_boost_round,watchlist)
            else:
                dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in xg_train_y])
                watchlist = [(dtrain,'train')]
                self.clf = xgb.train(self.params, dtrain, num_boost_round,watchlist)

    def predict(self, X):
        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

class XGBoostRegressor(XGBoostWrapper):
    def __init__(self,num_boost_round=10, **params):
        super(XGBoostRegressor,self).__init__(num_boost_round,**params)

    def fit(self, X, y,num_boost_round=None,early_stopping=True):
        transform_y = y
        num_boost_round = num_boost_round or self.num_boost_round
        #evallist  = [(dtrain,'train')]

        if early_stopping == True:
            xg_train,xg_validate,xg_train_y,xg_validate_y = train_test_split(X,transform_y,test_size=0.012)
            dtrain = xgb.DMatrix(xg_train, label=xg_train_y)
            dvalid = xgb.DMatrix(xg_validate, label=xg_validate_y)
            #evallist  = [(dtrain,'train')]

            watchlist = [(dtrain,'train'),(dvalid,'val')]
            #self.clf = xgb.train(self.params, dtrain, num_boost_round=num_boost_round,evals=watchlist,early_stopping_rounds=100,feval=xgb_custom.rmspe_evaluation)
            self.clf = xgb.train(self.params, dtrain, num_boost_round=num_boost_round,evals=watchlist,early_stopping_rounds=100)
        else:
            dtrain = xgb.DMatrix(X, label=y)
            watchlist = [(dtrain,'train')]
            self.clf = xgb.train(self.params, dtrain, num_boost_round,watchlist,feval=xgb_custom.rmspe_evaluation)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)