import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import XGBoost
import pickle
from sklearn.decomposition import PCA
import Util
import FeatureUtil

"""
original
"""
class FeatureVectorData(FeatureUtil.FeatureVector):
    def __init__(self):
        self.name = "feature_vector"
        self.list = None

    def convertFileToFeature(self):
        train = pd.read_csv('../data/training.csv', parse_dates=[2, ])
        test = pd.read_csv('../data/test.csv', parse_dates=[3, ])
        features = list(train.columns[1:-5])
        labels = train["signal"]

        return np.array(train[features]).astype(np.float32),np.array(labels).astype(np.float32),np.array(test[features]).astype(np.float32)

if __name__ == '__main__':
    feature_vector = FeatureVectorData()
    train, labels, test = feature_vector.getVector(std=False,one_of_k=False,label_base=False,pca=False,MinMaxScaler=True)
    filename = feature_vector.getFeatureName()
    pickle.dump((train, labels, test), open("feature_vector/" + filename, "w"))
