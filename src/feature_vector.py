import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import XGBoost
import pickle
from sklearn.decomposition import PCA
import Util
import FeatureUtil
from sklearn.preprocessing import LabelEncoder
from csv import DictReader
"""
original
"""
class FeatureVectorData(FeatureUtil.FeatureVector):
    def __init__(self):
        self.name = "feature_vector"
        self.list = None

    def convertFileToFeature(self):
        print ("train reading")
        df = pd.read_csv("../data/train.csv")
        encoder = LabelEncoder()
        labels = df.target
        labels = encoder.fit_transform(labels).astype(np.int32)

        df = df.drop('target',1)
        df = df.drop('ID',1)
        
        # Junk cols - Some feature engineering needed here
        df = df.ix[:, 520:660].fillna(-1)

        train = df.values.copy()

        print ("test reading")
        test_df = pd.read_csv("../data/test.csv")
        test_df = test_df.drop('ID',1)
        
        # Junk cols - Some feature engineering needed here
        test_df = test_df.ix[:, 520:660].fillna(-1)
        test = test_df.values.copy()

        return np.array(train).astype(np.float32),np.array(labels).astype(np.float32),np.array(test).astype(np.float32)

class FeatureVectorDataNew(FeatureUtil.FeatureVector):
    def __init__(self):
        self.name = "feature_vector_new"
        self.list = None

    def convertFileToFeature(self):
        train = pd.read_csv("../data/train.csv")
        mixCol = [8,9,10,11,12,18,19,20,21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 43, 44, 45, 
                  73, 74, 98, 99, 100, 106, 107, 108, 156, 157, 158, 159, 166, 167, 168, 169, 176, 177, 178, 179, 180, 
                  181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 202, 205, 206, 207, 
                  208, 209, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 240, 371, 372, 373, 374,
                  375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 
                  396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 
                  437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457,
                  458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
                  479, 480, 481, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509,
                  510, 511, 512, 513, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 840]

        #Columns with logical datatype
        alphaCol = [283, 305, 325, 352, 353, 354, 1934]

        #Columns with Places as entries
        placeCol = [200, 274, 342]

        #Columns with timestamps
        dtCol = [75, 204, 217]

        selectColumns = []
        rmCol = mixCol+alphaCol+placeCol+dtCol
        for i in range(1,1935):
            if i not in rmCol:
                selectColumns.append(i)

        cols = [str(n).zfill(4) for n in selectColumns]
        strColName = ['VAR_' + strNum for strNum in cols] 

        # Use only required columns
        nrows = 500
        train = pd.read_csv("../data/train.csv", skiprows=[107], usecols=strColName)
        label = pd.read_csv("../data/train.csv", skiprows=[107], usecols=['target'])

class FeatureVectorHashTrick(FeatureUtil.FeatureVector):
    def __init__(self):
        self.name = "hash_trick"
        self.list = None

    def convertFileToFeature(self):
        alpha = .005    # learning rate
        beta = 1        
        L1 = 0.         # L1 regularization, larger value means more regularized
        L2 = 0.         # L2 regularization, larger value means more regularized

        # C, feature/hash trick
        D = 2 ** 24             # number of weights to use
        interaction = False     # whether to enable poly2 feature interactions

        train_path='../data/train.csv'
        test_path='../data/test.csv'

        df = pd.read_csv("../data/train.csv")
        encoder = LabelEncoder()
        labels = df.target
        labels = encoder.fit_transform(labels).astype(np.int32)

        def data(path, D):
            ''' GENERATOR: Apply hash-trick to the original csv row
                           and for simplicity, we one-hot-encode everything

                INPUT:
                    path: path to training or testing file
                    D: the max index that we can hash to

                YIELDS:
                    ID: id of the instance, mainly useless
                    x: a list of hashed and one-hot-encoded 'indices'
                       we only need the index since all values are either 0 or 1
                    y: y = 1 if we have a click, else we have y = 0
            '''
            x_array = []
            time = 0
            count_time = 0
            for t, row in enumerate(DictReader(open(path), delimiter=',')):
                time += 1
                try:
                    ID=row['ID']
                    del row['ID']
                except:
                    pass
                # process clicks
                y = 0.
                target='target'#'IsClick' 
                if target in row:
                    if row[target] == '1':
                        y = 1.
                    del row[target]
                x = []
                for key in row:
                    value = row[key]

                    # one-hot encode everything with hash trick
                    index = abs(hash(key + '_' + value)) % D
                    x.append(index)
                count_time += 1
                x_array.append(x)
            print time,count_time
            return np.array(x_array)
        train = data(train_path,D)
        test = data(test_path,D)
        return np.array(train).astype(np.float32),np.array(labels).astype(np.float32),np.array(test).astype(np.float32)

class FeatureVectorHashTrick(FeatureUtil.FeatureVector):
    def __init__(self):
        self.name = "str_label"
        self.list = None

    def convertFileToFeature(self):
        print ("train reading")
        df = pd.read_csv("../data/train.csv")
        encoder = LabelEncoder()
        labels = df.target
        labels = encoder.fit_transform(labels).astype(np.int32)

        df = df.drop('target',1)
        df = df.drop('ID',1)
        
        # Junk cols - Some feature engineering needed here
        df = df.fillna(-1)

        train = df.values.copy()

        print ("test reading")
        test_df = pd.read_csv("../data/test.csv")
        test_df = test_df.drop('ID',1)
        
        # Junk cols - Some feature engineering needed here
        test_df = test_df.fillna(-1)
        test = test_df.values.copy()

        return np.array(train),np.array(labels).astype(np.float32),np.array(test)

if __name__ == '__main__':
    feature_vector = FeatureVectorHashTrick()
    train, labels, test = feature_vector.getVector(std=False,one_of_k=False,label_base=True,pca=False,MinMaxScaler=False)
    filename = feature_vector.getFeatureName()
    pickle.dump((train, labels, test), open("feature_vector/" + filename, "w"))
