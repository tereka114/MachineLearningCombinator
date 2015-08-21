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
        train = pd.read_csv('../data/train_set.csv', parse_dates=[2, ])
        test = pd.read_csv('../data/test_set.csv', parse_dates=[3, ])
        tube_data = pd.read_csv('../data/tube.csv')
        bill_of_materials_data = pd.read_csv('../data/bill_of_materials.csv')
        specs_data = pd.read_csv('../data/specs.csv')

        print("train columns")
        print(train.columns)
        print("test columns")
        print(test.columns)
        print("tube.csv df columns")
        print(tube_data.columns)
        print("bill_of_materials.csv df columns")
        print(bill_of_materials_data.columns)
        print("specs.csv df columns")
        print(specs_data.columns)

        print(specs_data[2:3])

        train = pd.merge(train, tube_data, on='tube_assembly_id')
        train = pd.merge(train, bill_of_materials_data, on='tube_assembly_id')
        test = pd.merge(test, tube_data, on='tube_assembly_id')
        test = pd.merge(test, bill_of_materials_data, on='tube_assembly_id')

        print("new train columns")
        print(train.columns)
        print(train[1:10])
        print(train.columns.to_series().groupby(train.dtypes).groups)

        # create some new features
        train['year'] = train.quote_date.dt.year
        train['month'] = train.quote_date.dt.month

        test['year'] = test.quote_date.dt.year
        test['month'] = test.quote_date.dt.month

        # drop useless columns and create labels
        idx = test.id.values.astype(int)
        test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis=1)
        labels = train.cost.values
        # 'tube_assembly_id', 'supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x'
        # for some reason material_id cannot be converted to categorical variable
        train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis=1)

        train['material_id'].replace(np.nan, ' ', regex=True, inplace=True)
        test['material_id'].replace(np.nan, ' ', regex=True, inplace=True)
        for i in range(1, 9):
            column_label = 'component_id_' + str(i)
            print(column_label)
            train[column_label].replace(np.nan, ' ', regex=True, inplace=True)
            test[column_label].replace(np.nan, ' ', regex=True, inplace=True)

        train.fillna(0, inplace=True)
        test.fillna(0, inplace=True)

        print("train columns")
        print(train.columns)

        # convert data to numpy array
        train = np.array(train)
        test = np.array(test)
        return train,labels,test

def feature_vector():
    train = pd.read_csv('../data/train_set.csv', parse_dates=[2, ])
    test = pd.read_csv('../data/test_set.csv', parse_dates=[3, ])

    tubes = pd.read_csv('../data/tube.csv')

    # create some new features
    train['year'] = train.quote_date.dt.year
    train['month'] = train.quote_date.dt.month
    train['dayofyear'] = train.quote_date.dt.dayofyear
    train['dayofweek'] = train.quote_date.dt.dayofweek
    train['day'] = train.quote_date.dt.day

    test['year'] = test.quote_date.dt.year
    test['month'] = test.quote_date.dt.month
    test['dayofyear'] = test.quote_date.dt.dayofyear
    test['dayofweek'] = test.quote_date.dt.dayofweek
    test['day'] = test.quote_date.dt.day

    train = pd.merge(train, tubes, on='tube_assembly_id', how='inner')
    test = pd.merge(test, tubes, on='tube_assembly_id', how='inner')

    train['material_id'].fillna('SP-9999', inplace=True)
    test['material_id'].fillna('SP-9999', inplace=True)

    # drop useless columns and create labels
    test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis=1)
    labels = train.cost.values
    train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis=1)

    # convert data to numpy array
    train = np.array(train)
    test = np.array(test)

    # label encode the categorical variables
    for i in range(train.shape[1]):
        if i in [0, 3, 10, 16, 17, 18, 19, 20, 21]:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[:, i]) + list(test[:, i]))
            train[:, i] = lbl.transform(train[:, i])
            test[:, i] = lbl.transform(test[:, i])

    # object array to float
    X_train = train.astype(float)
    X_test = test.astype(float)

    return X_train, labels, X_test


def feature_vector2():
    train = pd.read_csv('../data/train_set.csv', parse_dates=[2, ])
    test = pd.read_csv('../data/test_set.csv', parse_dates=[3, ])
    tube_data = pd.read_csv('../data/tube.csv')
    bill_of_materials_data = pd.read_csv('../data/bill_of_materials.csv')
    specs_data = pd.read_csv('../data/specs.csv')

    print("train columns")
    print(train.columns)
    print("test columns")
    print(test.columns)
    print("tube.csv df columns")
    print(tube_data.columns)
    print("bill_of_materials.csv df columns")
    print(bill_of_materials_data.columns)
    print("specs.csv df columns")
    print(specs_data.columns)

    print(specs_data[2:3])

    train = pd.merge(train, tube_data, on='tube_assembly_id')
    train = pd.merge(train, bill_of_materials_data, on='tube_assembly_id')
    test = pd.merge(test, tube_data, on='tube_assembly_id')
    test = pd.merge(test, bill_of_materials_data, on='tube_assembly_id')

    print("new train columns")
    print(train.columns)
    print(train[1:10])
    print(train.columns.to_series().groupby(train.dtypes).groups)

    # create some new features
    train['year'] = train.quote_date.dt.year
    train['month'] = train.quote_date.dt.month
    # train['dayofyear'] = train.quote_date.dt.dayofyear
    # train['dayofweek'] = train.quote_date.dt.dayofweek
    # train['day'] = train.quote_date.dt.day

    test['year'] = test.quote_date.dt.year
    test['month'] = test.quote_date.dt.month
    # test['dayofyear'] = test.quote_date.dt.dayofyear
    # test['dayofweek'] = test.quote_date.dt.dayofweek
    # test['day'] = test.quote_date.dt.day

    # drop useless columns and create labels
    idx = test.id.values.astype(int)
    test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis=1)
    labels = train.cost.values
    # 'tube_assembly_id', 'supplier', 'bracket_pricing', 'material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x'
    # for some reason material_id cannot be converted to categorical variable
    train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis=1)

    train['material_id'].replace(np.nan, ' ', regex=True, inplace=True)
    test['material_id'].replace(np.nan, ' ', regex=True, inplace=True)
    for i in range(1, 9):
        column_label = 'component_id_' + str(i)
        print(column_label)
        train[column_label].replace(np.nan, ' ', regex=True, inplace=True)
        test[column_label].replace(np.nan, ' ', regex=True, inplace=True)

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    print("train columns")
    print(train.columns)

    # convert data to numpy array
    train = np.array(train)
    test = np.array(test)


    # label encode the categorical variables
    for i in range(train.shape[1]):
        if i in [0, 3, 5, 11, 12, 13, 14, 15, 16, 20, 22, 24, 26, 28, 30, 32, 34]:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[:, i]) + list(test[:, i]))
            train[:, i] = lbl.transform(train[:, i])
            test[:, i] = lbl.transform(test[:, i])

    # object array to float
    train = train.astype(float)
    test = test.astype(float)

    return train, labels, test


def feature_vector2_std():
    train, labels, test = feature_vector2()
    std = preprocessing.StandardScaler()
    std.fit(train)
    train = std.transform(train)
    test = std.transform(test)

    print train, test

    return train, labels, test


def cross_inspect_feature():
    train, labels, test = original_data()

    train = np.array(train)
    test = np.array(test)

    parameter = {
        "alpha": 4.8e-06,
        "eta": 0.02,
        "model": "XGBREGLINEAR",
        "num_round": 10000,
        "objective": "reg:linear",
        "colsample_bytree": 0.7,
        "max_depth": 9,
        "min_child_weight": 8,
        "scale_pos_weight": 1,
        "subsample": 0.7
    }
    clf = Util.model_select(parameter)
    clf.fit(train, np.log1p(labels))

    pred_train = clf.predict(train).reshape((len(train)), 1)
    pred_test = clf.predict(test).reshape((len(test)), 1)

    train_s = np.hstack((train, pred_train))
    test_s = np.hstack((test, pred_test))

    return train_s, labels, test_s


def cross_inspect_feature_v2():
    train, labels, test = feature_vector2()

    train = np.array(train)
    test = np.array(test)

    parameter = {
        "alpha": 4.8e-06,
        "eta": 0.02,
        "model": "XGBREGLINEAR",
        "num_round": 10000,
        "objective": "reg:linear",
        "colsample_bytree": 0.7,
        "max_depth": 9,
        "min_child_weight": 8,
        "scale_pos_weight": 1,
        "subsample": 0.7
    }
    clf = Util.model_select(parameter)
    clf.fit(train, np.log1p(labels))

    pred_train = clf.predict(train).reshape((len(train)), 1)
    pred_test = clf.predict(test).reshape((len(test)), 1)

    train_s = np.hstack((train, pred_train))
    test_s = np.hstack((test, pred_test))

    return train_s, labels, test_s


if __name__ == '__main__':
    # function_list = [feature_vector,feature_vector_drop5,random_forest_selector,cross_inspect_feature]
    # function_name = ["feature_vector.pkl","feature_vector_drop5.pkl","random_forest_selector.pkl","cross_inspect_feature.pkl"]
    feature_vector = FeatureVectorData()
    train, labels, test = feature_vector.getVector(std=True,one_of_k=True,label_base=False,pca=False)
    filename = feature_vector.getFeatureName()
    pickle.dump((train, labels, test), open("feature_vector/" + filename, "w"))
