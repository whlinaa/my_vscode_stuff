# %%

# import dsar # custom library to read data from cloud
# from smart_open import open 

# data science lib
# from contextlib import AsyncExitStack
import numpy as np;
import pandas as pd;
import matplotlib as mpl;
import matplotlib.pyplot as plt;
import seaborn as sns 
import datetime
import sklearn
# import tensorflow as tf
# from tensorflow import keras
# import scipy as sp
# from scipy import stats

# standard library
import os
import sys
from typing import List
import joblib, pickle # for saving model objects. joblib is suggested for sotring objects with ndarray due to faster speed
# import re # regular expression
# import itertools
# import time

# functions and classes from standard library
# from collections import Counter, defaultdict
# from functools import reduce
# from itertools import combinations 

# r=np.random.RandomState(1)

# ipython settings
# https://ipython.readthedocs.io/en/stable/config/options/terminal.html
# ‘all’, ‘last’, ‘last_expr’ or ‘none’, ‘last_expr_or_assign’ specifying which nodes should be run interactively (displaying output from expressions). Defaults to ‘last_expr’
from IPython.core.interactiveshell import InteractiveShell; InteractiveShell.ast_node_interactivity = "all"
from IPython.display import display_html

# %%
# optional packages 
# import urllib; import tarfile; # for extracting data
# import fractions
# import logging
# import subprocess
# import heapq
# %%
# optional parameter settings

# change the plot style 
plt.style.use('ggplot') # plt.style.use('seaborn') 

# plt.rcParams['font.size'] = 13 # alternatively, plt.rcParams.update({'font.size': 13})
# plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = [8, 5] # change the figure size globally. default = [6,4]

# plt.rcParams['text.usetex'] = True
# mpl.rc('axes', labelsize=14); mpl.rc('xtick', labelsize=12); mpl.rc('ytick', labelsize=12)

# pd settings
# pd.options.display.max_columns = None # None means print all columns
# pd.options.display.min_rows = None

# alternatively,
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# pd.set_option('precision', 4)
# pd.reset_option("max_columns") # to reset
# %%HTML
# HTML setting to beautify DataFrame

# <style type="text/css">
# table.dataframe td, table.dataframe th {
#     border: 1px  white solid !important; # or `white dashed`
#     # border-left: 1px  white solid !important; # or `white dashed`
#     # border-right: 1px  white solid !important;
#     # border-top: 1px  white solid !important;
#     # color: white !important;
# }
# </style>
# %%
# modelling packages
# import xgboost 
# from smart_open import open # for opening files on cloud, such as S3 
# from boruta import BorutaPy # feature selection


# ------------------ sklearn classes and functions
# pipeline
# usually, `preprocessing.FunctionTransformer` and `preprocessing.KBinsDiscretizer` are part of pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

# preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV,  RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, PolynomialFeatures, FunctionTransformer, KBinsDiscretizer

# include d2_tweedie_score
from sklearn.metrics import *

# predictor
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor, Ridge, Lasso, ElasticNet, TweedieRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier, VotingRegressor
from sklearn.svm import LinearSVC, SVC, SVR


# unsupervised learning
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# feature selection
# from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, RFE, RFECV
from sklearn.feature_selection import *

# dataset
# from sklearn.datasets import load_breast_cancer, load_digits, load_iris, make_classification,make_blobs, make_circles, make_gaussian_quantiles, load_boston, load_diabetes
# %%
def value_counts_plus(s, pct=False, sort_index=False, cum=False, index_name=None): 
    """given a Series, show the value counts and show percentage"""
    count = s.value_counts()
    rate = s.value_counts(normalize=True)
    data = {'count':count, 'rate':rate}
    if cum:
        count_cum = count.cumsum()
        rate_cum = rate.cumsum()
        data['count_cum'] = count_cum
        data['rate_cum'] = rate_cum
    if pct:
        pct = s.value_counts(normalize=True).mul(100)
        data['pct']=pct
        if cum: 
            pct_cum = pct.cumsum()
            data['pct_cum'] = pct_cum

    res = pd.concat(data.values(), axis='columns', keys=data.keys(), names='stats')
    # res = pd.concat(data.values(), axis='columns', keys=data.keys(), names='stats')
    res.index.name=index_name if index_name else "values"
    return res.sort_index() if sort_index else res

if __name__ == '__main__':
    # data = gen_data(1000,1, df=True, duplicate=True)
    value_counts_plus(data, cum=True)
# %%
# functions for modelling
def sanity_check_dataset(X_train, X_test, y_train, y_test, show_y_dist=False):
    # sanity check for dataset size
    print(f"{'X_train.shape':<13} = {X_train.shape}")
    print(f"{'X_test.shape':<13} = {X_test.shape}")
    print(f"{'y_train.shape':<13} = {y_train.shape}")
    print(f"{'y_test.shape':<13} = {y_test.shape}")

    print() # empty line

    print(f"train-test ratio = {X_train.shape[0]/X_test.shape[0]}")

    if show_y_dist:
        print(f"y_train distribution: ")
        print(value_counts_plus(pd.Series(y_train), sort_index=True))
        # value_counts_plus(pd.Series(y_train), sort_index=True)['rate'].plot.bar()

        print() # empty line

        print(f"y_test distribution: ")
        print(value_counts_plus(pd.Series(y_train), sort_index=True))

# %%
# directory management
def savefig_plus(path, dpi='figure'):
    # save figure to a path specificed in `path`
    # e.g., path = './hello/world/myfig.png'
    dir_name = os.path.dirname(path) # extract the directory name of the figure file first. e.g., dir_name = './hello/world'
    # print(dir_name)
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=dpi)

# %%
# get formatted current timestamp for saving files
def get_timestamp():
    # return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# %%
def split_by_feature(df, feature='contract_no', test_size=0.3, random_state=12345):
    """
    this function splits a panel dataset by the given `feature`
    idea: 
    - find the unique feature numbers
    - sample `test_size`*100% of the unique feature numbers
    - split the df into two portions according to the results from previous step 
    """ 
    if feature:
        test_set_feature_no = df[feature].drop_duplicates().sample(frac=test_size, random_state=random_state).values # sample features (not rows). It contains the feature number for test set.
        # print(test_set_feature_no)

        # a mask such that it's True if the index is in test_set_feature_no
        test_set_index_filt = df[feature].isin(test_set_feature_no)
        # print(test_set_index_filt)

        df_test = df[test_set_index_filt].reset_index(drop=True)
        df_train = df[~test_set_index_filt].reset_index(drop=True)

    else:
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_train, df_test
# %%
def find_extreme(s, low_q=None, high_q=None, ignore_zero=True):
    # this function is to find records with the lowest values and highest values, according to quantiles.
    # can optionally ignore zero values
    # either low_q or high_q can be left unspecific, in which case, we filter only one extreme of the data
    if ignore_zero:
        s = s.replace(0, np.nan)
    if low_q:
        lower_lim = s.quantile(low_q)
    if high_q:
        upper_lim = s.quantile(high_q)
    if low_q is None:
        return s.gt(upper_lim)
    if high_q is None:
        return s.lt(lower_lim)
    return s.gt(upper_lim) | s.lt(lower_lim)
# %%
def get_feature_importance(model, selected_features):
    # model: a sklearn model object or a sklearn CV object
    # selected_features: list[str]
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_ , index=selected_features, name='feature importancce').sort_values(ascending=False)
    elif hasattr(model.best_estimator_, 'feature_importances_'):
        return pd.Series(model.best_estimator_.feature_importances_ , index=selected_features, name='feature importancce').sort_values(ascending=False)
    else:
        print("the model does not have `feature_importances_` attribute...")
# %%
def undersample(X, y, ratio, majority_name):
    """	undersample the majority class 
        ratio: 	number of times majority class is larger than the minority class. 
                ratio = 1 ==> (majority size == minority size)
    """
    X_combine = pd.concat([pd.DataFrame(X), y], axis = 1)

    minority_class = X_combine[y.name]!=majority_name
    majority_class = X_combine[y.name]==majority_name
    
    X_minor = X_combine[minority_class] # get the examples in minority class
    X_major = X_combine[majority_class].sample(n=X_minor.shape[0]*ratio, random_state=1234) # get the examples in majority class

    X_combine = pd.concat([X_minor, X_major])

    return X_combine.drop(y.name, axis=1).values, X_combine[y.name]
# %%
# def undersample(X, y, ratio, majority_name):
#     """	undersample the majority class 
#         ratio: 	number of times majority class is larger than the minority class. 
#                 ratio = 1 ==> (majority size == minority size)
#     """
#     majority_class = y[y==majority_name].index
#     minority_class = y[y!=majority_name].index
#     random_indices = np.random.choice(majority_class, size=len(minority_class)*ratio, replace=False)
#     X_sample = X[random_indices]
#     y_sample = y[random_indices]
#     # print(majority_class)


# undersample(X_train_with_exposure, df_train['has_cnt'], 3, False)

# # X_train, y_train = undersample(X_train_with_exposure, df_train['has_cnt'], 3, False)



# # print("after undersampling...\n")
# # sanity_check(X_train, X_test, y_train, y_test)
# %%
def find_loss_ratio_by_gp(df, by_feat=None, mean=True):
    if mean:
        return df.groupby(by_feat).loss_ratio.mean()
    else:
        return df.groupby(by_feat).apply(lambda df: df.tot_pay_amt.sum()/((df.annual_prem*df.exposure).sum()))
# %%
# functions to combine records
def get_first_k_years(df, k=5, strict=False):
    # get first k years of each record
    # NOTE: we get it only if the first k years have a sum of k exposure. 
    # e.g., if a contract lasts for less than k years, it won't be selected
    # e.g., if a contract last for k years, not in kth year, the exposure isn't 1, it won't be selected
    # e.g., if a contract has some years in the first k years removed (e.g., due to filtering), then 
    df_first_k = df.query(f'year_seq<={k}')
    if strict:
        useful_indices = df_first_k.groupby('contract_no').exposure.sum()[lambda exposure: exposure==k].index
    else:
        useful_indices = df_first_k.query(f'year_seq=={k}').contract_no.values
    return df_first_k[df_first_k.contract_no.isin(useful_indices)]

def combine_records(df):
    # return a new df
    df_temp = df.copy()
    df_temp['eff_annual_prem'] = df.annual_prem * df.exposure
    df_by_contracts_info = (df_temp.groupby('contract_no')[['year_seq','exposure','tot_pay_amt', 'tot_cnt', 'eff_annual_prem','has_clm']].aggregate({
        'exposure':'sum',
        'tot_pay_amt':'sum', 
        'tot_cnt':'sum', 
        'eff_annual_prem':'sum',
        'has_clm':'max',
        # 'year_seq':'max', # not needed, since we select the last unique record, and year_seq is automatically maximized
    })
    .assign(
        loss_ratio = lambda df: df['tot_pay_amt']/df['eff_annual_prem'],
        frequency = lambda df: df['tot_cnt']/df['exposure'],
        severity = lambda df: df['loss_ratio']/np.fmax(df['frequency'], 1e-10), # IMPORTANT: we need to give a nonzero value if df['frequency'] == 0, otherwise we have division of zero error.
        # severity = lambda df: df['loss_ratio']/df['frequency'], # this will not work!!
        has_clm = lambda df: df['has_clm']>0,
        )
    )

    df_by_contracts_info.drop(['eff_annual_prem'], axis=1, inplace=True)
    # print(df_by_contracts_info)

    df_unique = df_temp.drop_duplicates('contract_no', keep='last').set_index('contract_no')
    # print(df_unique)

    df_unique_drop = df_unique.drop(['exposure','tot_pay_amt','loss_ratio','frequency', 'severity', 'has_clm', 'eff_annual_prem','first_year','tot_cnt'], axis='columns')

    df_final = pd.concat([df_unique_drop, df_by_contracts_info],axis='columns')

    return df_final.reset_index()
# %%

def output_table(col_gp, df, y_true, y_pred):
    # TODO: rewrite using `pd.concat([df_output.reset_index(drop=True),pd.DataFrame(y_pred_test)], axis=1)`
    # df[col_gp]: the column to be exported to db
    # y_true: ground truth
    # y_pred: prediction. it is a dictionary! 
    df_output = df[col_gp].copy()
    df_output['y_true'] = y_true

    for model_name in model_names:
        df_output[model_name] = y_pred[model_name]
    
    return df_output