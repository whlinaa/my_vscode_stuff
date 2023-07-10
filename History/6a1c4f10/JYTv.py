# %%
print("calling ds_preamble..")
# %%
# ----------------------------- standard library  ---------------------------- #
import os
import joblib, pickle # for saving model objects. joblib is suggested for storing objects with ndarray due to faster speed
# import pathlib
import re # regular expression
import datetime
import time


# ---------------- functions and classes from standard library --------------- #
from typing import Any, Optional, Iterator, TypeVar
from glob import glob
from collections import Counter, defaultdict
from pathlib import Path
from scipy import stats
# from functools import reduce
# from itertools import combinations 

# ----------------------------- optional packages ---------------------------- #
# import urllib; import tarfile; # for extracting data
# import fractions
# import logging
# import subprocess
# import heapq
# import shutil # for copy, move etc
# import sys
# import subprocess
# import itertools

# ---------------------------- 3rd party packages ---------------------------- #
import numpy as np
import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# import duckdb

# import tensorflow as tf
# from tensorflow import keras
# import scipy as sp
# import xgboost 
# from boruta import BorutaPy # feature selection
# ---------------------------------------------------------------------------- #

# %%
# ------------------------------ sklearn-related ----------------------------- #

# pipeline
# usually, `preprocessing.FunctionTransformer` and `preprocessing.KBinsDiscretizer` are part of pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer, make_column_transformer
# from sklearn.pipeline import Pipeline, make_pipeline

# preprocessing
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV,  RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer, PolynomialFeatures, FunctionTransformer, KBinsDiscretizer

# include d2_tweedie_score
from sklearn.metrics import *

# predictor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet # ,SGDClassifier, SGDRegressor, TweedieRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
# from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier # , VotingClassifier, VotingRegressor
# from sklearn.svm import LinearSVC, SVC, SVR

# unsupervised learning
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# feature selection
# from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, RFE, RFECV
from sklearn.feature_selection import *

# dataset
# from sklearn.datasets import load_breast_cancer, load_digits, load_iris, make_classification,make_blobs, make_circles, make_gaussian_quantiles, load_boston, load_diabetes
# ---------------------------------------------------------------------------- #

# %%
# ----------------------------- ipython settings ----------------------------- #
# https://ipython.readthedocs.io/en/stable/config/options/terminal.html
# ‘all’, ‘last’, ‘last_expr’ or ‘none’, ‘last_expr_or_assign’ specifying which nodes should be run interactively (displaying output from expressions). Defaults to ‘last_expr’
from IPython.core.interactiveshell import InteractiveShell; InteractiveShell.ast_node_interactivity = "all"
from IPython.display import display_html
# ---------------------------------------------------------------------------- #

# %%
# ---------------------------- matplotlib setting ---------------------------- #
# change the plot style 
plt.style.use('ggplot') 
sns.set_style("whitegrid")
# plt.style.use(['science','grid'])
# plt.style.use(['grid'])

# plt.style.use('seaborn') 
# plt.style.use("seaborn-whitegrid")

# mpl.rcParams['axes.grid'] = True
# mpl.rcParams['grid.color'] = '.8'
# plt.rcParams['axes.facecolor'] = 'white'


plt.rcParams['font.size'] = 11 # alternatively, plt.rcParams.update({'font.size': 13})
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.title_fontsize'] = 11
plt.rcParams['figure.figsize'] = [6, 4] # change the figure size globally. default = [6,4]

# plt.rcParams['text.usetex'] = True
# mpl.rc('axes', labelsize=14); mpl.rc('xtick', labelsize=12); mpl.rc('ytick', labelsize=12)

# pd settings
# pd.options.display.max_columns = None # None means print all columns

# The numbers of rows to show in a truncated view
# pd.options.display.min_rows = None # set 30 to show 30 rows in truncated view

# If max_rows is exceeded, switch to truncate view
# pd.options.display.max_rows = 10

# alternatively,
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# pd.set_option('precision', 4)
# pd.reset_option("max_columns") # to reset
# ---------------------------------------------------------------------------- #
# %%
# ------------------------ pd.DataFrame output setting ----------------------- #
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
# ---------------------------------------------------------------------------- #

# %%
# ----------------------------- cloud packages ----------------------------- #
# import dsar # custom library to read data from cloud
# from smart_open import open # for opening files on cloud, such as S3 

