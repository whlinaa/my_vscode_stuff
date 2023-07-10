# %%
import dsar # custom library to read data from cloud
from smart_open import open

# data science lib
# from contextlib import AsyncExitStack
import numpy as np;
import pandas as pd;
import matplotlib as mpl;
import matplotlib.pyplot as plt;
import seaborn as sns
import datetime
import sklearn
# from ds_utils.ds_preamble import *
from ds_preamble import *
from re import sub
import xgboost
from skopt import BayesSearchCV
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor
from scipy import interpolate
# %%
print("calling ds_helper...")
# %%

def load_table(
    schema: str, table: str, columns: list=None,
    chunksize: int=None, debug: bool=False, where=None,
    sample_size=None
    ):
    """query database. Return a DataFrame

    Returns:
        DataFrame: a DataFrame containing the matched records
    """

    columns = ', '.join(columns) if columns else '*'

    if sample_size:
        sql = f"""
            SELECT setseed(0.5);
            -- sample some contract_no
            WITH tiny_contract_no AS (
                SELECT DISTINCT contract_no FROM {schema}.{table} WHERE random()<{sample_size}
            ),
            -- get the records with the sampled contract_no
            new_tab AS(
                SELECT * FROM {schema}.{table} WHERE contract_no IN (SELECT contract_no FROM tiny_contract_no) 
                ORDER BY contract_no, year_seq
            )
            SELECT {columns} 
            FROM new_tab
            """
    else:
        sql = f"select {columns} from {schema}.{table}"

    if where:
        sql+=f" where {where}"

    if debug:
        print(sql) # sanity check

    # read data from DB
    with dsar.psql_con("WRITE") as con:
        return pd.read_sql(con=con, sql=sql, chunksize=chunksize)

# %%
def execute_sql(sql, connection='WRITE'):
    # this function simply executes the given sql
    with dsar.psql_con(connection) as con:
        return pd.read_sql(con=con, sql=sql)
# %%
def value_counts_plus(s, pct: bool=False, sort_index: bool=False, cum=False, index_name: str=None):
    """given a Series, show the value counts and show percentage"""
    # print("hello")
    count = s.value_counts().sort_index()
    rate = s.value_counts(normalize=True).sort_index()
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

pd.Series.value_counts_plus = value_counts_plus # monkey patching, so that we can call the function as method

# %%
# functions for modelling
def sanity_check_dataset(X_train, X_test, y_train, y_test, show_y_dist=False):
    # sanity check for dataset size
    print(f"{'X_train.shape':<13} = {X_train.shape}")
    print(f"{'X_test.shape':<13} = {X_test.shape}")
    print(f"{'y_train.shape':<13} = {y_train.shape}")
    print(f"{'y_test.shape':<13} = {y_test.shape}")
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
def savefig_plus(path: str, dpi: int='figure'):
    # save figure to a path specificed in `path`
    # e.g., path = './hello/world/myfig.png'
    # dir_name = os.path.dirname(path)
    # print(dir_name)
    # os.makedirs(dir_name, exist_ok=True)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True) # extract the directory name of the figure file first. e.g., dir_name = './hello/world', and then create the directories as needed
    plt.savefig(path, bbox_inches='tight', dpi=dpi)

# %%
# get formatted current timestamp for saving files
def get_timestamp(microsecond=False, microsecond_only=False):
    # return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if microsecond_only:
        format = "%f"
    else:
        format = "%Y%m%d_%H%M%S_%f" if microsecond else "%Y%m%d_%H%M%S"

    return datetime.datetime.now().strftime(format)
    # return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") if microsecond else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# %%
def split_by_feature(df, feature: str='contract_no', test_size: float=0.3, random_state: int=12345):
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
def find_extreme(s, low_q: float=None, high_q: float=None, ignore_zero: bool=True):
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
def get_feature_importance(model, selected_features: str):
    # model: a sklearn model object or a sklearn CV object
    # selected_features: list[str]
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_ , index=selected_features, name='feature importance').sort_values(ascending=False)
    elif hasattr(model.best_estimator_, 'feature_importances_'):
        return pd.Series(model.best_estimator_.feature_importances_ , index=selected_features, name='feature importance').sort_values(ascending=False)
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
def find_loss_ratio(df):
    d = {}
    d['agg_lr'] = df.tot_pay_amt.sum()/df.eff_annual_prem.sum()
    d['mean_lr'] = df.loss_ratio.mean()
    return pd.Series(d)


def find_loss_ratio_by_gp(df, by=None, mean=True, pay_amt_name='tot_pay_amt', prem_name='annual_prem'):
    # NOTE: this function does not work if records have been combined!!!
    if by:
        if mean:
            return df.groupby(by).loss_ratio.mean()
        else:
            # print("hello")
            return df.groupby(by).apply(lambda df: df[pay_amt_name].sum()/((df[prem_name]*df.exposure).sum()))
    else:
        if mean:
            return df.loss_ratio.mean()
        else:
            # print("hello")
            return df[pay_amt_name].sum()/((df[prem_name]*df.exposure).sum())

# %%
# def get_first_k_years(df, k=5, strategy='strict'):
def get_first_k_years(df, k=5, strategy='upto'):
    # get first k years of each record
    # strategy can be ['exact', 'strict', 'upto']
    # - upto: get up to the first k years of a contract
    #   - e.g., a contract lasting for 3 yrs will also be selected. But if a contract lasts for 10 years,
    #   - we get its first k years only
    # - exact: get the first k years of a contract only if it lasts for k years
    #   - NOTE: if a contract last for k years, but in kth year the exposure isn't 1, we still get it
    # - strict: similar to `exact`, but we get the first k years of a contract only if its exposure in the first k years is k.
    # e.g., if a contract has some years in the first k years removed (e.g., due to filtering), then
    # if strict == True, then only those records whose first k years have k exposure will be selected

    # IMPORTANT: some records can have total exposure > 5, but in year_seq==5, the exposure isn't 1!!
    df_first_k = df.query(f'year_seq<={k}')

    if strategy == 'upto':
        return df_first_k
    elif strategy == 'strict':
        # print("hello")
        useful_indices = df_first_k.groupby('contract_no').exposure.sum()[lambda exposure: exposure==k].index
        # print(useful_indices)
    elif strategy == 'exact':
        useful_indices = df_first_k.query(f'year_seq=={k}').contract_no.values
    else:
        print("given an unknown strategy...")
        return

    return df_first_k[df_first_k.contract_no.isin(useful_indices)]
    # return df_first_k.query("contract_no in @useful_indices")



# def get_first_k_years(df, k=5, strict=False):
#     # get first k years of each record
#     # e.g., if a contract lasts for less than k years, it won't be selected
#     # e.g., if a contract last for k years, but in kth year the exposure isn't 1:
#     #   - if strict == True, it won't be selected
#     #   - if strict == False, it will be selected
#     # e.g., if a contract has some years in the first k years removed (e.g., due to filtering), then
#     # if strict == True, then only those records whose first k years have k exposure will be selected

#     # IMPORTANT: some records can have total exposure > 5, but in year_seq==5, the exposure isn't 1!!
#     df_first_k = df.query(f'year_seq<={k}')
#     if strict:
#         useful_indices = df_first_k.groupby('contract_no').exposure.sum()[lambda exposure: exposure==k].index
#     else:
#         useful_indices = df_first_k.query(f'year_seq=={k}').contract_no.values
#     return df_first_k[df_first_k.contract_no.isin(useful_indices)]
#     # return df_first_k.query("contract_no in @useful_indices")

# %%
# QUESTION: why some records have year_seq > 5 but in fifth year, the exposure isn't 1?
"""
def try_first_k(df, k=5):
    first_k_indices = df.groupby('contract_no').exposure.sum()[lambda exposure: exposure>=5].index
    return df.query("(contract_no in @first_k_indices) and year_seq<=@k")

get_first_k_years(df_test).sort_values(['contract_no','year_seq']).contract_no
problem_indices = df_test.pipe(try_first_k).pipe(combine_records)[lambda df: df.exposure!=5].contract_no.values
problem_indices
df_test.query("contract_no in ('83888071', '80000137', '80000307')")
"""
# %%
def combine_records(df):
    # print("hello")
    # this is to avoid overwriting the given df
    df_original = df.sort_values(['contract_no','year_seq'])

    # specify which columns need to be aggregated, and the corresponding aggregate function
    col_to_agg = {
        'exposure':'sum',
        'tot_pay_amt':'sum',
        'tot_clm_amt':'sum',
        'tot_cnt':'sum',
        'eff_annual_prem':'sum',
        'has_clm':'any',
        }

    df_by_contracts_agg = (df_original
                            .groupby('contract_no')
                            .agg(col_to_agg)
                            .assign(
                                loss_ratio = lambda df: df['tot_pay_amt']/df['eff_annual_prem'],
                                frequency = lambda df: df['tot_cnt']/df['exposure'],
                                severity = lambda df: df['loss_ratio']/np.fmax(df['frequency'], 1e-10), # IMPORTANT: we need to give a nonzero value if df['frequency'] == 0, otherwise we have division of zero error.
                                # severity = lambda df: df['loss_ratio']/df['frequency'], # this will not work!! Use the above line instead!
                                # has_clm = lambda df: df['has_clm']>0,
                            )
                        )

    # drop these columns from the original df.
    # - for columns in `col_to_agg.keys()`, they should be replaced by their aggregation
    # - for ['loss_ratio', 'frequency', 'severity'], they should be recalculated
    # - for `first_year`, no longer needed, since each combined record must have first_year == True
    col_to_drop = list(col_to_agg.keys()) + ['first_year', 'loss_ratio', 'frequency', 'severity']


    return (df_original
            .drop_duplicates('contract_no', keep='last') # keep only the first year record of each contract
            .drop(col_to_drop, axis=1, errors='ignore')
            .merge(df_by_contracts_agg, on='contract_no') # get the aggregated columns
            )

    # ['exposure','tot_pay_amt','loss_ratio','frequency', 'severity', 'has_clm', 'eff_annual_prem','first_year','tot_cnt', 'tot_clm_amt']


# %%

# OLD VERSION
# def combine_records(df):
#     # return a new df, such that each contract has only one row

#     # this is to avoid overwritting the given df
#     df_temp = df.copy()

#     # after combining contract years, we need to do some aggregation on some features
#     df_by_contracts_info = (df_temp.groupby('contract_no')[['year_seq','exposure','tot_pay_amt', 'tot_clm_amt', 'tot_cnt', 'eff_annual_prem','has_clm']].agg({
#         'exposure':'sum',
#         'tot_pay_amt':'sum',
#         'tot_clm_amt':'sum',
#         'tot_cnt':'sum',
#         'eff_annual_prem':'sum',
#         'has_clm':'max',
#         # 'year_seq':'max', # not needed, since we select the last record when combining a record, and year_seq is automatically maximized
#     })
#     # after obtaining the above aggregated columns, we create the following features based on the above aggregated columns
#     .assign(
#         loss_ratio = lambda df: df['tot_pay_amt']/df['eff_annual_prem'],
#         frequency = lambda df: df['tot_cnt']/df['exposure'],
#         severity = lambda df: df['loss_ratio']/np.fmax(df['frequency'], 1e-10), # IMPORTANT: we need to give a nonzero value if df['frequency'] == 0, otherwise we have division of zero error.
#         # severity = lambda df: df['loss_ratio']/df['frequency'], # this will not work!! Use the above line instead!
#         has_clm = lambda df: df['has_clm']>0,
#         )
#     )

#     # print("df_by_contracts_info: ")
#     # print(df_by_contracts_info.iloc[:,:5])

#     # for each record, keep only the last contract year, and use contract_no as index, since contract_no is now unique
#     df_unique = df_temp.drop_duplicates('contract_no', keep='last').set_index('contract_no')

#     # print("df_unique: ")
#     # print(df_unique.iloc[:,:5])

#     # after combining, we need to
#     #   (1) aggregate some columns (e.g., exposure, loss_ratio)
#     #   (2) remove some features (e.g., first_year)
#     # so, we first drop those columns that need to be aggregated from `df_unique` and get them back from `df_by_contracts_info`. And for (2), we simply drop them
#     df_unique_drop = df_unique.drop(['exposure','tot_pay_amt','loss_ratio','frequency', 'severity', 'has_clm', 'eff_annual_prem','first_year','tot_cnt', 'tot_clm_amt'], axis='columns')

#     # concat the two dataframes based on contract_no. Note that each of the two dataframes should have the same number of rows, and so the final output `df_final` should have the same # rows as the two dataframes.
#     df_final = pd.concat([df_unique_drop, df_by_contracts_info], axis='columns')

#     # print("after concat: ")
#     return df_final.reset_index()
# %%
def export_to_db(df, schema='temp', table=None, if_exists='fail'):
    # export `df` to the `schema.table`
    with dsar.psql_con("WRITE") as con:
        df.to_sql(name=table, con=con, schema=schema, index=False, if_exists=if_exists)

def output_table(col_gp, df, y_true, y_pred, model_names):
    # TODO: rewrite using `pd.concat([df_output.reset_index(drop=True),pd.DataFrame(y_pred_test)], axis=1)`
    # df[col_gp]: the column to be exported to db
    # y_true: ground truth
    # y_pred: prediction. it is a dictionary!
    df_output = df[col_gp].copy()
    df_output['y_true'] = y_true

    for model_name in model_names:
        df_output[model_name] = y_pred[model_name]

    return df_output
# %%
def cal_metrics(y_true, y_pred, metrics, y_score, output_path=None, precision=6):
    # print("hello")
    y_pred = pd.DataFrame(y_pred)
    y_score = pd.DataFrame(y_score)
    res = dict()
    for name, metric in metrics.items():
        res[name] = y_score.apply(lambda x: metric(y_true, x)) if name in ['roc_auc', 'pr_auc'] else y_pred.apply(lambda x: metric(y_true, x))
    # print(res)
    # return pd.DataFrame(res).T.reset_index().rename(columns={'index':'metrics'})
    if output_path:
        # res.to_csv(f"{output_path}/eval_metrics.csv")
        res.to_csv(Path(output_path) / "eval_metrics.csv")
    return pd.DataFrame(res).T.round(precision)

def combine_train_test_metrics(metrics_train, metrics_test, same_name=False, long_form=False):
    if same_name:
        if long_form:
            metrics_train = metrics_train.assign(train=1)
            metrics_test = metrics_test.assign(train=0)
            metrics_test_train = pd.concat([metrics_test, metrics_train], axis=0)
        else:
            metrics_test_train = pd.concat([metrics_test, metrics_train], keys=['test','train'], axis='columns')
    else:
        pass # TODO
    return metrics_test_train
# %%
def rank_corr(models, X_test, y_true):
    for name, model in models.items():
        print(f"for model {name}, the spearman rank correlation matrix is:")
        y_pred = model.predict(X_test)
        df_temp = pd.DataFrame(dict(y_true=y_true, y_pred=y_pred))
        # debugging...
        # print(y_true[:5])
        # print(y_pred[:5])
        # print(df_temp)
        print(df_temp.corr(method='spearman'))
        # print(df_temp.rank().corr(method='spearman'))

# rank_corr(models, X_test, y_test)
# rank_corr(models, X_train, y_train)


# %%
# ------------------------ for manipulating file name ------------------------ #
def remove_quotes(seq, to_camel_case=True):
    # seq = list(str)
    # remove the quotes of the strings in the given sequence
    # return f"[{','.join(seq)}]"
    if to_camel_case:
        seq = [camel_case(s) for s in seq]
    return f"{'&'.join(seq)}"
def camel_case(s):
    # https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-96.php
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return ''.join([s[0].lower(), s[1:]])
    # return ''.join([s[0], s[1:]])
def get_acronym(s, to_upper=True):
    # e.g., comb_first_k_years => CFKY
    if '_' in s:
        return ''.join( [ component[0].upper()
                        if to_upper else component[0]
                        for component in s.split('_') if component]
                        )
    return s
# ---------------------------------------------------------------------------- #
# %%
def y_summary(s):
    d = {}
    d['zero_pct'], d['nonzero_pct'] = s.eq(0).value_counts(normalize=True).values
    d['zero'], d['nonzero'] = s.eq(0).value_counts().values
    d['shape'] = s.shape
    d['mean'] = s.mean()
    d['max'] = s.max()
    return pd.Series(d)

# def X_summary(X):
#     d={}
#     d['shape'] = X.shape
#     return pd.Series(d)

def df_summary_lr(df):
    d                       = {}
    d['shape']              = df.shape
    d['agg_lr']             = df.tot_pay_amt.sum()/df.eff_annual_prem.sum()
    d['mean_lr']            = df.loss_ratio.mean()
    d['median_lr']          = df.loss_ratio.median()
    d['mean_pay']           = df.tot_pay_amt.mean()
    d['median_pay']         = df.tot_pay_amt.median()
    d['mean_exposure']      = df.exposure.mean()
    d['median_exposure']    = df.exposure.median()
    d['mean_annual_prem']   = df.annual_prem.mean()
    d['median_annual_prem'] = df.annual_prem.median()
    return pd.Series(d)
# %%
def train_model(
    X_train,
    y_train,
    model_name,
    param_distributions,
    selected_features,
    output_path,
    object_path,
    sample_weight=None,
    filter_zero=False,
    # monotone_constraints=None,
    scoring = None,
    cv=3,
    n_iter=1,
    random_state=1234,
    # objective='reg:tweedie',
    # model_type='reg' # can be ['reg','cla']
    estimator=xgboost.XGBRegressor,
    **kwargs,
    ):
    """
    This function is a wrapper for `XGBRegressor`, `XGBClassifier` and `GammaRegressor`. 
    Other sklearn-like models are not tested yet. 
    """
    # if os.path.isfile(f"./model_files/{model_filename}"):
    #     return joblib.load(f"./model_files/{model_filename}")

    # mask = pd.Series(True, index=y_train.index)
    # if filter_zero:
    #     mask = y_train>0
    print(kwargs)
    mask = y_train > 0 if filter_zero else pd.Series(True, index=y_train.index)
    # if filter_zero:
    #     mask = y_train > 0

    # if 'monotone_constraints' in locals():
    #     print("hello")
    #     print(f"{monotone_constraints = }")

    # -------------------------- define model parameter -------------------------- #
    model = BayesSearchCV(
            estimator(
                #  objective=objective,
                #  monotone_constraints=monotone_constraints,
                random_state=random_state,
                **kwargs,
                ),
            # xgboost.XGBRegressor(verbosity=0,
            #                      objective=objective,
            #                      monotone_constraints=monotone_constraints,
            #                     ),
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            scoring=scoring,
            # n_jobs=-1,
            # verbose=10,
            )

    model.fit(
        X_train[mask],
        y_train[mask],
        sample_weight=sample_weight[mask] if sample_weight is not None else None
        )
    # if filter_zero:
    #     model.fit(X_train[mask],
    #               y_train[mask],
    #               sample_weight=sample_weight[mask] if sample_weight is not None else None
    #               )
    # else:
    #     model.fit(X_train,
    #               y_train,
    #               sample_weight=sample_weight
    #               )

    # ----------------------- check the feature importance ----------------------- #
    feat_importance = get_feature_importance(model, selected_features)

    # ---------------------- save feature importance locally --------------------- #
    if isinstance(feat_importance, pd.Series):
        feat_importance.to_csv(output_path/f"feature_importance_{model_name}_model.csv")
        print(feat_importance[:20])

    # --------------------- save the hyperparameters setting --------------------- #
    with open(output_path/f"param_{model_name}.txt", 'w') as f:
        print(model.best_estimator_.get_params(), file=f)

    # ------------------------- save model object locally ------------------------ #
    joblib.dump(model.best_estimator_, object_path/f"object_{model_name}_model.sav")


    # ------------------------ upload the model file to S3 ----------------------- #
    # with open(f"s3://dsa-data-ape1-s3/hb_poc_2021/model_files/{model_filename}", 'wb') as fout:
    # joblib.dump(model, fout)

    return model, feat_importance
# %%
def create_string(t):
    if not t:  # check whether t is empty; same as "if bool(t)==False"
        return ''
    elif len(t) == 1:  # if the list contains one string only, return that string
        return t[0]
    else:
        return ', '.join(t[:-1]) + f' and {t[-1]}'
# %%
# ---------------------------------------------------------------------------- #
#                   code for finding acceptable range of bmi                   #
# ---------------------------------------------------------------------------- #
def interpolate_child(samples_age: list, samples_bmi: dict[list]):
    res = []
    for name, y in samples_bmi.items():
        f = interpolate.interp1d(samples_age, y, kind='cubic')
        res.append(pd.Series(f(np.arange(2,18)), index=np.arange(2,18), name=name))

    return pd.concat(res, axis=1).rename_axis("age",axis=0)

def find_thresh_bmi_infant(df_unique, row_index, col_index):
    # ------------------------------- 0<=app_age<=1 ------------------------------ #
    # for those aged between 0 and 1, can't find recommended bmi on the web. So, we find the 5th and 85th percentile as the acceptable range of bmi for those aged between 0 and 1.
    thresh_infant = (
        df_unique
        .query("0<=app_age<=1")
        .groupby(["age_gp_bmi",'sex_male'], observed=True)
        .bmi
        .quantile([0.05,0.85])
        .unstack([1,2])
        .set_axis(row_index[:2], axis=0) # get index names for ages <= 1
        .set_axis(col_index, axis=1)
        # .set_axis(["female_5", "female_85", "male_5", "male_85",], axis=1)
        # .set_axis(pd.Index([0,1], name='age'))
    )
    return thresh_infant    

def find_thresh_bmi_child(row_index, col_index):
    # ------------------------------ 2<=app_age<=17 ------------------------------ #
    # for this age range, we use the recommended bmi range from CDC (Centers for Disease Control and Prevention) https://www.cdc.gov/healthyweight/assessing/bmi/childrens_bmi/about_childrens_bmi.html.
    # below, `samples_bmi` is from the same website above
    # only a few points are sampled from the curve on the website, and then we do interpolation based on the sample points.
    # why sample points? Because it seems no website that shows the bmi values for each age of the 85 and 5 percentile curves
    samples_age=[2,8,13,17]
    samples_bmi = {
        "female_5": [14.4, 13.55, 15.3, 17.2],
        "female_85": [18, 18.3, 22.5, 25.2],
        "male_5": [14.7, 13.8, 15.5, 17.6],
        "male_85": [18.2, 18, 21.9, 25],
    }
    thresh_child = (
        interpolate_child(samples_age, samples_bmi)
        .set_axis(row_index[2:18], axis=0) # get index names for ages between 2 and 17
        .set_axis(col_index, axis=1)
    )
    return thresh_child

def find_thresh_bmi_adult(row_index, col_index):
    # -------------------------------- app_age>=18 ------------------------------- #
    # the data is based on the `optimal BMI` from a study
    # for more info: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4681110/
    index=["18-34","35-44","45-49","50-54","55-64","65-74","75-inf"]
    data = {
        "female_5": [15.5, 19.0, 20.0, 22.0, 23.0, 24.0, 24.0,],
        "female_85": [24.9, 23.9, 25.9, 26.9, 27.9, 28.9, 29.9,],
        "male_5": [23.0, 23.0, 24.0, 24.0, 24.0, 25.0, 25.0,],
        "male_85": [25.9, 26.9, 27.9, 27.9, 28.9, 28.9, 32.9,],
    }
    # thresh_adult = pd.DataFrame(data=data, index=index, columns=data.keys())
    thresh_adult = (
        pd.DataFrame(data=data, index=pd.Index(index,name='age'))
        .set_axis(row_index[18:], axis=0) # get index names for ages >= 18
        .set_axis(col_index, axis=1)
    )
    
    return thresh_adult

def find_thresh_bmi(df_unique, debug=True):
    # ----------------- define row and column index for later use ---------------- #
    row_index = df_unique.age_gp_bmi.cat.categories.rename("age_gp_bmi")
    col_index = pd.MultiIndex.from_arrays(
        [[0,0,1,1],['low','high','low','high']],
        names=['sex_male','thres']
        )

    # ------------------------------- 0<=app_age<=1 ------------------------------ #
    thresh_infant = find_thresh_bmi_infant(df_unique, row_index, col_index)

    # ------------------------------ 2<=app_age<=17 ------------------------------ #
    thresh_child = find_thresh_bmi_child(row_index, col_index)

    # -------------------------------- app_age>=18 ------------------------------- #
    thresh_adult = find_thresh_bmi_adult(row_index, col_index)

    # -------------------- combine the three threshold results ------------------- #
    thresh_all_age = pd.concat([thresh_infant, thresh_child,thresh_adult], axis=0)
    # thresh_all_age = (
    #     pd.concat([thresh_infant, thresh_child,thresh_adult])
    #     .set_axis(df_unique.age_gp_bmi.cat.categories.rename("age_gp_bmi"), axis=0)
    #     .set_axis(pd.MultiIndex.from_arrays([[0,0,1,1],['low','high','low','high']], names=['sex_male','thres']), axis=1)
    #     # .sort_index(axis=1)
    # )
    
    if debug:
        for df in [thresh_infant, thresh_child, thresh_adult]:
            print(df, end='\n'*2)
        

    return thresh_all_age
# %%
# ---------------------------------------------------------------------------- #
#       code for finding the difference from the acceptable range of BMI       #
# ---------------------------------------------------------------------------- #
def find_diff_bmi(s: pd.Series, thresh_all_age: pd.DataFrame, n_return: int = 2):
    # this function should be called by `GroupBySeries.apply` method, so that `s.name` is the group name
    # thresh_all_age: contains the range of optimal bmi for each cohort
    # n_return: output size. Can be one of [1,2,3].
    #  - If n_return == 1, return the absolute difference between the upper optimal or lower optimal bmi, if the person's bmi is out of the optimal range; else, return 0.
    #  - If n_return == 2, return the two values indicating the difference between the upper optimal AND lower optimal.
    #  - If n_return == 3, similar to `n_return==2`, but also return a clip version of the person's bmi. That is, clip by max = upper optimal, and min = lower optimal
    low, high = thresh_all_age.loc[s.name]
    
    # TODO: can we eliminate the `if, elif, ...` clause?
    if n_return == 1:
        res = s.apply(
            lambda v: 0 if low <= v <= high 
            else v - high if v > high 
            else low - v
            )
        # return pd.DataFrame(res.to_list(), columns=['diff'], index=res.index)
    elif n_return == 2:
        res = s.apply(
            lambda v: [0, 0] if low <= v <= high 
            else [0, v - high] if v > high 
            else [low - v, 0]
            )
        # return pd.DataFrame(res.to_list(), columns=['diff_low', 'diff_high'], index=res.index)
    elif n_return == 3:
        res = s.apply(
            lambda v: [0, 0, v] if low <= v <= high 
            else [0, v - high, high] if v > high 
            else [low - v, 0, low]
            )
        # return pd.DataFrame(res.to_list(), columns=['diff_low', 'diff_high', 'bmi_clip'], index=res.index)
    else:
        raise ValueError(f"`n_return` should be in [1,2,3]")
    
    return pd.DataFrame(
        # `res` is a Series of lists. We use `res.to_list` so that it becomes a list of lists, and then use DataFrame constructor to make `res` becomes a frame.
        res.to_list(),
        columns=['diff'] if n_return == 1 else ['diff_low', 'diff_high'] if n_return == 2 else ['diff_low', 'diff_high', 'bmi_clip'],
        index=res.index,
    )
