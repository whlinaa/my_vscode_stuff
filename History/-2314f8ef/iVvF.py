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
from ds_preamble import *
# %%
def load_table(schema: str, table: str, columns: str=None, chunksize: int=None, debug: bool=False, where=None):
    """query database. Return a DataFrame

    Returns:
        DataFrame: a DataFrame containing the matched records
    """	

    columns = ', '.join(columns) if columns else '*'

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

pd.Series.value_counts_plus = value_counts_plus # monkey patching, so that we can call the function as method

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
def savefig_plus(path: str, dpi: int='figure'):
    # save figure to a path specificed in `path`
    # e.g., path = './hello/world/myfig.png'
    dir_name = os.path.dirname(path) # extract the directory name of the figure file first. e.g., dir_name = './hello/world'
    # print(dir_name)
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=dpi)

# %%
# get formatted current timestamp for saving files
def get_timestamp(microsecond=False):
    # return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") if microsecond else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
def get_first_k_years(df, k=5, strict=False):
    # get first k years of each record
    # e.g., if a contract lasts for less than k years, it won't be selected
    # e.g., if a contract last for k years, but in kth year the exposure isn't 1: 
    #   - if strict == True, it won't be selected
    #   - if strict == False, it will be selected
    # e.g., if a contract has some years in the first k years removed (e.g., due to filtering), then 
    # if strict == True, then only those records whose first k years have k exposure will be selected

    # IMPORTANT: some records can have total exposure > 5, but in year_seq==5, the exposure isn't 1!!
    df_first_k = df.query(f'year_seq<={k}')
    if strict:
        useful_indices = df_first_k.groupby('contract_no').exposure.sum()[lambda exposure: exposure==k].index
    else:
        useful_indices = df_first_k.query(f'year_seq=={k}').contract_no.values
    return df_first_k[df_first_k.contract_no.isin(useful_indices)]
    # return df_first_k.query("contract_no in @useful_indices")

# %%
# QUESTION: why some records have year_seq > 5 but in fifth year, the exposure isn't 1?
"""
def try_first_k(df, k=5):
    first_k_indices = df.groupby('contract_no').exposure.sum()[lambda exposure: exposure>=5].index
    return df.query("(contract_no in @first_k_indices) and year_seq<=@k")

get_first_k_years(df_test).sort_values(['contract_no','year_seq']).contract_no
problem_indices = df_test.pipe(try_first_k).pipe(combine_records)[lambda df: df.exposure!=5].contract_no.values
problem_indices
"""
# %%
def combine_records(df):
    # return a new df, such that each contract has only one row
    df_temp = df.copy()
    # df_temp['eff_annual_prem'] = df.annual_prem * df.exposure
    df_by_contracts_info = (df_temp.groupby('contract_no')[['year_seq','exposure','tot_pay_amt', 'tot_clm_amt', 'tot_cnt', 'eff_annual_prem','has_clm']].aggregate({
        'exposure':'sum',
        'tot_pay_amt':'sum', 
        'tot_clm_amt':'sum', 
        'tot_cnt':'sum', 
        'eff_annual_prem':'sum',
        'has_clm':'max',
        # 'year_seq':'max', # not needed, since we select the last unique record, and year_seq is automatically maximized
    })
    .assign(
        loss_ratio = lambda df: df['tot_pay_amt']/df['eff_annual_prem'],
        frequency = lambda df: df['tot_cnt']/df['exposure'],
        severity = lambda df: df['loss_ratio']/np.fmax(df['frequency'], 1e-10), # IMPORTANT: we need to give a nonzero value if df['frequency'] == 0, otherwise we have division of zero error.
        # severity = lambda df: df['loss_ratio']/df['frequency'], # this will not work!! Use the above line stead!
        has_clm = lambda df: df['has_clm']>0,
        )
    )

    # df_by_contracts_info.drop(['eff_annual_prem'], axis=1, inplace=True)
    # print(df_by_contracts_info)

    df_unique = df_temp.drop_duplicates('contract_no', keep='last').set_index('contract_no')
    # print(df_unique)

    df_unique_drop = df_unique.drop(['exposure','tot_pay_amt','loss_ratio','frequency', 'severity', 'has_clm', 'eff_annual_prem','first_year','tot_cnt', 'tot_clm_amt'], axis='columns')

    df_final = pd.concat([df_unique_drop, df_by_contracts_info],axis='columns')

    return df_final.reset_index()
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
def cal_metrics(y_true, y_pred, metrics, output_path=None, precision=6):
    # print("hello")
    y_pred = pd.DataFrame(y_pred)
    res = dict()
    for name, metric in metrics.items():
        res[name] = y_pred.apply(lambda x: metric(y_true, x))
    # print(res)
    # return pd.DataFrame(res).T.reset_index().rename(columns={'index':'metrics'})
    if output_path:
        res.to_csv(f"{output_path}/eval_metrics.csv")
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
def find_corr(y_true, y_pred):
    # y_true and y_pred are 1d array
    # define a custom metric to evaluate model performance. Specifically, we use Spearman's rank correlation
    df = pd.DataFrame(dict(y_pred = y_pred, y_true = y_true))
    return df.corr(method='spearman').iloc[0,1]

