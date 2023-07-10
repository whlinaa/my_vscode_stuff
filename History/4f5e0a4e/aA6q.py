# %%
print('calling ds_helper...')
# %%
from ds_utils.ds_preamble import *
from sqlalchemy import create_engine
from IPython.core.magic import register_cell_magic
from pathlib import Path
# %%
from IPython.display import display, HTML, display_html
# %%
def display_side_by_side(*args):
    # https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
# %%
def read_from_db(sql, db_name='sample_db'):
    engine = create_engine(f'postgresql://postgres@localhost:5432/{db_name}')
    return pd.read_sql(sql, con=engine)
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

pd.Series.value_counts_plus = value_counts_plus # monkey patching, so that can call the function as method
# %%
def value_counts_new(s, sort_index=False, ascending=True, pct=False, cum=False, index_name=True):
    res = {}
    res['count'] = s.value_counts(ascending=ascending)
    if sort_index:
        res['count'] = res['count'].sort_index(ascending=ascending)
        
    if pct:
        res['pct'] = res['count'].div(len(s)).mul(100)
    else:
        res['prop'] = res['count'].div(len(s))
        
    if cum:
        res['count_cum'] = res['count'].cumsum()
        if pct:
            res['pct_cum'] = res['pct'].cumsum().mul(100)
        else:
            res['prop_cum'] = res['prop'].cumsum()
            
        cols_reordered = ['count','count_cum'] + (['pct','pct_cum'] if pct else ['prop','prop_cum'])
        res = {k: res[k] for k in (cols_reordered)}
    
    return pd.concat(res.values(), axis=1, keys=res.keys()).rename_axis(index=s.name if index_name else 'values')
    
# value_counts_new(tips.day, ascending=False, sort_index=False, cum=True)

pd.Series.value_counts_new = value_counts_new # monkey patching, so that can call the function as method

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

# def output_table(col_gp, df, y_true, y_pred):
#     # TODO: rewrite using `pd.concat([df_output.reset_index(drop=True),pd.DataFrame(y_pred_test)], axis=1)`
#     # df[col_gp]: the column to be exported to db
#     # y_true: ground truth
#     # y_pred: prediction. it is a dictionary! 
#     df_output = df[col_gp].copy()
#     df_output['y_true'] = y_true

#     for model_name in model_names:
#         df_output[model_name] = y_pred[model_name]
    
#     return df_output
# %%
def cal_metrics(y_true, y_pred: dict, metrics, precision=6):
    pd.DataFrame(y_pred)
    res = dict()
    for name, metric in metrics.items():
        res[name] = y_pred.apply(lambda x: metric(y_true, x))
    # print(res)
    # return pd.DataFrame(res).T.reset_index().rename(columns={'index':'metrics'})
    return pd.DataFrame(res).T.round(precision)

# output:
#       total_bill_pred_1  total_bill_pred_2  total_bill_pred_3
# mape           0.188606           0.287540           0.385573
# mse           11.727551          33.258763          55.102530
# r2             0.851312           0.578328           0.301382
# mae            2.824101           4.620504           5.962351
# corr           0.890957           0.796394           0.747462
# %%
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

    # metrics_test_train.columns = metrics_test_train.columns.swaplevel(0,1)

# non-long_form
#                   test             train
#      total_bill_pred_1 total_bill_pred_1
# mape          0.153290          0.188606
# mse          10.969282         11.727551
# r2            0.858942          0.851312
# mae           2.426319          2.824101
# corr          0.911206          0.890957

# long form
#       total_bill_pred_1  train
# mape           0.153290      0
# mse           10.969282      0
# r2             0.858942      0
# mae            2.426319      0
# corr           0.911206      0
# mape           0.188606      1
# mse           11.727551      1
# r2             0.851312      1
# mae            2.824101      1
# corr           0.890957      1

# true long form
#       train         prediction      score
# mape      0  total_bill_pred_1   0.153290
# mse       0  total_bill_pred_1  10.969282
# r2        0  total_bill_pred_1   0.858942
# mae       0  total_bill_pred_1   2.426319
# corr      0  total_bill_pred_1   0.911206
# mape      1  total_bill_pred_1   0.188606
# mse       1  total_bill_pred_1  11.727551
# r2        1  total_bill_pred_1   0.851312
# mae       1  total_bill_pred_1   2.824101
# corr      1  total_bill_pred_1   0.890957# %%
# %%
def crange(start, end, lower=True):
    # return [chr(i) for i in range(ord('a')+start, ord('a')+end)]
    a_ascii = ord('a' if lower else 'A') 
    return map(chr, range(a_ascii+start, a_ascii+end))
# %%
# https://stackoverflow.com/questions/33358611/ipython-notebook-writefile-and-execute-cell-at-the-same-time
@register_cell_magic
def write_and_run(line, cell):
    argz = line.split()
    file = argz[-1]
    mode = 'w'
    if len(argz) == 2 and argz[0] == '-a':
        mode = 'a'
    with open(file, mode) as f:
        f.write(cell)
    get_ipython().run_cell(cell)
# %%
def safe_mkdir(path, parent_only=False):
    path = Path(path).expanduser()
    eval(f"path{'.parent' if parent_only else ''}.mkdir(exist_ok=True, parents=True)")
    # if parent_only:
    #     path.parent.mkdir(exist_ok=True, parents=True)
    # else:
    #     path.mkdir(exist_ok=True, parents=True)
# %%
