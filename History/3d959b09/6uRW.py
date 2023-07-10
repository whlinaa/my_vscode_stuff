# %%
print('calling ds_helper_specific...')
# %%
from ds_utils.ds_preamble import *
from pathlib import Path
from sqlalchemy import create_engine
from IPython.core.magic import register_cell_magic
import inspect
# from IPython.display import display, HTML, display_html
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
# https://stackoverflow.com/questions/33358611/ipython-notebook-writefile-and-execute-cell-at-the-same-time
# @register_cell_magic
# def write_and_run(line, cell):
#     argz = line.split()
#     file = argz[-1]
#     mode = 'w'
#     if len(argz) == 2 and argz[0] == '-a':
#         mode = 'a'
#     with open(file, mode) as f:
#         f.write(cell)
#     get_ipython().run_cell(cell)