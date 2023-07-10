# %%
print('calling ds_helper...')
# %%
from ds_utils.ds_preamble import *
from pathlib import Path
from sqlalchemy import create_engine
from IPython.core.magic import register_cell_magic
import inspect
# from IPython.display import display, HTML, display_html
# %%
def find_extreme(
    df: pd.DataFrame | pd.Series,
    col: str = None,
    low_q: bool = None,
    high_q: bool = None,
    ignore_zero: bool = False,
    sort_index: bool = False,
    ascending: bool = True,
    ) -> pd.DataFrame | pd.Series:
    # return those records that are in the tail ends specified by `low_q` and `high_q`
    # df can be a dataframe or series. If dataframe, then `col` must be specified
    # this function is to find records with the lowest values and highest values, according to quantiles.
    # can optionally ignore zero values. This is sometimes needed for zero-inflated feature, and we just want to find low quantile without regard to zero values
    # either low_q or high_q can be left unspecific, in which case, we filter only one extreme of the data

    assert low_q or high_q, "at least one of `low_q` or `high_q` must be set"
    assert isinstance(df, pd.Series) or col, "if given a dataframe, `col` must be set"

    s = df[col] if col else df
    if ignore_zero:
        s = s.replace(1, np.nan)
    if low_q:
        lower_lim = s.quantile(low_q)
    if high_q:
        upper_lim = s.quantile(high_q)

    res = (
        df[s.gt(upper_lim)] if low_q is None
        else df[s.lt(lower_lim)] if high_q is None
        else df[s.gt(upper_lim) | s.lt(lower_lim)]
    )

    if sort_index:
        return res.sort_index(ascending=ascending)
    else:
        return (
            res.sort_values(col, ascending=ascending) if col
            else res.sort_values(ascending=ascending)
        )

# find_extreme(tips.total_bill, low_q=0.05, high_q=0.95)

# find_extreme(tips, col='size', low_q=0.05, high_q=0.95)
# find_extreme(tips, col='size', high_q=0.95)
# find_extreme(tips.tip, high_q=0.95)
# find_extreme(tips.tip, low_q=0.05, high_q=0.95, sort_index=True)
# find_extreme(tips.tip, sort_index=True)
# find_extreme(tips, low_q=0.05, sort_index=True)
# %%
# ---------------------------------------------------------------------------- #
#                               string management                              #
# ---------------------------------------------------------------------------- #
def concat_string(t: list[str]) -> str:
    # check whether t is empty (=[]); same as "if bool(t)==False"
    if not t:
        return ''
    # if the list contains one string only, return that string
    elif len(t) == 1:
        return t[0]
    else:
        # return ', '.join(t[:-1]) + f' and {t[-1]}'
        return f"{', '.join(t[:-1])}" + f' and {t[-1]}'
# concat_string(t) # 'abc, def, gh and ik'

def remove_quotes(seq, to_camel_case=True):
    # seq = list(str)
    # remove the quotes of the strings in the given sequence
    return f"[{','.join(seq)}]"

    # TODO: what is the purpose of the code below?
    if to_camel_case:
        seq = [camel_case(s) for s in seq]
    return f"{'&'.join(seq)}"

# remove_quotes(['abc', 'def', 'gh']) # '[abc,def,gh]'

def camel_case(s) -> str:
    # https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-96.php
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return ''.join([s[0].lower(), s[1:]])
    # return ''.join([s[0], s[1:]])

def get_acronym(s, to_upper=True) -> str:
    # e.g., comb_first_k_years => CFKY
    if '_' in s:
        return ''.join(
            [
                component[0].upper() if to_upper else component[0]
                for component in s.split('_')
                if component # if not None
            ]
        )
    return s
# %%
def crange(
    start: int|str = 'a',
    end: int|str = 'z',
    lower: bool = True,
    as_list: bool = True,
    ) -> list | Iterator[str]:
    """
    range-like function, but return a list of English characters in order
    """
    # crange(2,5) # ['c', 'd', 'e']
    # crange('a','f') # ['a', 'b', 'c', 'd', 'e']

    # a_ascii = ord('a') if lower else ord('A')

    # --------- idea: make sure everything is in lower letter code first --------- #
    a_ascii = ord('a')
    start = ord(start.lower()) if isinstance(start, str) else a_ascii + start
    end = ord(end.lower()) if isinstance(end, str) else a_ascii + end

    if not lower:
        to_upper_size = ord('A') - ord('a')
        start += to_upper_size
        end += to_upper_size

    res = map(chr, range(start, end))
    return list(res) if as_list else res

# crange(10, 20)
# crange('c', chr(ord('c') + 3))
# crange(2,5)
# crange('a','f')
# crange('a','f', lower=False)
# crange(0, 3, lower=False)
# crange('A','F')

# def crange(start, end, lower=True):
#     # return [chr(i) for i in range(ord('a')+start, ord('a')+end)]
#     a_ascii = ord('a' if lower else 'A')
#     return map(chr, range(a_ascii+start, a_ascii+end))

# %%
def summary_na(
    df: pd.DataFrame,
    as_pct: bool = False,
    ascending: bool = True,
    sort_index: bool = False,
    show_cnt: bool = True,
    show_prop: bool = True,
    set_caption: bool = False,
    background_gradient: bool = False
    ) -> pd.DataFrame :
    # missing value summary

    assert show_cnt or show_prop, "either show_count or show_prop must be set True"

    res = df.isna().sum()[lambda x: x>0].rename_axis("feature with missing")
    if sort_index:
        res = res.sort_index(ascending=ascending)
    else:
        res = res.sort_values(ascending=ascending)

    res = res.to_frame('missing_cnt').assign(missing_prop= lambda res: res.missing_cnt.div(len(df)))

    if not show_cnt:
        res = res.drop("missing_cnt", axis=1)
    if not show_prop:
        res = res.drop("missing_prop", axis=1)

    if show_prop and as_pct:
        res.missing_prop = res.missing_prop.mul(100)
        res = res.rename({"missing_prop": "missing_pct"}, axis=1)

    if set_caption or background_gradient:
        res = res.style
        if set_caption:
            res = res.set_caption("features with missing values")
        if background_gradient:
            res = res.background_gradient(cmap='Blues')
    return res
# %%
# --------------------------- styling of dataframe --------------------------- #
def style_df(df, caption=None, cmap=None, subset=None, precision=None, head=None):
    res = df.style
    if precision:
        res = res.format(precision=precision)
    if caption:
        res = res.set_caption(caption)
    if cmap:
        res = res.background_gradient(cmap, subset=subset)
    if head:
        res = res.hide(axis=0, subset=slice(head,None)) # hide rows
    return res
# %%
def display_side_by_side(*args):
    # https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'), raw=True)
# %%
def read_from_db(sql:str, db_name:str = 'sample_db'):
    # read from local database
    engine = create_engine(f'postgresql://postgres@localhost:5432/{db_name}')
    return pd.read_sql(sql, con=engine)
# %%
def to_csv_plus(
    df: pd.DataFrame,
    output_path: str = None,
    add_timestamp: bool|str = False,
    **kwargs, # for the original `to_csv` method 
    ):
    # improvements:
    # if output_path == None: use the format df_out_{timestamp}.csv
    # can automatically add timestamp
    # create intermediate parent folders if not exists

    if output_path is None:
        # use default name
        # print(kwargs)
        df.to_csv(Path(f"df_out_{get_timestamp()}.csv"), **kwargs)
        # df.to_csv(Path(f"df_out_{get_timestamp()}.csv"), header=False, index=False)
    else:
        timestamp = (
            '' if add_timestamp is False
            else add_timestamp if isinstance(add_timestamp, str)
            else get_timestamp()
        )

        path = Path(output_path).expanduser()

        # extract the directory name of the figure file first.
        # e.g., dir_name = './hello/world', and then create the directories as needed
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            path.with_stem(
                '_'.join(p for p in [path.stem, timestamp] if p) # timestamp could be None
                ),
            **kwargs
            )

# %%
def value_counts_plus(
    s:pd.Series,
    sort_index=False,
    ascending=True,
    show_count=True,
    show_prop=True,
    show_cum=False, # if True, then cum of both count and prop are shown
    index_name:str=False, # user-specified index name
    as_pct=False,
    ):

    assert show_count or show_prop, "either show_count or show_prop must be set True"

    res = {'count': s.value_counts(ascending=ascending)}
    if sort_index:
        res['count'] = res['count'].sort_index(ascending=ascending)
    if show_prop:
        prop = res['count'].div(len(s))
        if as_pct:
            prop *= 100
            res['pct'] = prop
        else:
            res['prop'] = prop
    if show_cum:
        for key, val in list(res.items()):
            res[f"{key}_cum"] = val.cumsum()

    cols_ordered = []
    if show_count:
        cols_ordered.append('count')
        if show_cum:
            cols_ordered.append('count_cum')
    if show_prop:
        cols_ordered.append('pct' if as_pct else 'prop')
        if show_cum:
            cols_ordered.append('pct_cum' if as_pct else 'prop_cum')

    res = pd.DataFrame({k: res[k] for k in cols_ordered})
    if index_name:
        res = res.rename_axis(index=index_name)
    return res

pd.Series.value_counts_plus = value_counts_plus # monkey patching, so that can call the function as method
# %%
def get_timestamp(microsecond=False, microsecond_only=False, long_year=True, verbose=False):
    # remember that we can't show millisecond (10^-3 second), since python only support microsecond (10^-6 second)!
    if microsecond_only:
        format = "%f"
    else:
        format = ('%Y' if long_year else '%y') + ("_%m_%d_%H_%M_%S" if verbose else "%m%d_%H%M%S")
        if microsecond:
            format += '_%f'

    return datetime.datetime.now().strftime(format)
# %%
# directory management
def savefig_plus(path: str, dpi: int|str ='figure'):
    # save figure to a path specificed in `path`. Create parent directory if not exists
    # e.g., path = './hello/world/myfig.png'
    # this function deal with the shortcoming of matplotlib that its `savefig` function
    # will fail if the parent directory of `path` doesn't exist
    # it will first create the necessary folders in the parent of `path` (if not exist)

    path = Path(path).expanduser() # deal with relative path (e.g., `./`)
    path.parent.mkdir(parents=True, exist_ok=True) # extract the directory name of the figure file first. e.g., dir_name = './hello/world', and then create the directories as needed
    plt.savefig(path, bbox_inches='tight', dpi=dpi)
# %%
def y_summary(y):
    # BUG: what if y has all non-zero or zero values?
    # summarize zero and non-zero counts
    y = pd.Series(y) # intput could be ndarray
    d = {}
    d['zero_pct'], d['nonzero_pct'] = y.eq(0).value_counts(normalize=True).values
    d['zero_cnt'], d['nonzero_cnt'] = y.eq(0).value_counts().values
    return pd.concat([pd.Series(d), pd.Series(y).describe()])

# def df_summary_lr(df):
#     d                       = {}
#     d['shape']              = df.shape
#     d['agg_lr']             = df.tot_pay_amt.sum()/df.eff_annual_prem.sum()
#     d['mean_lr']            = df.loss_ratio.mean()
#     d['median_lr']          = df.loss_ratio.median()
#     d['mean_pay']           = df.tot_pay_amt.mean()
#     d['median_pay']         = df.tot_pay_amt.median()
#     d['mean_exposure']      = df.exposure.mean()
#     d['median_exposure']    = df.exposure.median()
#     d['mean_annual_prem']   = df.annual_prem.mean()
#     d['median_annual_prem'] = df.annual_prem.median()
# return pd.Series(d)
# %%
def cal_metrics(
    y_true: pd.Series,
    metrics: dict,
    y_pred: pd.DataFrame = None,
    y_score: pd.DataFrame = None,
    precision: int = 6,
    index_name: str = 'metric',
    columns_name: str = 'model',
    output_path: str = None,
):
    # better to create a df like the following, then do subset
    #   portion model  y     score  pred    thresh
    # 0   train    lr  1  0.969596     1  0.729687
    # 1   train    lr  1  0.561030     0  0.729687
    # 2   train    lr  0  0.018647     0  0.729687

    # y_pred: each column is the prediction result of a model
    #         dt  lr  svc  xgb
    # 0       0   1    0    1
    # 1       0   0    1    0
    # 2       0   0    0    0
    # 3       1   1    1    0
    # 4       0   0    1    1

    # metrics: {metric_name: metric_function}
    # {'accuracy': <function accuracy_score at 0x15a433910>, 'precision': <function precision_score at 0x15a4480d0>, 'recall': <function recall_score at 0x15a448160>}

    # problem: some metrics require y_score instead of y_pred...
    # solution: call once for y_pred, and another one for y_score

    res = dict()
    for name, metric in metrics.items():
        # for each metric...
        # if the metric has `y_score` parameter, use `y_score` parameter instead

        # for each column, apply the metric function to the column
        if 'y_score' in inspect.signature(metric).parameters:
            assert y_score is not None, f"y_score can't be None, since {name} requires a score!"
            res[name] = y_score.apply(lambda col: metric(y_true, col))
        else:
            assert y_pred is not None, f"y_pred can't be None, since {name} requires a prediction!"
            res[name] = y_pred.apply(lambda col: metric(y_true, col))
    # return pd.DataFrame(res).T.reset_index().rename(columns={'index':'metrics'})
    res = pd.DataFrame(res).T.round(precision).rename_axis(index=index_name, columns=columns_name)

    if output_path:
        path = Path(output_path).expanduser()
        res.to_csv(
            # Path(".") / f"eval_metrics_{get_timestamp()}.csv"
            Path(f"eval_metrics_{get_timestamp()}.csv")
            if output_path == 'auto'
            else path
            )

    return res

# output:
# model            dt        lr       svc       xgb
# metric
# accuracy   0.453333  0.620000  0.500000  0.460000
# precision  0.500000  0.765957  0.536082  0.509804
# recall     0.402439  0.439024  0.634146  0.317073
# roc_auc    0.476506  0.635760  0.517396  0.495158
# pr_auc     0.534773  0.657617  0.596948  0.544910


# example
# np.random.seed(1)
# n_train = 150
# n_test = 100
# y_train = np.random.randint(2, size=n_train) # label of each record
# y_test = np.random.randint(2, size=n_test) # label of each record
# model_names = ['lr', 'xgb', 'dt', 'svc']
# # thresh = 0.5 # threshold for y == 1
# thresh = np.clip(
#     np.random.normal(loc=0.5, scale=0.1, size=len(model_names)), 0, 1
#     ) # threshold for y == 1

# # train set
# train_test = []

# for portion in ['train', 'test']:
#     res = []
#     for i, model_name in enumerate(model_names):
#         d = {}
#         d['y'] = y_train if portion == 'train' else y_test
#         d['score'] = np.random.uniform(size=n_train if portion == 'train' else n_test)
#         d['pred'] = (d['score']>thresh[i]).astype(int) # change to `int` to avoid having decimal number
#         # d['thresh'] = 0.4 if portion == 'train' else 0.5
#         d['thresh'] = thresh[i]
#         # print(pd.DataFrame(d))
#         res.append(pd.DataFrame(d))
#     # print(pd.concat(res, keys=model_names, names=['model']).reset_index(0))
#     train_test.append(pd.concat(res, keys=model_names, names=['model']).reset_index(0))

# df = pd.concat(train_test, keys=['train','test'], names=['portion']).reset_index(0)
# print(df.head(3))
# print(df.query("portion == 'train'").pivot(columns='model', values='pred').head(3))

# metrics = {
#     'accuracy':accuracy_score,
#     'precision': precision_score,
#     'recall': recall_score,
#     'roc_auc':roc_auc_score,
#     'pr_auc': average_precision_score
#     }

# # print(df.query("portion == 'train'").pivot(columns='model', values='pred'))

# train = cal_metrics(
#     y_train,
#     y_pred = df.query("portion == 'train'").pivot(columns='model', values='pred'),
#     y_score = df.query("portion == 'train'").pivot(columns='model', values='score'),
#     metrics=metrics,
#     output_path='model_out_testing.csv'
#     )
# test = cal_metrics(
#     y_test,
#     y_pred = df.query("portion == 'test'").pivot(columns='model', values='pred'),
#     y_score = df.query("portion == 'test'").pivot(columns='model', values='score'),
#     metrics=metrics,
#     output_path='model_out_testing.csv'
#     )
# %%
def combine_train_test_metrics(
    metrics_train: pd.DataFrame,
    metrics_test: pd.DataFrame,
    same_name: bool = True,
    as_long_form: bool = False,
    melt_model: bool = False
):
    # default is to output as wide-form, with multi-level column index, with level 0 indicating 'test' or 'train'
    # melt_model: whether change the model columns into two columns, making it a truly long form
    if same_name:
        if as_long_form:
            metrics_test_train = (
                pd
                .concat(
                    [metrics_train, metrics_test],
                    axis=0,
                    keys=['train','test'],
                    names=['portion']
                    )
                .reset_index()
                .rename_axis('', axis=1)
            )
            if melt_model:
                metrics_test_train = metrics_test_train.melt(id_vars=['portion','metric'], var_name='model')

            # metrics_train = metrics_train.assign(train=1)
            # metrics_test = metrics_test.assign(train=0)
            # metrics_test_train = pd.concat([metrics_test, metrics_train], axis=0)
        else:
            metrics_test_train = pd.concat(
                [metrics_test, metrics_train],
                keys=['test','train'],
                axis='columns',
                names=['portion']
                )
    else:
        pass # TODO
    return metrics_test_train
# ------------------------- if as_long_form == False: ------------------------ #
# portion        test               train
# model            dt        lr        dt        lr
# metric
# accuracy   0.610000  0.490000  0.453333  0.620000
# precision  0.638889  0.461538  0.500000  0.765957
# recall     0.469388  0.244898  0.402439  0.439024
# roc_auc    0.609044  0.443778  0.476506  0.635760
# pr_auc     0.589570  0.445782  0.534773  0.657617

# ------------------------- if as_long_form == True: ------------------------- #
#   portion     metric        dt        lr
# 0   train   accuracy  0.453333  0.620000
# 1   train  precision  0.500000  0.765957
# 2   train     recall  0.402439  0.439024
# 3   train    roc_auc  0.476506  0.635760
# 4   train     pr_auc  0.534773  0.657617
# 5    test   accuracy  0.610000  0.490000
# 6    test  precision  0.638889  0.461538
# 7    test     recall  0.469388  0.244898
# 8    test    roc_auc  0.609044  0.443778
# 9    test     pr_auc  0.589570  0.445782

# ------------------------- if melt_model==True : ------------------------- #
#    portion     metric model     value
# 0    train   accuracy    dt  0.453333
# 1    train  precision    dt  0.500000
# 2    train     recall    dt  0.402439
# 3    train    roc_auc    dt  0.476506
# 4    train     pr_auc    dt  0.534773
# 5     test   accuracy    dt  0.610000
# 6     test  precision    dt  0.638889
# 7     test     recall    dt  0.469388
# 8     test    roc_auc    dt  0.609044
# 9     test     pr_auc    dt  0.589570
# 10   train   accuracy    lr  0.620000
# 11   train  precision    lr  0.765957
# 12   train     recall    lr  0.439024
# 13   train    roc_auc    lr  0.635760
# 14   train     pr_auc    lr  0.657617
# 15    test   accuracy    lr  0.490000
# 16    test  precision    lr  0.461538
# 17    test     recall    lr  0.244898
# 18    test    roc_auc    lr  0.443778
# 19    test     pr_auc    lr  0.445782

# ex:
# print(combine_train_test_metrics(train, test, as_long_form=True, melt_model=True).query("model.isin(['dt','lr'])"))
# %%
# ---------------------------- below not yet done ---------------------------- #
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
def safe_mkdir(path, parent_only=False):
    path = Path(path).expanduser()
    eval(f"path{'.parent' if parent_only else ''}.mkdir(exist_ok=True, parents=True)")
    # if parent_only:
    #     path.parent.mkdir(exist_ok=True, parents=True)
    # else:
    #     path.mkdir(exist_ok=True, parents=True)
