# %%
# print("calling ds_plotting..")
# %%
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ds_utils.ds_helper import savefig_plus
import inspect
from sklearn.metrics import auc
import matplotlib
from matplotlib import markers
from itertools import cycle 
# %%
def plot_bar(metrics_score, ax=None, fig_name=None):
    # plot bar char for performance metrics 
    if not ax: fig,ax, = plt.subplots(1,1, figsize=(16,9))
    metrics_score.plot.bar(ax=ax)
    # ax.set_xticks(rotation=0) # TODO: why ax.set_xticks() doesn't have `rotation` parameter??
    plt.xticks(rotation=0) 
    ax.set_yticks(np.arange(0.1,1.1,0.1))
    ax.set_title(f"Performance of different models")
    ax.set_ylabel("metric value")

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()

# %%
def plot_bar_gp(data=None, x=None, y=None, row=None, col=None, orient='v', aspect=1, output_path=None):
    if output_path:
        fig_name = '+'.join(val for key,val in locals().items() if isinstance(val, str) and key !='output_path')
        fun_name = inspect.stack()[0][3]
    # print(fig_name)
    g = sns.catplot(
        data=data, hue='output', y=y, 
        kind='bar', x=x, col=col, row=row, ci=None,
        orient=orient, aspect=aspect,
    )

    # add "(mean)" word to either ylabel if xlabel, depending on `orient`
    if orient == 'v':
        g.set(ylabel=f"{y} (mean)")
        g.figure.suptitle(f'{y} (mean)')
    else:
        g.set(xlabel=f"{x} (mean)")
        g.figure.suptitle(f'{x} (mean)')
    g.tight_layout()

    # g.savefig(f'{output_path}/bar_{fig_name}.png', dpi=200)
    savefig_plus(f'{output_path}/{fun_name}_{fig_name}.png', dpi=200)
    # print(f'{output_path}/bar_{fig_name}.png')
    

    return g
# %%
def plot_box_gp(data, x=None, y=None, row=None, col=None, aspect=1, hue=None, yscale='log', output_path=None):
    if output_path:
        fig_name = '+'.join(val for key,val in locals().items() if isinstance(val, str) and key !='output_path')
        fun_name = inspect.stack()[0][3]
    g = sns.catplot(
        data=data, hue=hue, y=y, 
        kind='box', x=x, col=col, row=row, ci=None, aspect=aspect, whis=3,
        )
    g.set(yscale=yscale)
    g.fig.suptitle(f'{y} (exclude zero claims)')
    g.tight_layout()
    savefig_plus(f'{output_path}/{fun_name}_{fig_name}.png', dpi=200)
    return g
# %%
def plot_hist_gp(data, x, row=None, col=None, col_wrap=None, aspect=1, log_scale=(True,True), kde=False, suptitle=None, output_path=None):
    if output_path:
        fig_name = '+'.join(str(val) for key,val in locals().items() if isinstance(val, (str, tuple)) and key !='output_path')
        fun_name = inspect.stack()[0][3]
        # print(fig_name)
    g = sns.displot(
        data, x=x, hue='output',alpha=0.3,bins=30, palette='muted', element='step',
        common_norm=False, log_scale=log_scale, col=col, row=row, col_wrap=col_wrap, 
        stat='density', aspect=aspect, kde=kde,
    )
    g.fig.suptitle(suptitle if suptitle else f'{x} distribution', y=1.0)
    g.tight_layout()
    savefig_plus(f'{output_path}/{fun_name}_{fig_name}.png', dpi=200)
    return g
# %%
def plot_kde_gp(data, x, row=None, col=None, col_wrap=None, log_scale=(True, False), output_path=None):
    if output_path:
        fig_name = '+'.join(str(val) for key,val in locals().items() if isinstance(val, (str, tuple)) and key !='output_path')
        fun_name = inspect.stack()[0][3]
    g = sns.displot(data=data, x=x, hue='output', palette='muted', common_norm=False, kind='kde', log_scale=log_scale, bw_adjust=0.5, col=col, row=row, col_wrap=col_wrap)
    g.figure.suptitle(f'{x}: kde', y=1.0)
    # g.figure.subplots_adjust(top=.92)1
    g.tight_layout()
    savefig_plus(f'{output_path}/{fun_name}_{fig_name}.png', dpi=200)
    return g
# %%
def plot_scatter_gp(data, x, y, hue=None, style=None, col=None, row=None, xscale='log', yscale='log', output_path=None):
    if output_path:
        fig_name = '+'.join(val for key,val in locals().items() if isinstance(val, str) and key !='output_path')
        fun_name = inspect.stack()[0][3]
    g = sns.relplot(
        # data=df_gp_no_zero.sample(10_00), x='y_true', y='y_pred', kind='scatter',             
        data=data, x=x, y=y, kind='scatter',
        hue=hue, style=style, col=col, row=row, alpha=0.5,
        s=8
        ).set(xscale=xscale, yscale=yscale)
    # g.map(plt.axline, xy1=[100,100], xy2=[101,101], label='perfect prediction', linestyle='--', color='black', alpha=0.5)
    g.map(plt.axline, xy1=[0,0], xy2=[1,1], label='perfect prediction', linestyle='--', color='black', alpha=0.5)
    # g.fig.suptitle(f'{y_name}: true vs predicted')
    g.fig.suptitle(f'true vs predicted: {y}')
    g.tight_layout()
    savefig_plus(f'{output_path}/{fun_name}_{fig_name}.png')
    return g
# %%
def plot_decile_gp(df_gp, y_name, y_pred_names, row=None, col=None, sharey=True, debug=False, plot_type='line', yscale='linear', output_path=None, col_wrap=None, show_predict=True, reflines: list[tuple] = None):

    from matplotlib import markers
    sns_colors = list(sns.color_palette())
    # sns_marker = markers.MarkerStyle.markers.keys()
    sns_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    colors = cycle(sns_colors)
    markers = cycle(sns_markers)

    if output_path:
        fig_name = '+'.join(val for key,val in locals().items() if isinstance(val, str) and key !='output_path')
        fun_name = inspect.stack()[0][3] # get the function name

    df_decile = df_gp.copy()
    gp_features = [feature for feature in [row, col] if feature]

    # --------------------- add new decile columns to the df --------------------- #
    if gp_features: # if col or row: => has at least one grouping feature
        # find the decile of each group 
        for y_pred_name in y_pred_names:
            df_decile[f'decile_{y_pred_name}'] = df_decile.groupby(gp_features)[y_pred_name].transform(
                lambda x: pd.qcut(x, q=10, labels=range(1,11))
                )
    else: # both col and row are None. That means we do not have any group-by condition
        for y_pred_name in y_pred_names:
            df_decile[f'decile_{y_pred_name}'] = pd.qcut(df_decile[y_pred_name], q=10, labels=range(1,11))
    # ---------------------------------------------------------------------------- #

    if plot_type!='line':
        # if not `line`, then plot the first given model only. 
        # need to melt the df...
        model_to_show = y_pred_names[0]
        # print([y_name, model_to_show])
        df_decile_melt=df_decile.melt(
            id_vars = [ col for col in df_decile if col not in [y_name, model_to_show]],
            value_vars=[y_name, model_to_show],
            value_name="value", 
            var_name='output',
        )

    # ------------------------------- sanity check ------------------------------- #
    if debug:
        print(f"overall:\n{df_decile.decile.value_counts(normalize=True)}")
        if gp_features: 
            for name, x in df_decile.groupby(gp_features):
                print(f"decile {name}")
                print(x['decile'].value_counts(normalize=True))
    # ---------------------------------------------------------------------------- #
    
    # return df_decile_melt

    if plot_type=='line':
        g = sns.FacetGrid(data=df_decile, height=6, aspect=0.8, row=row, col=col, col_wrap=col_wrap,)
        for y_pred_name in y_pred_names:
        # plot the lines for each prediction
            g.map_dataframe(
                sns.lineplot, x=f'decile_{y_pred_name}', y='total_bill', 
                ci=None, label=f"total_bill_{y_pred_name}", 
                marker=next(markers), markersize=15, color=next(colors), 
                alpha=0.8, fillstyle='none', markeredgewidth=3, markerfacecolor='red'
                )

            if show_predict: 
                g.map_dataframe(
                    sns.lineplot, x=f'decile_{y_pred_name}', y=y_pred_name, 
                    ci=None, label=y_pred_name, marker=next(markers), 
                    markersize=7, color=next(colors), alpha=0.8, fillstyle='none'
                    )

    elif plot_type=='box': 
        # boxplot
        g = sns.catplot(
            data=df_decile_melt, x=f'decile_{model_to_show}', y='value', 
            hue='output', kind='box',
            facet_kws={'sharey': sharey}, row=row, col=col, aspect=2, whis=3, col_wrap=col_wrap)
    
    elif plot_type=='violin':
        # violin plot
        g = sns.catplot(
            data=df_decile_melt, x=f'decile_{model_to_show}', y='value', 
            hue='output', kind='violin',
            row=row, col=col, aspect=2, col_wrap=col_wrap,
            # scale='count',
            scale='width',
            )

    if reflines:
        for axis, val, label in reflines:
            if axis == 'x':
                g.refline(x=val, label=label, color=next(colors))
            else:
                g.refline(y=val, label=label, color=next(colors))

    g.set(
        ylabel=f"{y_name}", 
        xlabel=f'decile (ordered by prediction)', 
        # xticks=range(1,11),
        yscale=yscale,
        )
    
    g.figure.suptitle(f'{y_name}: decile plot')
    g.tight_layout()
    if plot_type != 'line':
        g.legend.remove()
    g.add_legend()

    if output_path:
        savefig_plus(f'{output_path}/{fun_name}_{fig_name}.png', dpi=200)
    return g
        

# %%
def lorenz_curve(y_true, y_pred, exposure):
    # order samples by increasing predicted risk (the higher pure_premium, the riskier):
    # rank in ascending order of y_pred
    df_risk = (pd.DataFrame(dict(y_pred=y_pred, exposure=exposure, y_true=y_true))
                .sort_values(by='y_pred')
                .assign(cum_clm_amt= lambda df_risk: np.cumsum(df_risk['y_true'] * df_risk['exposure']),
                        cum_clm_amt_rate=lambda df_risk: df_risk['cum_clm_amt']/df_risk['cum_clm_amt'].iloc[-1],
                        # ordered_samples= lambda df_risk: np.linspace(0, 1, len(df_risk)),
                        ordered_samples= lambda df_risk: np.linspace(1, 0, len(df_risk), endpoint=False)[::-1],
                )
                )

    # print(type(df_risk['cum_clm_amt']))
    # return pd.DataFrame(dict(y_pred=y_pred, exposure=exposure, y_true=y_true))

    # print(df_risk) # sanity check
    # return df_risk
    return df_risk['ordered_samples'], df_risk['cum_clm_amt_rate']    

# for debugging
# lorenz_curve(y_test, models['xgb'].predict(X_test), df_test["exposure"])

def gini_for_tuning(y_true, y_pred, exposure):
    pass
    # TODO
    
def find_gini(x, y):
    return 1 - 2 * auc(x, y)

# def plot_lorenz_curve(models, X_test, y_true, y_pred, exposure, title=None, output_path=None):
def plot_lorenz_curve(y_true, y_pred, exposure, title=None, output_path=None):
    if output_path:
        fun_name = inspect.stack()[0][3]
        fig_name = f'{fun_name}_{title}'
    plt.figure(figsize=(8,8))
    # plot lorenz curve for each model
    # for label, model in models.items():
    #     if "base" not in label: # except base models 
    #         y_pred = model.predict(X_test)

    for label, pred in y_pred.items():
        ordered_samples, cum_clm_amt_rate = lorenz_curve(y_true, pred, exposure)
        gini = find_gini(ordered_samples, cum_clm_amt_rate)
        label += f" (Gini index: {gini:.3f})"
        plt.plot(ordered_samples, cum_clm_amt_rate, linestyle="-", label=label)
    
    # plot ground truth: y_pred == y_test
    ordered_samples, cum_clm_amt_rate = lorenz_curve(y_true, y_true, exposure)
    gini = find_gini(ordered_samples, cum_clm_amt_rate)
    label = f"ground truth (Gini index: {gini:.3f})"
    plt.plot(ordered_samples, cum_clm_amt_rate, linestyle="-.", color="gray", label=label)

    # plot ground truth in reverse order
    ordered_samples, cum_clm_amt_rate = lorenz_curve(y_true, -y_true, exposure)
    gini = find_gini(ordered_samples, cum_clm_amt_rate)
    label = f"ground truth in reverse order"
    plt.plot(ordered_samples, cum_clm_amt_rate, linestyle="dashdot", color="red", label=label)

    # Random baseline
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="random baseline")

    # plt.title("Lorenz curves for each model")
    plt.title(title)
    plt.xlabel("Fraction of contracts\n(ordered by model from safest to riskiest)")
    plt.ylabel("Fraction of total claim amount")
    plt.legend(loc="upper left")
    savefig_plus(f'{output_path}/{fig_name}.png', dpi=300)
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