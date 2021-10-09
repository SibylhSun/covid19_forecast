import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20}) # set plot font sizes

import numpy as np
from scipy.stats import beta, truncnorm

from model import Compartments


def plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end,
                 ax, title, in_sample_preds=None,
                 oos_preds=None,
                 truth=None,
                 plot_legend=False, plot_ticks=False):
    provided_vals = []

    if truth is not None:
        ax.plot(truth[0],truth[1],
                 label='truth')
        provided_vals += [truth[1]]

    if in_sample_preds is not None:
        ax.plot(in_sample_preds[0], in_sample_preds[1],
                 label='in-sample preds')
        provided_vals += [in_sample_preds[1]]
    if oos_preds is not None: #oos_preds=out of sample predictions
        ax.plot(oos_preds[0], oos_preds[1],
                 label='oos_preds')
        provided_vals += [oos_preds[1]]

    max_y = max([max(vals) for vals in provided_vals if vals is not None])
    if truth is not None:
        max_y = max(truth[1])

    h1 = ax.fill_between(df.loc[warmup_start:warmup_end].index.values, 0, max_y, alpha=0.15, color='red',
                         label='warmup')
    h2 = ax.fill_between(df.loc[train_start:train_end].index.values, 0, max_y, alpha=0.15, color='green', label='train')
    h3 = ax.fill_between(df.loc[test_start:test_end].index.values, 0, max_y, alpha=0.15, color='yellow', label='test')

    if plot_legend:
        ax.legend()
    if plot_ticks:
        month_ticks = matplotlib.dates.MonthLocator(interval=1)
        ax.xaxis.set_major_locator(month_ticks)
        # cut off last tick
        ax.set_xticks(ax.get_xticks()[:-1])
    else:
        ax.set_xticks([])

    ax.title.set_text(title)
    ax.set_ylim(0, max_y)


def plot_beta_prior(title, learned_values, alpha_param, beta_param, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 1000)
    pdf_vals = beta.pdf(x, alpha_param, beta_param)
    ax.plot(x, pdf_vals,
            'k-', lw=5, label='beta prior');
    ax.plot([learned_values['everyone'], learned_values['everyone']], [0, max(pdf_vals)],
            label=f'{title}',
            linestyle='--', linewidth=5)
    ax.legend(prop={'size': 10})
    ax.title.set_text(title)


def plot_pi_prior(title, learned_values, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
    days = len(learned_values['everyone'])
    x = range(1, days + 1)
    ax.plot(x, learned_values['everyone'], '.',
            label=f'{title}', markersize=15)
    ax.legend(prop={'size': 10})
    ax.title.set_text(title)


def make_all_plots(df, model,
                   alpha_bar_M, beta_bar_M,
                   alpha_bar_X, beta_bar_X,
                   alpha_bar_G, beta_bar_G,
                   warmup_start, warmup_end,
                   train_start, train_end,
                   test_start, test_end,
                   train_preds, test_preds,
                   forecasted_fluxes,
                   save_path=None):
    all_days = df.loc[warmup_start:test_end].index.values
    train_days = df.loc[train_start:train_end].index.values
    test_days = df.loc[test_start:test_end].index.values
    train_test_days = df.loc[train_start:test_end].index.values
    
    fig = plt.figure(figsize=(15, 10))
    ax_hosp_tot = plt.subplot2grid((1, 1), (0, 0))
    # Make big G total plot
    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_hosp_tot, 'G, Total',
                 in_sample_preds=(train_days, train_preds),
                 oos_preds=(train_test_days, test_preds),
                 truth=(all_days,
                        df.loc[warmup_start:test_end, 'general_ward'].values),
                 plot_legend=True, plot_ticks=True)
    fig.savefig('./figs/total_v3.pdf', dpi=100)
    
    fig = plt.figure(figsize=(15, 5))
    ax_rt = plt.subplot2grid((1, 3), (0, 0))
    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_rt, 'Rt',
                 truth=(all_days, df.loc[warmup_start:test_end, 'Rt'].values))
    fig.savefig('./figs/Rt_v3.pdf', dpi=100)
    
    
    fig = plt.figure(figsize=(15, 15))
    ax_a_tot = plt.subplot2grid((3, 3), (0, 0))
    ax_m_tot = plt.subplot2grid((3, 3), (0, 1))
    ax_x_tot = plt.subplot2grid((3, 3), (0, 2))
    
    ax_rho_M = plt.subplot2grid((3, 3), (1, 0))
    ax_rho_X = plt.subplot2grid((3, 3), (1, 1))
    ax_rho_G = plt.subplot2grid((3, 3), (1, 2))
    ax_pi_M = plt.subplot2grid((3, 3), (2, 0))
    ax_pi_X = plt.subplot2grid((3, 3), (2, 1))
    ax_pi_G = plt.subplot2grid((3, 3), (2, 2))


    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_a_tot, 'A, Everyone',
                 truth=(all_days, df.loc[warmup_start:test_end, 'asymp'].values),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.asymp.value]['everyone'].stack()))
   
    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_m_tot, 'M, Everyone',
                 truth=(all_days, df.loc[warmup_start:test_end, 'mild'].values),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.mild.value]['everyone'].stack()))


    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_x_tot, 'X, Everyone',
                 truth=(all_days, df.loc[warmup_start:test_end, 'extreme'].values),
                 oos_preds=(all_days,  forecasted_fluxes[Compartments.extreme.value]['everyone'].stack()))
    

    plot_beta_prior('Rho M', model.rho_M, alpha_bar_M, beta_bar_M, ax=ax_rho_M)
    plot_beta_prior('Rho X', model.rho_X, alpha_bar_X, beta_bar_X, ax=ax_rho_G)
    plot_beta_prior('Rho G', model.rho_G, alpha_bar_G, beta_bar_G, ax=ax_rho_X)
    plot_pi_prior('pi_M', model.pi_M, ax=ax_pi_M)
    plot_pi_prior('pi_X', model.pi_X, ax=ax_pi_X)
    plot_pi_prior('pi_G', model.pi_G, ax=ax_pi_G)

    if save_path is not None:
        plt.savefig(save_path)



