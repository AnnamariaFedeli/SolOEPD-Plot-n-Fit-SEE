import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as pltt
import make_the_fit_tripl as fitting
import combining_files as comb


def run_all(
    path,
    data,
    savefig,
    plot_title='',
    x_label='Energy [MeV]',
    y_label='Intensity [/]',
    legend_title='',
    data_label_for_legend=None,
    which_fit='best',
    e_min=None,
    e_max=None,
    g1_guess=-1.9,
    g2_guess=-2.5,
    g3_guess=-4,
    c1_guess=1000,
    alpha_guess=10,
    beta_guess=10,
    break_guess_low=0.6,
    break_guess_high=1.2,
    cut_guess=1.2,
    exponent_guess=2,
    use_random=True,
    iterations=20,
    legend_details=False,
    ):
    """
    General-purpose function for combining datasets, fitting a spectrum,
    plotting the individual datasets, and optionally saving the figure.

    Parameters
    ----------
    path : str
        Base output path.

    data : list of pandas.DataFrame
        List of datasets. Each dataset must contain four columns:

            x
            y
            x error
            y error

        x is the energy and y is the intensity.

    savefig : bool
        Whether to save the resulting plot.

    plot_title : str
        Plot title and base filename.

    x_label : str
        X-axis label. Defaults to 'Energy [MeV]'.

    y_label : str
        Y-axis label. Defaults to 'Intensity [/]'.

    legend_title : str
        Title of the plot legend.

    data_label_for_legend : list of str or None
        Labels for the individual datasets.

    which_fit : str
        Fit selection passed to MAKE_THE_FIT.

    e_min, e_max : float or None
        Energy limits passed to MAKE_THE_FIT.

    g1_guess, g2_guess, g3_guess : float
        Initial power-law slope guesses.

    c1_guess : float
        Initial normalization guess.

    alpha_guess, beta_guess : float
        Initial smoothness parameter guesses.

    break_guess_low, break_guess_high : float
        Initial break-energy guesses.

    cut_guess : float
        Initial exponential cutoff guess.

    exponent_guess : float
        Initial cutoff exponent guess.

    use_random : bool
        Whether randomized initial guesses are used.

    iterations : int
        Number of randomized fitting iterations.

    legend_details : bool
        Whether detailed fit information is shown in the legend.
    """

    # INPUT VALIDATION
    
    if not isinstance(data, (list, tuple)) or len(data) == 0:
        raise ValueError("data must be a non-empty list or tuple of DataFrames.")

    if data_label_for_legend is None:
        data_label_for_legend = [f'Data {i + 1}' for i in range(len(data))]

    if len(data_label_for_legend) != len(data):
        raise ValueError("data_label_for_legend must contain one label for each dataset.")

    # OUTPUT PATHS
    
    title_from_path = path

    fit_var_dir = os.path.join(title_from_path, 'fit-result-variables')
    plot_dir = os.path.join(title_from_path, 'plots')

    os.makedirs(fit_var_dir, exist_ok=True)

    if savefig:
        os.makedirs(plot_dir, exist_ok=True)

    fit_var_path = os.path.join(fit_var_dir, f'{plot_title}-fit-result-variables_{which_fit}.csv')

    # PREPARE DATA
    
    prepared_data = []

    for dataset in data:

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Every element of data must be a pandas DataFrame.")

        df = dataset.copy()

        if len(df.columns) != 4:
            raise ValueError(
                "Each dataset must contain exactly four columns: "
                "'x', 'y', 'x error', 'y error'.")

        # Standardize column names without modifying the original DataFrame.
        df.columns = ['x', 'y', 'x error', 'y error']

        # Remove non-positive intensities.
        df = df[df['y'] > 0].copy()

        # Remove rows containing NaNs/infinite values in relevant columns.
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['x', 'y', 'x error', 'y error'])

        df.reset_index(drop=True, inplace=True)

        if len(df) > 0:
            prepared_data.append(df)

    if len(prepared_data) == 0:
        raise ValueError("No valid data remain after cleaning.")

    # COMBINE DATASETS
    
    if len(prepared_data) > 1:

        all_data = comb.combine_data_general(
            prepared_data,
            path + f'-combined-data-{plot_title}-{which_fit}.csv')

        all_data = all_data.copy()
        all_data.columns = ['x', 'y', 'x error', 'y error']

    else:

        all_data = prepared_data[0].copy()

    # Remove invalid combined data.
    all_data = all_data.replace([np.inf, -np.inf], np.nan)
    all_data = all_data.dropna(
        subset=['x', 'y', 'x error', 'y error'])

    all_data = all_data[all_data['y'] > 0].copy()
    all_data.reset_index(drop=True, inplace=True)

    if len(all_data) == 0:
        raise ValueError("No valid combined data remain.")

    # EXTRACT COMBINED DATA FOR FIT
    
    x_data = all_data['x']
    x_data_err = all_data['x error']

    y_data = all_data['y']
    y_data_err = all_data['y error']

    # FIT RANGE
    
    fit_e_min = min(x_data) if e_min is None else e_min
    fit_e_max = max(x_data) if e_max is None else e_max

    # FIGURE
    
    fig, ax = plt.subplots(1, figsize=(6, 5), dpi=200)

    # FIT
    
    fitting.MAKE_THE_FIT(
        x_data,
        y_data,
        x_data_err,
        y_data_err,
        ax,
        direction='sun',
        e_min=fit_e_min,
        e_max=fit_e_max,
        which_fit=which_fit,
        g1_guess=g1_guess,
        g2_guess=g2_guess,
        g3_guess=g3_guess,
        alpha_guess=alpha_guess,
        beta_guess=beta_guess,
        break_low_guess=break_guess_low,
        break_high_guess=break_guess_high,
        cut_guess=cut_guess,
        c1_guess=c1_guess,
        exponent_guess=exponent_guess,
        use_random=use_random,
        iterations=iterations,
        path=None,
        path2=fit_var_path,
        detailed_legend=legend_details
        )

    # PLOT INDIVIDUAL DATASETS
    
    colors = [
        'red',
        'darkorange',
        'maroon',
        'blue',
        'purple',
        'green',
        'black',
        'cyan',
        ]

    for i, dataset in enumerate(prepared_data):

        x = dataset['x']
        x_err = dataset['x error']

        y = dataset['y']
        y_err = dataset['y error']

        color = colors[i % len(colors)]

        ax.errorbar(
            x,
            y,
            yerr=y_err,
            xerr=x_err,
            marker='o',
            markersize=3,
            linestyle='',
            color=color,
            alpha=0.5,
            label=data_label_for_legend[i],
            zorder=-1
            )

    # AXES
    
    x_range_min = min(all_data['x'])
    x_range_max = max(all_data['x'])

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(x_range_min / 2, x_range_max * 1.5)

    # Minor ticks on logarithmic axes.
    locmin = pltt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)

    ax.xaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_locator(locmin)

    ax.xaxis.set_minor_formatter(pltt.NullFormatter())
    ax.yaxis.set_minor_formatter(pltt.NullFormatter())

    # LABELS / TITLE
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(plot_title)

    # LEGEND
    
    ax.legend(title=legend_title, prop={'size': 7})

    # SAVE
    
    if savefig:

        plot_path = os.path.join(plot_dir, f'{plot_title}-fit-plot_{which_fit}.png')

        fig.savefig(plot_path, dpi=300, bbox_inches='tight')

    # SHOW
    
    plt.show()

    return fig, ax, all_data

