import datetime as dt
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as pltt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from sunpy.coordinates import get_horizons_coord

import run_associated_functions as runn
import combining_files as comb
import make_the_fit_tripl as fitting
import savecsv as save



def FIT_DATA(path, date, averaging, fit_type, step=True, ept=True, het=True,
             direction='sun', which_fit='best', sigma=3, rel_err=0.5,
             frac_nan_threshold=0.9, fit_to='peak', e_min=None, e_max=None,
             g1_guess=-1.9, g2_guess=-2.5, g3_guess=-4, I0_guess=1000, E_0 = 0.1,
             alpha_guess=10, beta_guess=10, break_guess_low=0.6,
             break_guess_high=1.2, cut_guess=1.2, exponent_guess=2,
             use_random=True, iterations=20, shift_step_data=False,
             auto_shift=False, shift_factor=None, save_fig=True,
             save_pickle=False, save_fit_variables=True, save_fitrun=True,
             legend_details=False, legend_outside=False, no_legend=False,
             bg_subtraction=True,
             fit_to_separate_folder=False, centre_pix=False, fsize=12,
             channels_to_exclude=None, detailed_plot=False,
            
             do_not_plot_bad_channels=False, title_of_plot=None,
             make_the_fit=True, quality_factor=None):
    """Fit Solar Orbiter proton data.

    Args:
        path (str): Path to the folder containing the data and where fit/variable
            files will be saved.
        date (datetime or str): dt.datetime(yyyy, mm, dd, HH, MM) or
            'yyyy-mm-dd-HHMM'.
        averaging (int): The averaging of the data.
        fit_type (str): Fit type options: 'step', 'ept', 'het', 'step_ept',
            'step_ept_het', 'ept_het'.
        step (bool, optional): If True, include STEP data in the plot.
        ept (bool, optional): If True, include EPT data in the plot.
        het (bool, optional): If True, include HET data in the plot.
        direction (str, optional): Direction of the data.
        which_fit (str, optional): Fit selection. Options include 'single',
            'double', 'best_sb', 'cut', 'double_cut', 'best_cb', 'triple',
            and 'best'.
        sigma (int, optional): Standard deviation from the background.
        rel_err (float, optional): Maximum allowed relative uncertainty of the
            background-subtracted flux.
        frac_nan_threshold (float, optional): Minimum fraction of valid data
            within the search window required for a reliable peak.
        fit_to (str, optional): Fit either the peak or average flux within the
            search window.
        e_min (float, optional): Lower energy limit for the fit.
        e_max (float, optional): Upper energy limit for the fit.
        g1_guess (float, optional): Initial guess for the first spectral slope.
        g2_guess (float, optional): Initial guess for the second spectral slope.
        g3_guess (float, optional): Initial guess for the third spectral slope.
        I0_guess (float, optional): Initial intensity value at 0.1 MeV.
        E_0 (float, optional): The energy (in MeV) that corresponds to intensity at I_0. Defaults to E_0=0.1 (MeV).
        alpha_guess (float, optional): Smoothness of the transition between
            gamma1 and gamma2.
        beta_guess (float, optional): Smoothness of the transition between
            gamma3 and gamma2.
        break_guess_low (float, optional): Initial guess for the first spectral break.
        break_guess_high (float, optional): Initial guess for the second spectral break.
        cut_guess (float, optional): Initial guess for the exponential cutoff.
        exponent_guess (float, optional): Initial guess for the exponential exponent.
        use_random (bool, optional): If True, additional random initial values
            close to the guesses are used during fitting.
        iterations (int, optional): Number of fitting iterations using random
            initial values.
        shift_step_data (bool, optional): If True, shift STEP data by a
            multiplicative intensity factor.
        auto_shift (bool, optional): If True, calculate the STEP shift factor
            automatically.
        shift_factor (float, optional): Factor used to shift STEP data when
            auto_shift is False.
        save_fig (bool, optional): If True, save the fit figure.
        save_pickle (bool, optional): If True, save the fit result as a pickle file.
        save_fit_variables (bool, optional): If True, save the variables from
            the final fit.
        save_fitrun (bool, optional): If True, save the fitting parameters and
            run information.
        legend_details (bool, optional): If True, include additional information
            in the legend.
        bg_subtraction (bool, optional): If True, plot/fit background-subtracted
            flux.
        fit_to_separate_folder (bool, optional): If True, save the fit in a
            plots folder.
        centre_pix (bool, optional): If True, use the centre-pixel data.
        fsize (int, optional): Font size used for plotting.
        channels_to_exclude (list, optional): Channel indices to exclude from
            the fit.
        detailed_plot (bool, optional): If True, include detailed diagnostic plots.
        legend_outside (bool, optional): If True, place the legend outside the plot.
        no_legend (bool, optional): If True, do not display a legend.
        do_not_plot_bad_channels (bool, optional): If True, do not plot channels
            rejected by the quality cuts.
        title_of_plot (str, optional): Custom title for the fit plot.
        make_the_fit (bool, optional): If True, perform the fit. If False, only
            produce the associated data plot.
        quality_factor (dict or None, optional): Quality-factor results for STEP,
            EPT, and HET.
    """
    if not isinstance(fit_to, str) or fit_to not in ['peak', 'average']:
        raise ValueError("fit_to must be a string: 'peak' or 'average'")

    
    date_string = ''
    folder_time = date

    if isinstance(date, str):
        date_string = date[:-5]
    else:
        date_string = str(date.date())
        folder_time = str(date)[:-3].replace(' ', '-').replace(':', '')

    separator = ';'
    averaging_str = 'no' if averaging is None else str(averaging)
    pix = '-centre_pix' if centre_pix else ''

    step_file_name = (
        f'proton_data-{date_string}-STEP-{direction}-L2-'
        f'{averaging_str}_averaging{pix}.csv'
    )
    ept_file_name = (
        f'proton_data-{date_string}-EPT-{direction}-L2-'
        f'{averaging_str}_averaging.csv'
    )
    het_file_name = (
        f'proton_data-{date_string}-HET-{direction}-L2-'
        f'{averaging_str}_averaging.csv'
    )

    make_fit = make_the_fit

    fit_to_comb = fit_to.capitalize()

    intensity_label = 'Intensity\n/(s cm² sr MeV)'
    energy_label = 'Energy (MeV)'
    peak_info = f'{fit_to} spectrum'
    legend_title = 'Protons'
    data_product = 'l2'

    date_str = str(date)[:-3]

    try:
        pos = get_horizons_coord('Solar Orbiter', date)
        dist = np.round(pos.radius.value, 2)
    except Exception as e:
        print(f'Warning: Could not retrieve spacecraft position ({e})')
        dist = None

    # <---------------------------------------------------------------LOADING AND SAVING FILES------------------------------------------------------------------->

    data_list = []
    step_shift_factor = 1

    # SHIFTING DATA
    if step:
        step_data = pd.read_csv(f'{path}{step_file_name}', sep=separator)

        if ept:
            ept_data = pd.read_csv(f'{path}{ept_file_name}', sep=separator)

            if shift_step_data:
                if auto_shift:
                    step_shift_factor = runn.calculate_shift_factor(
                        step_data, ept_data, sigma, rel_err,
                        frac_nan_threshold, fit_to
                    )
                else:
                    step_shift_factor = shift_factor

                print(f'SHIFT FACTOR: {step_shift_factor}')

                step_data[f'Bg_subtracted_{fit_to}'] /= step_shift_factor
                step_data[f'Flux_{fit_to}'] /= step_shift_factor
                step_data['Background_flux'] /= step_shift_factor

        data_list.append(step_data)

    if ept:
        ept_data = pd.read_csv(f'{path}{ept_file_name}', sep=separator)
        data_list.append(ept_data)

    if het:
        het_data = pd.read_csv(f'{path}{het_file_name}', sep=separator)
        data_list.append(het_data)

    all_file = f'{path}{date_string}-all-l2-{direction}-{averaging_str}.csv'

    data = comb.combine_data(
        data_list, all_file,
        sigma=sigma, rel_err=rel_err,
        frac_nan_threshold=frac_nan_threshold,
        fit_to=fit_to_comb,
        channels_to_exclude=channels_to_exclude
    )
    data = pd.read_csv(all_file, sep=separator)

    if step and ept:
        step_ept_file = f'{path}{date_string}-step_ept-l2-{averaging_str}.csv'

        step_ept_data = comb.combine_data(
            [step_data, ept_data], step_ept_file,
            sigma=sigma, rel_err=rel_err,
            frac_nan_threshold=frac_nan_threshold,
            fit_to=fit_to_comb,
            channels_to_exclude=channels_to_exclude
        )
        step_ept_data = pd.read_csv(step_ept_file, sep=separator)

    if ept and het:
        ept_het_file = f'{path}{date_string}-ept_het-{direction}-l2-{averaging_str}.csv'

        ept_het_data = comb.combine_data(
            [ept_data, het_data], ept_het_file,
            sigma=sigma, rel_err=rel_err,
            frac_nan_threshold=frac_nan_threshold,
            fit_to=fit_to_comb,
            channels_to_exclude=channels_to_exclude
        )
        ept_het_data = pd.read_csv(ept_het_file, sep=separator)

    # Saving the contaminated data so it can be plotted separately,
    # then deleting it from the data so it doesn't overlap.
    contaminated_data_sigma = comb.extract_low_sigma_rows(
        data_list, sigma=sigma, fit_to=fit_to_comb
    )
    contaminated_data_nan = comb.extract_nan_heavy_rows(
        data_list, frac_nan_threshold=frac_nan_threshold
    )
    contaminated_data_rel_err = comb.extract_high_rel_err_rows(
        data_list, rel_err=rel_err
    )

    contaminated_data = pd.concat([
        contaminated_data_sigma,
        contaminated_data_nan,
        contaminated_data_rel_err
    ])
    contaminated_data.reset_index(drop=True, inplace=True)

    # Deleting bad data so it doesn't overplot.
    if step:
        step_data = comb.delete_bad_data(
            step_data, sigma=sigma, rel_err=rel_err,
            frac_nan_threshold=frac_nan_threshold,
            fit_to=fit_to_comb,
            channels_to_exclude=channels_to_exclude
        )

    if ept:
        ept_data = comb.delete_bad_data(
            ept_data, sigma=sigma, rel_err=rel_err,
            frac_nan_threshold=frac_nan_threshold,
            fit_to=fit_to_comb,
            channels_to_exclude=channels_to_exclude
        )

    if het:
        het_data = comb.delete_bad_data(
            het_data, sigma=sigma, rel_err=rel_err,
            frac_nan_threshold=frac_nan_threshold,
            fit_to=fit_to_comb,
            channels_to_exclude=channels_to_exclude
        )

    # <---------------------------------------------------------------------DATA--------------------------------------------------------------------->

    def extract_energy(df):
        """Return energy and asymmetric energy errors."""
        return (
            df['Primary_energy'],
            [df['Energy_error_low'], df['Energy_error_high']]
        )

    def extract_flux(df, flux_col, err_col):
        """Return flux and uncertainty."""
        return df[flux_col], df[err_col]

    # ENERGY DATA
    spec_energy, energy_err = extract_energy(data)

    if step and ept:
        spec_energy_step_ept, energy_err_step_ept = extract_energy(step_ept_data)

    if ept and het:
        spec_energy_ept_het, energy_err_ept_het = extract_energy(ept_het_data)

    if step:
        spec_energy_step, energy_err_step = extract_energy(step_data)

    if ept:
        spec_energy_ept, energy_err_ept = extract_energy(ept_data)

    if het:
        spec_energy_het, energy_err_het = extract_energy(het_data)

    # Contaminated data
    spec_energy_c, energy_err_c = extract_energy(contaminated_data)
    spec_energy_c_sigma, energy_err_c_sigma = extract_energy(contaminated_data_sigma)
    spec_energy_c_nan, energy_err_c_nan = extract_energy(contaminated_data_nan)
    spec_energy_c_rel_err, energy_err_c_rel_err = extract_energy(contaminated_data_rel_err)

    # There is no difference between average and peak uncertainty,
    # so Backsub_peak_uncertainty is used for both peak and average fits.
    if bg_subtraction:
        flux_col = f'Bg_subtracted_{fit_to}'
        err_col = 'Backsub_peak_uncertainty'
    else:
        flux_col = f'Flux_{fit_to}'
        err_col = 'Peak_proton_uncertainty'

    # INTENSITY DATA
    spec_flux, flux_err = extract_flux(data, flux_col, err_col)

    if step and ept:
        spec_flux_step_ept, flux_err_step_ept = extract_flux(
            step_ept_data, flux_col, err_col
        )

    if ept and het:
        spec_flux_ept_het, flux_err_ept_het = extract_flux(
            ept_het_data, flux_col, err_col
        )

    if step:
        spec_flux_step, flux_err_step = extract_flux(
            step_data, flux_col, err_col
        )

    if ept:
        spec_flux_ept, flux_err_ept = extract_flux(
            ept_data, flux_col, err_col
        )

    if het:
        spec_flux_het, flux_err_het = extract_flux(
            het_data, flux_col, err_col
        )

    spec_flux_c, flux_err_c = extract_flux(contaminated_data, flux_col, err_col)
    spec_flux_c_sigma, flux_err_c_sigma = extract_flux(
        contaminated_data_sigma, flux_col, err_col
    )
    spec_flux_c_nan, flux_err_c_nan = extract_flux(
        contaminated_data_nan, flux_col, err_col
    )
    spec_flux_c_rel_err, flux_err_c_rel_err = extract_flux(
        contaminated_data_rel_err, flux_col, err_col
    )

    # ENERGY RANGE SELECTION
    energy_map = {
        'step': spec_energy_step if step else None,
        'ept': spec_energy_ept if ept else None,
        'het': spec_energy_het if het else None,
        'step_ept': spec_energy_step_ept if (step and ept) else None,
        'ept_het': spec_energy_ept_het if (ept and het) else None,
        'step_ept_het': spec_energy if (step and ept and het) else None
    }

    selected_energy = energy_map.get(fit_type)

    if selected_energy is None:
        raise ValueError(f'Invalid fit_type: {fit_type}')

    min_energy = min(selected_energy) if e_min is None else e_min
    max_energy = max(selected_energy) if e_max is None else e_max

    # <---------------------------------------------------------------- QUALITY FACTORS ---------------------------------------------------------------->

    qf_step = qf_ept = qf_het = None
    qf_step_av = qf_ept_av = qf_het_av = None

    if quality_factor is not None:
        expected_qf = sum([step, ept, het])

        if len(quality_factor) < expected_qf:
            raise ValueError(
                'Not enough quality_factor entries for selected instruments'
            )

        def get_qf(name):
            qf_vals = quality_factor[f'QF {name} all channels']
            qf_avg_raw = quality_factor[f'QF {name} average']

            if hasattr(qf_avg_raw, 'iloc'):
                qf_avg = qf_avg_raw.iloc[0]
            else:
                qf_avg = qf_avg_raw

            return qf_vals, qf_avg

        if step:
            qf_step, qf_step_av = get_qf('STEP')

        if ept:
            qf_ept, qf_ept_av = get_qf('EPT')

        if het:
            qf_het, qf_het_av = get_qf('HET')

    # <--------------------------------------------------------------- FILE PATHS ---------------------------------------------------------------->

    color = {
        'sun': 'crimson',
        'asun': 'orange',
        'north': 'darkslateblue',
        'south': 'c'
    }

    def build_path(suffix):
        """Build standardized output filenames."""
        return (
            f'{path}{folder_time}-{suffix}_{fit_type}-{fit_to}-{which_fit}'
            f'-l2-{averaging_str}_averaging-{direction}{pix}'
        )

    pickle_path = build_path('pickle') + '.p' if save_pickle else None
    fit_var_path = (
        build_path('fit-result-variables') + '.csv'
        if save_fit_variables else None
    )

    fitrun_path = None
    if save_fitrun:
        fitrun_path = build_path('all-fit-variables') + '.csv'

        save.save_info_fit(
            fitrun_path, date_string, averaging, direction, data_product, dist,
            step, ept, het, sigma, rel_err, frac_nan_threshold,
            False, step_shift_factor, fit_type, fit_to, which_fit,
            min_energy, max_energy, g1_guess, g2_guess, I0_guess, E_0,
            alpha_guess, break_guess_low, cut_guess,
            use_random, iterations,
            qf_step_av, qf_ept_av, qf_het_av, centre_pix
        )

        qf_path = build_path('quality-factor') + '.csv'
        save.save_quality_factor(qf_path, qf_step, qf_ept, qf_het)

    # <----------------------------------------------------------------------FIT AND PLOT------------------------------------------------------------------->

    f, ax = plt.subplots(1, figsize=(8, 6), dpi=300)

    distance = f' (R={dist} au)'

    if legend_details:
        if bg_subtraction:
            ax.plot([], [], ' ', label='bg subtraction on')
        else:
            ax.plot([], [], ' ', label='bg subtraction off')

        if shift_step_data:
            ax.plot(
                [], [], ' ',
                label=f'Shift factor (STEP) {np.round(step_shift_factor, 2)}'
            )

    # FITTING
    if make_fit:
        fit_map = {
            'step': (
                spec_energy_step, spec_flux_step,
                energy_err_step, flux_err_step, 'STEP'
            ) if step else None,

            'ept': (
                spec_energy_ept, spec_flux_ept,
                energy_err_ept, flux_err_ept, 'EPT'
            ) if ept else None,

            'het': (
                spec_energy_het, spec_flux_het,
                energy_err_het, flux_err_het, 'HET'
            ) if het else None,

            'step_ept': (
                spec_energy_step_ept, spec_flux_step_ept,
                energy_err_step_ept, flux_err_step_ept,
                'STEP and EPT'
            ) if (step and ept) else None,

            'ept_het': (
                spec_energy_ept_het, spec_flux_ept_het,
                energy_err_ept_het, flux_err_ept_het,
                'EPT and HET'
            ) if (ept and het) else None,

            'step_ept_het': (
                spec_energy, spec_flux, energy_err, flux_err,
                'STEP, EPT and HET'
            ) if (step and ept and het) else None
        }

        if fit_type not in fit_map or fit_map[fit_type] is None:
            raise ValueError(f'Invalid fit_type: {fit_type}')

        energy, flux, energy_err_local, flux_err_local, plot_title_label = fit_map[fit_type]

        plot_title = f'Solar Orbiter {distance} {plot_title_label}'

        fitting.MAKE_THE_FIT(
            energy, flux, energy_err_local[1], flux_err_local, ax,
            direction=direction, e_min=e_min, e_max=e_max,
            which_fit='single' if fit_type == 'het' else which_fit,
            g1_guess=g1_guess, g2_guess=g2_guess, g3_guess=g3_guess,
            alpha_guess=alpha_guess, beta_guess=beta_guess,
            break_low_guess=break_guess_low,
            break_high_guess=break_guess_high,
            cut_guess=cut_guess, I0_guess=I0_guess, E_0= E_0,
            exponent_guess=exponent_guess, use_random=use_random,
            iterations=iterations, path=pickle_path, path2=fit_var_path,
            detailed_legend=legend_details
        )

    def plot_errorbar(x, y, yerr, xerr, **kwargs):
        if len(x) > 0:
            ax.errorbar(
                x, y, yerr=yerr, xerr=xerr,
                marker='o', linestyle='', markersize=3,
                zorder=-1, **kwargs
            )

    # PLOTTING DATA
    if step:
        plot_errorbar(
            spec_energy_step, spec_flux_step,
            flux_err_step, energy_err_step,
            color='darkorange', label='STEP'
        )

    if ept:
        plot_errorbar(
            spec_energy_ept, spec_flux_ept,
            flux_err_ept, energy_err_ept,
            color=color[direction], label='EPT ' + direction
        )

    if het:
        plot_errorbar(
            spec_energy_het, spec_flux_het,
            flux_err_het, energy_err_het,
            color='maroon', label='HET ' + direction
        )

    if not do_not_plot_bad_channels:
        plot_errorbar(
            spec_energy_c, spec_flux_c,
            flux_err_c, energy_err_c,
            color='gray',
            label='excluded from fit' if make_fit else 'cont. data'
        )

    # Detailed view of the different categories of excluded data.
    if detailed_plot and not do_not_plot_bad_channels:
        plot_errorbar(
            spec_energy_c_sigma, spec_flux_c_sigma,
            flux_err_c_sigma, energy_err_c_sigma,
            label='Sigma below ' + str(sigma)
        )

        plot_errorbar(
            spec_energy_c_nan, spec_flux_c_nan,
            flux_err_c_nan, energy_err_c_nan,
            label='excluded (NaNs)'
        )

        plot_errorbar(
            spec_energy_c_rel_err, spec_flux_c_rel_err,
            flux_err_c_rel_err, energy_err_c_rel_err,
            label='excluded (rel err)'
        )

    # Background flux
    if step:
        plot_errorbar(
            spec_energy_step, step_data['Background_flux'],
            step_data['Bg_proton_uncertainty'], energy_err_step,
            color='darkorange', alpha=0.3
        )

    if ept:
        plot_errorbar(
            spec_energy_ept, ept_data['Background_flux'],
            ept_data['Bg_proton_uncertainty'], energy_err_ept,
            color=color[direction], alpha=0.3
        )

    if het:
        plot_errorbar(
            spec_energy_het, het_data['Background_flux'],
            het_data['Bg_proton_uncertainty'], energy_err_het,
            color='maroon', alpha=0.3
        )

    if not do_not_plot_bad_channels:
        background_color = 'gray' if make_fit else 'maroon'

        plot_errorbar(
            spec_energy_c, contaminated_data['Background_flux'],
            contaminated_data['Bg_proton_uncertainty'], energy_err_c,
            color=background_color, alpha=0.3
        )

    # Proton energy ranges
    step_energy_range = [0.004323343613, 0.07803193193]
    ept_energy_range = [2., 5.]
    het_energy_range = [10., 100.]

    e_range_min = step_energy_range[0]
    e_range_max = het_energy_range[1]

    ax.set_xscale('log')
    ax.set_yscale('log')

    locmin = pltt.LogLocator(
        base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12
    )

    ax.set_xlim(
        e_range_min - e_range_min / 2,
        e_range_max + e_range_max / 2
    )

    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(pltt.NullFormatter())

    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.ylabel(intensity_label, fontsize=fsize)
    plt.xlabel(energy_label, fontsize=fsize)

    if title_of_plot is not None:
        final_title = title_of_plot
    elif centre_pix:
        final_title = (
            plot_title + '  ' + peak_info + '\n' +
            date_str + '  ' + averaging_str +
            '  averaging, centre pixels'
        )
    else:
        final_title = (
            plot_title + '  ' + peak_info + '\n' +
            date_str + '  ' + averaging_str + '  averaging'
        )

    plt.title(final_title, fontsize=fsize + 2)

    legend = None

    if not no_legend:
        if legend_outside:
            legend = ax.legend(
                title=legend_title,
                prop={'size': 7},
                fontsize=fsize - 2,
                title_fontsize=fsize,
                bbox_to_anchor=(1.02, 1),
                loc='upper left'
            )
        else:
            legend = ax.legend(
                title=legend_title,
                prop={'size': 7},
                fontsize=fsize - 2,
                title_fontsize=fsize
            )

    # SAVING
    plot_path = path
    if fit_to_separate_folder:
        plot_path = path + 'plots/'
        os.makedirs(plot_path, exist_ok=True)

    def savefig_safe(filename):
        if legend is not None:
            plt.savefig(
                filename, dpi=300, bbox_inches='tight',
                bbox_extra_artists=[legend]
            )
        else:
            plt.savefig(filename, dpi=300)

    if save_fig:
        base = f'{plot_path}protons-{date_string}-{averaging_str}'

        if make_fit:
            suffix = f'-{direction}-{which_fit}-{fit_type}-{fit_to}'

            if bg_subtraction:
                suffix += '-bg_sub'

            savefig_safe(base + suffix + pix)

        else:
            combo = '_'.join([
                key for key, value in {
                    'step': step,
                    'ept': ept,
                    'het': het
                }.items() if value
            ])

            savefig_safe(base + f'-no_fit-{combo}' + pix)

    plt.show()


