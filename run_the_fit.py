import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as pltt
from matplotlib import font_manager
font_manager.fontManager.ttflist
from matplotlib import rc
#from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from sunpy.coordinates import get_horizons_coord
import make_the_fit_tripl as fitting
import savecsv as save
import combining_files as comb
import os
from tabulate import tabulate
import shutil
#import matplotlib.font_manager
from IPython.core.display import HTML

def make_html(fontname):
    return f"<p>{fontname}: <span style='font-family:{fontname}; font-size: 24px;'>{fontname}</p>"


if __name__ == "__main__":
    fonts = sorted({f.name for f in font_manager.fontManager.ttflist})
    code = "\n".join([make_html(font) for font in fonts])

    HTML(f"<div style='column-count: 2;'>{code}</div>")

# <--------------------------------------------------------------- ALL NECESSARY INPUTS HERE ----------------------------------------------------------------->

def quality_factor_PA_coverage(data, coverage, direction = 'sun', angle = 180): 
    qf = [] 

    for j in range(0, len(data[1])): 
        df = coverage[direction] 
        df = df.reset_index() 
        df = df.drop(np.where(df['EPOCH'] < data[2][0][j])[0]) 
        df.reset_index(drop = True, inplace = True) 
        df = df.drop(np.where(df['EPOCH'] > data[2][1][j])[0]) 
        df.reset_index(drop = True, inplace = True) 
        factors = [] 
        for i in range(0,len(df)): 
            r = df.center[i] 
            if angle == 180: 
                r = 180-r 
            if r <=15.: 
                factors.append(100) 
            elif r>15: 
                f = np.exp(-np.square(r-12)/2*0.0007)*100 
                factors.append(f) 
            else: 
                factors.append(0) 

        qf.append(sum(factors)/len(factors)) 
                #qf = sum(factors)/len(factors) 
                #print(factors) 
        
    quality_factor = sum(qf)/len(qf) 
    return [qf, quality_factor]


def compute_quality_factors(plot_pa, step, ept, het, pixels, data_step, data_step_pix, data_ept, data_het, coverage_ept, coverage_het, direction, angle):
    if not plot_pa:
        return None, None, None, None

    results = {}
    results_pix = {}

    def process(name, data, coverage):
        qf_vals, qf_avg = quality_factor_PA_coverage(data, coverage, direction=direction, angle=angle)

        results[f"QF {name} average"] = qf_avg
        results[f"QF {name} all channels"] = qf_vals

        return qf_vals, qf_avg

    # --- STEP ---
    if step:
        process("STEP", data_step, coverage_ept)

        if pixels:
            qf_vals, qf_avg = quality_factor_PA_coverage(data_step_pix, coverage_ept, direction=direction, angle=angle)
            results_pix["QF STEP average"] = qf_avg
            results_pix["QF STEP all channels"] = qf_vals

    # --- EPT ---
    if ept:
        qf_vals, qf_avg = process("EPT", data_ept, coverage_ept)

        if pixels:
            results_pix["QF EPT average"] = qf_avg
            results_pix["QF EPT all channels"] = qf_vals

    # --- HET ---
    if het:
        qf_vals, qf_avg = process("HET", data_het, coverage_het)

        if pixels:
            results_pix["QF HET average"] = qf_avg
            results_pix["QF HET all channels"] = qf_vals

    # --- convert to pandas Series (exactly like your notebook) ---
    d = {k: pd.Series(v) for k, v in results.items()}
    d_pix = {k: pd.Series(v) for k, v in results_pix.items()} if pixels else None

    return d, d_pix, results, results_pix

def print_channel(step=None, ept=None, het=None):
    """
    Print channel indices and corresponding primary energies for each instrument.

    Parameters
    ----------
    step, ept, het : pandas.DataFrame or None
        DataFrames containing a 'Primary_energy' column.
        Each row corresponds to one energy channel.

    Notes
    -----
    - Channels are indexed consecutively across instruments.
    - This function is intended for visualization/debugging only.
    """

    instruments = [('STEP', step), ('EPT', ept), ('HET', het)]

    start_idx = 0

    for name, df in instruments:
        if df is None:
            continue

        out = pd.DataFrame({'Channel': range(start_idx, start_idx + len(df)), 'Primary Energy [MeV]': df['Primary_energy']})

        print(f'\n{name} CHANNELS')
        print(out.to_string(index=False))

        start_idx += len(df)


def calculate_shift_factor(step_data, ept_data, sigma, rel_err, frac_nan_threshold, fit_to):
    """
    Calculate shift factor between STEP and EPT intensities over a fixed energy range.

    Filters both datasets to the same energy window, combines channels, and computes
    the ratio of mean fluxes.

    Args:
        step_data (pd.DataFrame): STEP data with 'Primary_energy' and flux columns.
        ept_data (pd.DataFrame): EPT data with same structure.
        sigma (float): Used in channel combination.
        rel_err (float): Relative error threshold.
        frac_nan_threshold (float): NaN filtering threshold.
        fit_to (str): Flux column suffix (e.g., 'peak', 'avg').

    Returns:
        float: Shift factor (STEP / EPT), or 1 if calculation is not possible.
    """

    fit = fit_to.capitalize()

    # energy range 
    E_MIN = 0.037
    E_MAX = 0.057

    # filter STEP data
    data_step = step_data[(step_data['Primary_energy'] >= E_MIN) & (step_data['Primary_energy'] <= E_MAX)].reset_index(drop=True)

    data_step = comb.combine_data([data_step], path=None, sigma=sigma, rel_err=rel_err, frac_nan_threshold=frac_nan_threshold, leave_out_1st_het_chan=False, fit_to=fit)

    # determine number of STEP channels 
    if len(step_data['Primary_energy']) > 8:
        n_step_chans = 4
    else:
        n_step_chans = 1

    # filter EPT data
    data_ept = ept_data[(ept_data['Primary_energy'] >= E_MIN) & (ept_data['Primary_energy'] <= E_MAX)].reset_index(drop=True)

    data_ept = comb.combine_data([data_ept], path=None, sigma=sigma, rel_err=rel_err, frac_nan_threshold=frac_nan_threshold, leave_out_1st_het_chan=False, fit_to=fit)

    # sanity check
    if (len(data_step) < n_step_chans or len(data_ept) < 4 or data_step['Primary_energy'].iloc[-1] < data_ept['Primary_energy'].iloc[0]):
        print('There are too few energy channels to do a comparison and find a shift factor. '
              'If you still want to shift STEP data, please set automatic_shift to False '
              'and provide a shift_factor.')
        return 1

    # compute averages
    step_intensity_average = data_step['Flux_' + fit_to].mean()
    ept_intensity_average = data_ept['Flux_' + fit_to].mean()

    
    if (pd.isna(step_intensity_average) or pd.isna(ept_intensity_average) or ept_intensity_average == 0):
        print('Invalid intensity averages → cannot compute shift factor.')
        return 1

    shift_factor = step_intensity_average / ept_intensity_average

    print(shift_factor)
    return shift_factor	

def save_fit_and_run_variables_to_separate_folders(path, date, fit_var_file, run_var_file):
    """
    Copy fit and run variable files into dedicated subfolders.

    Creates 'fit_variables' and 'run_variables' directories if they do not exist,
    and copies the corresponding files from the date-specific folder.

    Args:
        path (str): Base directory path.
        date (str): Subfolder name (e.g., date string).
        fit_var_file (str): Filename of fit variables file.
        run_var_file (str): Filename of run variables file.
    """

    fitvariables = os.path.join(path, 'fit_variables')
    runvariables = os.path.join(path, 'run_variables')
    newpath = os.path.join(path, date)

    # ensure directories exist 
    os.makedirs(fitvariables, exist_ok=True)
    os.makedirs(runvariables, exist_ok=True)

    # build full source and destination paths
    src_fit = os.path.join(newpath, fit_var_file)
    dst_fit = os.path.join(fitvariables, fit_var_file)

    src_run = os.path.join(newpath, run_var_file)
    dst_run = os.path.join(runvariables, run_var_file)

    # copy files
    shutil.copy(src_fit, dst_fit)
    shutil.copy(src_run, dst_run)
    
def FIT_DATA(path, date, averaging, fit_type, step=True,
    ept=True, het=True, direction='sun', which_fit='best',
    channels_to_exclude=None, sigma=3, rel_err=0.5,
    frac_nan_threshold=0.9, fit_to='peak', e_min=None,
    e_max=None, g1_guess=-1.9, g2_guess=-2.5, g3_guess=-4,
    c1_guess=1000, alpha_guess=10, beta_guess=10, 
    break_guess_low=0.6, break_guess_high=1.2,
    cut_guess=1.2, exponent_guess=2, use_random=True,
    iterations=20, leave_out_1st_het_chan=True,
    shift_step_data=False, auto_shift=False,
    shift_factor=None, save_fig=True,
    save_pickle=False, save_fit_variables=True,
    save_fitrun=True, legend_details=False, detailed_plot = False,        
    ion_correction=True, bg_subtraction=True,
    fit_to_separate_folder=False, centre_pix=False, quality_factor=None,
    fsize=12, legend_outside=False, no_legend=False,
    do_not_plot_bad_channels=False, title_of_plot=None,
    make_the_fit=True):
    """
    Main driver function for loading data, performing spectral fits, and saving results.

    Parameters
    ----------
    path : str
        Directory where input data is stored and outputs will be saved.

    date : datetime or str
        Either datetime object or string in format 'yyyy-mm-dd-HHMM'.

    averaging : int
        Averaging applied to the data.

    fit_type : str
        Options:
        'step', 'ept', 'het', 'step_ept', 'step_ept_het', 'ept_het'

    step, ept, het : bool
        Include respective instrument data in the plot (not necessarily the fit).

    direction : str
        Viewing direction (default: 'sun').

    which_fit : str
        Fit selection mode:
        - 'single'
        - 'double'
        - 'best_sb'
        - 'cut'
        - 'double_cut'
        - 'best_cb'
        - 'triple'
        - 'best'

    sigma : int
        Standard deviation threshold for background.

    rel_err : float
        Relative uncertainty threshold.

    frac_nan_threshold : float
        Fraction of allowed NaNs in search window.

    fit_to : str
        'peak' or 'average'.

    e_min, e_max : float
        Energy bounds for fitting.

    g1_guess, g2_guess, g3_guess : float
        Power-law slopes (ordered g1 > g2 > g3).

    c1_guess : float
        Flux normalization at 0.1 MeV.

    alpha_guess, beta_guess : float
        Smoothness parameters for breaks.

    break_guess_low, break_guess_high : float
        Energy break guesses.

    cut_guess : float
        Exponential cutoff energy.

    exponent_guess : float
        Exponent for cutoff.

    use_random : bool
        Use randomized initial guesses.

    iterations : int
        Number of random trials.

    leave_out_1st_het_chan : bool
        Exclude first HET channel.

    shift_step_data : bool
        Apply intensity scaling to STEP data.

    auto_shift : bool
        Automatically compute shift factor.

    shift_factor : float
        Manual scaling factor.

    save_fig, save_pickle : bool
        Save outputs.

    save_fit_variables, save_fitrun : bool
        Save fit parameters and run configuration.

    legend_details : bool
        Show fit details in legend.

    detailed_plot : bool
            When True, enables additional diagnostic plotting and more rigorous inspection.

    ion_correction, bg_subtraction : bool
        Apply corrections.

    fit_to_separate_folder : bool
        Save plots in separate directory.

    centre_pix : bool
        Pixel-centering option.

    quality_factor : float or None
        Optional quality filter.

    fsize : int
        Font size.

    legend_outside, no_legend : bool
        Legend display options.

    do_not_plot_bad_channels : bool
        Skip bad channels in plotting.

    title_of_plot : str or None
        Custom plot title.

    make_the_fit : bool
        If False, skips fitting and only processes data.
    """
    if not isinstance(fit_to, str) or fit_to not in ['peak', 'average']:
        raise ValueError("fit_to must be a string: 'peak' or 'average'")

    
    #  Date handling
    date_string = ''
    folder_time = date
    separator = ';'

    if isinstance(date, str):
        date_string = date[:-5]
    else:
        date_string = str(date.date())
        folder_time = str(date)[:-3].replace(' ', '-').replace(':', '')


    #  Averaging handling 
    if averaging is None:
        averaging_str = 'no'
    else:
        averaging_str = str(averaging)


    # Pixel flag
    pix = '-centre_pix' if centre_pix else ''


    #  ------ FILE NAMES ------
    step_file_name = (f"electron_data-{date_string}-STEP-{direction}-L2-{averaging_str}_averaging{pix}.csv")

    if ion_correction:
        ept_file_name = (f"electron_data-{date_string}-EPT-{direction}-L2-{averaging_str}_averaging-ion_corr.csv")
    else:
        ept_file_name = (f"electron_data-{date_string}-EPT-{direction}-L2-{averaging_str}_averaging.csv")

    het_file_name = (f"electron_data-{date_string}-HET-{direction}-L2-{averaging_str}_averaging.csv")

    # <--------------------------------------------END OF NECESSARY INPUTS ----------------------------------------------->

    make_fit = make_the_fit

    # Fit label formatting
    fit_to_comb = fit_to.capitalize() if isinstance(fit_to, str) else fit_to

    # Plot labels 
    intensity_label = 'Intensity\n[1/(s cm² sr MeV)]'
    energy_label = 'Energy [MeV]'
    peak_info = f"{fit_to} spectrum"
    legend_title = 'Electrons'
    data_product = 'l2'


    # Date formatting 
    date_str = str(date)[:-3]


    # Spacecraft position
    try:
        pos = get_horizons_coord('Solar Orbiter', date)
        dist = np.round(pos.radius.value, 2)
    except Exception as e:
        print(f"Warning: Could not retrieve spacecraft position ({e})")
        dist = None

    # <-----------------------------------LOADING AND SAVING FILES + SHIFT DATA ------------------------------------>

    data_list = []
    step_shift_factor = shift_factor

    base_name = f"{path}{date_string}"
    all_file = f"{base_name}-all-l2-{direction}-{averaging}.csv"

    # -------- LOAD DATA ----------
    step_data = None
    ept_data = None
    het_data = None

    if step:
        step_data = pd.read_csv(f"{path}{step_file_name}", sep=separator)

    if ept:
        ept_data = pd.read_csv(f"{path}{ept_file_name}", sep=separator)

    if het:
        het_data = pd.read_csv(f"{path}{het_file_name}", sep=separator)


    # ------- SHIFT STEP DATA -------
    if step and ept and shift_step_data:

        if auto_shift:
            step_shift_factor = calculate_shift_factor(step_data, ept_data, sigma, rel_err, frac_nan_threshold, fit_to)
        else:
            step_shift_factor = shift_factor

        print(f"SHIFT FACTOR: {step_shift_factor}")

        columns_to_scale = [f'Bg_subtracted_{fit_to}', f'Flux_{fit_to}', 'Background_flux', f'{fit_to_comb}_electron_uncertainty', 'Bg_electron_uncertainty', 'Backsub_peak_uncertainty']

        for col in columns_to_scale:
            if col in step_data.columns:  # safer
                step_data[col] /= step_shift_factor

    # ----- DATA ------
    # Build data list 
    for dataset in [step_data, ept_data, het_data]:
        if dataset is not None:
            data_list.append(dataset)

    # Combine all data
    data = comb.combine_data(data_list, all_file, sigma=sigma, rel_err=rel_err, frac_nan_threshold=frac_nan_threshold, leave_out_1st_het_chan=leave_out_1st_het_chan, fit_to=fit_to_comb,channels_to_exclude=channels_to_exclude)
    data = pd.read_csv(all_file, sep=separator)


    # Telescope combinations
    if step and ept:
        step_ept_file = f"{base_name}-step_ept-l2-{averaging}.csv"

        step_ept_data = comb.combine_data([step_data, ept_data], step_ept_file, sigma=sigma, rel_err=rel_err, frac_nan_threshold=frac_nan_threshold, leave_out_1st_het_chan=leave_out_1st_het_chan, fit_to=fit_to_comb,channels_to_exclude=channels_to_exclude)

    if ept and het:
        ept_het_file = f"{base_name}-ept_het-{direction}-l2-{averaging}.csv"

        ept_het_data = comb.combine_data([ept_data, het_data], ept_het_file, sigma=sigma, rel_err=rel_err, frac_nan_threshold=frac_nan_threshold, leave_out_1st_het_chan=leave_out_1st_het_chan, fit_to=fit_to_comb, channels_to_exclude=channels_to_exclude)


    # Contaminated data
    contaminated_data_sigma = comb.extract_low_sigma_rows(data_list, sigma=sigma, leave_out_1st_het_chan=leave_out_1st_het_chan, fit_to=fit_to_comb)

    contaminated_data_nan = comb.extract_nan_heavy_rows(data_list, frac_nan_threshold=frac_nan_threshold, leave_out_1st_het_chan=leave_out_1st_het_chan)

    contaminated_data_rel_err = comb.extract_high_rel_err_rows(data_list, rel_err=rel_err, leave_out_1st_het_chan=leave_out_1st_het_chan)


    # ----- CHANNEL EXCLUSION ------
    step_channels_to_exclude = []
    ept_channels_to_exclude = []
    het_channels_to_exclude = []

    if channels_to_exclude is not None:

        excluded_channels = comb.excluded_channels_from_fit(data_list, channels_to_exclude)

        contaminated_data = pd.concat([contaminated_data_sigma, contaminated_data_nan, contaminated_data_rel_err, excluded_channels])

        for i in list(channels_to_exclude):

            if step and i <= len(step_data):
                step_channels_to_exclude.append(i)

            elif ept and i <= len(step_data) + len(ept_data):
                ept_channels_to_exclude.append(i - len(step_data))

            elif het:
                het_channels_to_exclude.append(i - (len(step_data) + len(ept_data)))

    else:
        contaminated_data = pd.concat([contaminated_data_sigma, contaminated_data_nan, contaminated_data_rel_err]).reset_index(drop=True)


    # Clean data
    if step:
        step_data = comb.delete_bad_data(step_data, sigma=sigma, rel_err=rel_err, frac_nan_threshold=frac_nan_threshold, fit_to=fit_to_comb, channels_to_exclude=step_channels_to_exclude)

    if ept:
        ept_data = comb.delete_bad_data(ept_data, sigma=sigma, rel_err=rel_err, frac_nan_threshold=frac_nan_threshold, fit_to=fit_to_comb, channels_to_exclude=ept_channels_to_exclude)

    if het:
        first_het_data = comb.extract_first_het_channel(het_data)

        het_data = comb.delete_bad_data(het_data, sigma=sigma, rel_err=rel_err, frac_nan_threshold=frac_nan_threshold, leave_out_1st_het_chan=leave_out_1st_het_chan, fit_to=fit_to_comb, channels_to_exclude=het_channels_to_exclude)
        
    # -------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # LEGACY / FALLBACK BLOCK
    # (Keep for debugging or reproducibility if main pipeline fails)
    # -------------------------------------------------------------------------
    #color = {'sun':'crimson','asun':'orange', 'north':'darkslateblue', 'south':'c'}

    # quick change for sec resolution, change later
    #if av < 1.:
    #	averaging = av_string

    #pickle_path = None
    #if save_pickle:
    #	pickle_path = path+folder_time+'-pickle_'+fit_type+'-'+fit_to+'-'+which_fit+'-l2-'+averaging+'-'+direction+'.p'

    #fit_var_path = None
    #if save_fit_variables:
    #	fit_var_path = path+folder_time+'-fit-result-variables_'+fit_type+'-'+fit_to+'-'+which_fit+'-l2-'+averaging+'-'+direction+'.csv'

    #fitrun_path = None
    #if save_fitrun:
    #	fitrun_path = path+folder_time+'-all-fit-variables_'+fit_type+'-'+fit_to+'-'+which_fit+'-l2-'+averaging+'-'+direction+'.csv'
        
    #	save.save_info_fit(fitrun_path, date_string, averaging, direction, data_product, dist, step, ept, het,
    #	sigma, rel_err, frac_nan_threshold, leave_out_1st_het_chan, shift_factor, fit_type, fit_to,
    #	which_fit, e_min, e_max, g1_guess, g2_guess, c1_guess, alpha_guess, break_guess_low, cut_guess,
    #	use_random, iterations)

    # <---------------------------------------------------------DATA--------------------------------------------------------->

    # ---- HELPERS -----
    def extract_energy(df):
        """Return energy and asymmetric errors."""
        return (df['Primary_energy'], [df['Energy_error_low'], df['Energy_error_high']])


    def extract_flux(df, flux_col, err_col):
        """Return flux and uncertainty."""
        return df[flux_col], df[err_col]


    # ----- ENERGY DATA -----
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

    # visual studio marks the following as not used but they are used if the specific version is uncommented do not delete in case of future use
    # Actually fixed aug2026
    # Contaminated data sigma
    spec_energy_c_sigma, energy_err_c_sigma = extract_energy(contaminated_data_sigma)

    # Contaminated data nan
    spec_energy_c_nan, energy_err_c_nan = extract_energy(contaminated_data_nan)

    # Contaminated data rel err
    spec_energy_c_rel_err, energy_err_c_rel_err = extract_energy(contaminated_data_rel_err)


    if het and leave_out_1st_het_chan:
        # First HET
        spec_energy_first_het, energy_err_first_het = extract_energy(first_het_data)


    # -------------------------------------------------------------------------
    # there is no physical difference between average uncertainty and peak uncertainty
    # so for both peak and avg fit use Backsub_peak_uncertainty
    # -------------------------------------------------------------------------

    # Intensity Selection
    if bg_subtraction:
        flux_col = f'Bg_subtracted_{fit_to}'
        err_col = 'Backsub_peak_uncertainty'
    else:
        flux_col = f'Flux_{fit_to}'
        err_col = 'Peak_electron_uncertainty'


    # ------ INTENSITY DATA -----
    spec_flux, flux_err = extract_flux(data, flux_col, err_col)

    if step and ept:
        spec_flux_step_ept, flux_err_step_ept = extract_flux(step_ept_data, flux_col, err_col)

    if ept and het:
        spec_flux_ept_het, flux_err_ept_het = extract_flux(ept_het_data, flux_col, err_col)

    if step:
        spec_flux_step, flux_err_step = extract_flux(step_data, flux_col, err_col)

    if ept:
        spec_flux_ept, flux_err_ept = extract_flux(ept_data, flux_col, err_col)

    if het:
        spec_flux_het, flux_err_het = extract_flux(het_data, flux_col, err_col)

    # Contaminated data 
    spec_flux_c, flux_err_c = extract_flux(contaminated_data, flux_col, err_col)

    # visual studio marks the following as not used but they are used if the specific version is uncommented do not delete in case of future use
    # Actually fixed aug26
    # Contaminated data sigma
    spec_flux_c_sigma, flux_err_c_sigma = extract_flux(contaminated_data_sigma, flux_col, err_col)

    # Contaminated data nan
    spec_flux_c_nan, flux_err_c_nan = extract_flux(contaminated_data_nan, flux_col, err_col)

    # Contaminated data rel err
    spec_flux_c_rel_err, flux_err_c_rel_err = extract_flux(contaminated_data_rel_err, flux_col, err_col)


    if het and leave_out_1st_het_chan:
        # First HET
        spec_flux_first_het, flux_err_first_het = extract_flux(first_het_data, flux_col, err_col)


    # --------- ENERGY RANGE SELECTION -------
    energy_map = {
        'step': spec_energy_step if step else None,
        'ept': spec_energy_ept if ept else None,
        'het': spec_energy_het if het else None,
        'step_ept': spec_energy_step_ept if (step and ept) else None,
        'ept_het': spec_energy_ept_het if (ept and het) else None,
        'step_ept_het': spec_energy}

    selected_energy = energy_map.get(fit_type)

    if selected_energy is None:
        raise ValueError(f"Invalid fit_type: {fit_type}")

    min_energy = min(selected_energy) if (e_min is None and selected_energy is not None) else e_min
    max_energy = max(selected_energy) if (e_max is None and selected_energy is not None) else e_max

#----------------------------------------------------------------------------------------------------------------------
    # Plotting colours
    color = {'sun': 'crimson', 'asun': 'orange',  'north': 'darkslateblue', 'south': 'c'}

    # quick change for sec resolution, change later
    # if av < 1.:
    #     averaging = av_string


    # --------- QUALITY FACTORS ----------
    qf_step = qf_ept = qf_het = None
    qf_step_av = qf_ept_av = qf_het_av = None

    if quality_factor is not None:
        # prevents silent misalignment bugs
        expected_qf = sum([step, ept, het])
        if len(quality_factor) < expected_qf:
            raise ValueError("Not enough quality_factor entries for selected instruments")

        qf_index = 0

        def get_qf(name):
            qf_vals = quality_factor[f"QF {name} all channels"]

            qf_avg_raw = quality_factor[f"QF {name} average"]

            # Handle both Series and scalar
            if hasattr(qf_avg_raw, "iloc"):
                qf_avg = qf_avg_raw.iloc[0]
            else:
                qf_avg = qf_avg_raw

            return qf_vals, qf_avg

        if step:
            qf_step, qf_step_av = get_qf("STEP")

        if ept:
            qf_ept, qf_ept_av = get_qf("EPT")

        if het:
            qf_het, qf_het_av = get_qf("HET")


    # ------- FILE PATHS -------
    def build_path(suffix):
        """Helper to standardize file naming."""
        return (f"{path}{folder_time}-{suffix}_{fit_type}-{fit_to}-{which_fit}-l2-{averaging}-{direction}{pix}")


    pickle_path = build_path("pickle") + ".p" if save_pickle else None
    fit_var_path = build_path("fit-result-variables") + ".csv" if save_fit_variables else None


    fitrun_path = None
    if save_fitrun:
        fitrun_path = build_path("all-fit-variables") + ".csv"

        save.save_info_fit(fitrun_path,
            date_string, averaging, direction, data_product, dist,
            step, ept, het,
            sigma, rel_err, frac_nan_threshold,
            leave_out_1st_het_chan, step_shift_factor,
            fit_type, fit_to, which_fit,
            min_energy, max_energy,
            g1_guess, g2_guess, c1_guess,
            alpha_guess, break_guess_low, cut_guess,
            use_random, iterations,
            qf_step_av, qf_ept_av, qf_het_av,
            centre_pix)

        # save quality factors separately
        qf_path = build_path("quality-factor") + ".csv"

        save.save_quality_factor(qf_path, qf_step, qf_ept, qf_het)
    
    # <------------------------------------------------------FIT AND PLOT---------------------------------------------------------->

    f, ax = plt.subplots(1, figsize=(8, 6), dpi=300)
    # plt.rcParams["font.family"] = "Times New Roman"
    #matplotlib.rc('font', family='Times New Roman')

    # distance  = ''
    distance = f' (R={dist} au)'

    # Legend info
    if legend_details:
        ax.plot([], [], ' ', label=f"Ion corr {'on' if ion_correction else 'off'}")
        ax.plot([], [], ' ', label=f"Bg subtraction {'on' if bg_subtraction else 'off'}")

        if shift_step_data:
            ax.plot([], [], ' ', label="Shift factor (STEP) " + str(np.round(step_shift_factor, 2)))


    # ------- FITTING -------
    if make_fit:
        fit_map = {
            'step': (spec_energy_step, spec_flux_step, energy_err_step, flux_err_step, 'STEP'),
            'ept': (spec_energy_ept, spec_flux_ept, energy_err_ept, flux_err_ept, 'EPT'),
            'het': (spec_energy_het, spec_flux_het, energy_err_het, flux_err_het, 'HET'),
            'step_ept': (spec_energy_step_ept, spec_flux_step_ept, energy_err_step_ept, flux_err_step_ept, 'STEP and EPT'),
            'ept_het': (spec_energy_ept_het, spec_flux_ept_het, energy_err_ept_het, flux_err_ept_het, 'EPT and HET'),
            'step_ept_het': (spec_energy, spec_flux, energy_err, flux_err, 'STEP, EPT and HET'),}

        if fit_type not in fit_map:
            raise ValueError(f"Unknown fit_type: {fit_type}")

        energy, flux, energy_err_local, flux_err_local, label = fit_map[fit_type]
        plot_title = f'Solar Orbiter {distance} {label}'

        fitting.MAKE_THE_FIT(energy, flux,energy_err_local[1],flux_err_local,ax, direction=direction, e_min=e_min, e_max=e_max,
        which_fit='single' if fit_type == 'het' else which_fit, g1_guess=g1_guess, g2_guess=g2_guess, g3_guess=g3_guess,
        alpha_guess=alpha_guess, beta_guess=beta_guess, break_low_guess=break_guess_low, break_high_guess=break_guess_high,
        cut_guess=cut_guess, c1_guess=c1_guess, exponent_guess=exponent_guess, use_random=use_random, iterations=iterations,
        path=pickle_path, path2=fit_var_path, detailed_legend=legend_details)

    # ------- PLOTTING DATA ------
    def plot_errorbar(x, y, yerr, xerr, **kwargs):
        if len(x) > 0:
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, marker='o', linestyle='', markersize=3, zorder=-1, **kwargs)


    if step:
        plot_errorbar(spec_energy_step, spec_flux_step, flux_err_step, energy_err_step, color='darkorange', label='STEP')

    if ept:
        plot_errorbar(spec_energy_ept, spec_flux_ept, flux_err_ept, energy_err_ept, color=color[direction], label='EPT ' + direction)

    if het:
        plot_errorbar(spec_energy_het, spec_flux_het, flux_err_het, energy_err_het, color='maroon', label='HET ' + direction)

        if leave_out_1st_het_chan:
            plot_errorbar(spec_energy_first_het, spec_flux_first_het, flux_err_first_het, energy_err_first_het,
                        color='black', label='First HET channel')

    if not do_not_plot_bad_channels:
        plot_errorbar(spec_energy_c, spec_flux_c, flux_err_c, energy_err_c,
                    color='gray',
                    label='Excluded from the fit' if make_fit else 'Non-significant channels')


    # Background 
    if step:
        plot_errorbar(spec_energy_step, step_data['Background_flux'],
                    step_data['Bg_electron_uncertainty'], energy_err_step,
                    color='darkorange', alpha=0.3)

    if ept:
        plot_errorbar(spec_energy_ept, ept_data['Background_flux'],
                    ept_data['Bg_electron_uncertainty'], energy_err_ept,
                    color=color[direction], alpha=0.3)

    if het:
        plot_errorbar(spec_energy_het, het_data['Background_flux'],
                    het_data['Bg_electron_uncertainty'], energy_err_het,
                    color='maroon', alpha=0.3)



  # ----- DETAILED PLOTTING ------

    plot_sigma_version = False
    plot_nan_version = False
    plot_rel_err_version = False

    if detailed_plot:
    # for a more detailed version
    # Contaminated / excluded data  
        if plot_sigma_version:
            ax.errorbar(spec_energy_c_sigma, spec_flux_c_sigma, yerr=flux_err_c_sigma, xerr=energy_err_c_sigma,
                        marker='o', linestyle='', markersize=3, color='blue', label='Sigma below ' + str(sigma),zorder=-1)

            ax.errorbar(spec_energy_c_sigma, contaminated_data_sigma['Background_flux'],
                        yerr=contaminated_data_sigma['Bg_electron_uncertainty'], xerr=energy_err_c_sigma,
                        marker='o', linestyle='', markersize=3, color='blue', alpha=0.3,zorder=-2)

        if plot_nan_version:
            ax.errorbar(spec_energy_c_nan, spec_flux_c_nan, yerr=flux_err_c_nan, xerr=energy_err_c_nan, 
                        marker='o', linestyle='', markersize=3, color='gray', label='excluded (NaNs)', zorder=-1)

            ax.errorbar(spec_energy_c_nan, contaminated_data_nan['Background_flux'], 
                        yerr=contaminated_data_nan['Bg_electron_uncertainty'], xerr=energy_err_c_nan,
                        marker='o', linestyle='', markersize=3, color='gray', alpha=0.3, zorder=-2)

        if plot_rel_err_version:
            ax.errorbar(spec_energy_c_rel_err, spec_flux_c_rel_err, yerr=flux_err_c_rel_err, xerr=energy_err_c_rel_err,
                        marker='o', linestyle='', markersize=3, color='purple', label='excluded (rel err)', zorder=-1)

            ax.errorbar(spec_energy_c_rel_err, contaminated_data_rel_err['Background_flux'],
                        yerr=contaminated_data_rel_err['Bg_electron_uncertainty'], xerr=energy_err_c_rel_err,
                        marker='o', linestyle='', markersize=3, color='purple', alpha=0.3, zorder=-2)

    # ------- AXIS -------
    step_energy_range = [0.004323343613, 0.07803193193]
    het_energy_range = [0.6859485403, 10.62300288]

    e_range_min = step_energy_range[0]
    e_range_max = het_energy_range[1]

    if do_not_plot_bad_channels:
        e_range_max = spec_energy[-1]

    ax.set_xscale('log')
    ax.set_yscale('log')

    locmin = pltt.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    ax.set_xlim(e_range_min - e_range_min / 2, e_range_max + e_range_max / 2)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(pltt.NullFormatter())

    ax.tick_params(which='major', width=1, length=4, color='black')
    ax.tick_params(which='minor', width=1, length=4, color='black')
        
    ax.tick_params(labelsize=fsize + 2)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1)


    # ------ LABELS + LEGEND -------
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)

    legend = None
    if not no_legend:
        if legend_outside:
            legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                                title=legend_title, prop={'size': fsize},
                                fontsize=fsize - 2, title_fontsize=fsize)
        else:
            legend = plt.legend(title=legend_title,
                                prop={'size': fsize - 2},
                                fontsize=fsize - 2,
                                title_fontsize=fsize)

    plt.ylabel(intensity_label, fontsize=fsize)
    plt.xlabel(energy_label, fontsize=fsize)


    # ---- TITLE -----
    if title_of_plot is None:
        extra = 'centre pixels' if centre_pix else ''
        plt.title(plot_title + '  ' + peak_info + '\n' +
                date_str + '  ' + str(averaging) + '  averaging ' + extra,
                fontsize=fsize + 2)
    else:
        plt.title(title_of_plot, fontsize=fsize + 2)


    # ----- SAVING ------
    plot_path = path
    if fit_to_separate_folder:
        plot_path = path + 'plots/'
        os.makedirs(plot_path, exist_ok=True)

    
    shift_string = ''
    if shift_step_data:
        shift_string = ('-auto-step-shift_' if auto_shift else '-step-shift_') + \
                    str(np.round(step_shift_factor, 2)).replace('.', '_')


    def savefig_safe(filename):
        if legend is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])
        else:
            plt.savefig(filename, dpi=300)

    if save_fig:
        base = f"{plot_path}electrons-{date_string}-{averaging}"

        if make_fit:
            suffix = f"-{direction}-{which_fit}-{fit_type}-{fit_to}"

            if ion_correction:
                suffix += "-ion_corr"
            if bg_subtraction:
                suffix += "-bg_sub"

            savefig_safe(base + suffix + pix + shift_string)

        else:
            combo = '_'.join([k for k, v in {'step': step, 'ept': ept, 'het': het}.items() if v])
            savefig_safe(base + f"-no_fit-{combo}" + pix + shift_string)


    plt.show()

