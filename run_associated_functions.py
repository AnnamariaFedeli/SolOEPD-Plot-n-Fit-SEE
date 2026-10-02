import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as pltt
from matplotlib import font_manager
font_manager.fontManager.ttflist
from matplotlib import rc
import matplotlib.ticker as ticker
#from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from sunpy.coordinates import get_horizons_coord
import make_the_fit_tripl as fitting
import savecsv as save
import combining_files as comb
import os
import shutil
from IPython.core.display import HTML

def make_html(fontname):
    return f"<p>{fontname}: <span style='font-family:{fontname}; font-size: 24px;'>{fontname}</p>"


if __name__ == "__main__":
    fonts = sorted({f.name for f in font_manager.fontManager.ttflist})
    code = "\n".join([make_html(font) for font in fonts])

    HTML(f"<div style='column-count: 2;'>{code}</div>")


def quality_factor_PA_coverage(data, coverage, direction = 'sun', angle = 180): 
    # TO DO: need to add min and max into the calculation and pixels for STEP
    qf = [] 

    for j in range(0, len(data[1])): 
        df = coverage.loc[:, direction] 
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

        if len(factors) > 0:
            qf.append(sum(factors) / len(factors))
        else:
            qf.append(np.nan)
        
    quality_factor = np.nanmean(qf) 
    return [qf, quality_factor]


def compute_quality_factors(plot_pa, step, ept, het, data_step = None, data_ept = None, data_het = None, 
                            coverage_step = None, coverage_ept = None, coverage_het = None, direction = 'sun', 
                            angle = 0, data_step_pix = None, pixels = False):
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
        process("STEP", data_step, coverage_step) # using ept because need to implement step calculation

        if pixels:
            qf_vals, qf_avg = quality_factor_PA_coverage(data_step_pix, coverage_step, direction=direction, angle=angle)
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


def calculate_shift_factor(step_data, ept_data, sigma, rel_err, frac_nan_threshold, fit_to, species):
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
    E_MIN = None
    E_MAX = None
    
    species = species.lower()
    
    if species in ['electron', 'electrons', 'e']:
        E_MIN = 0.037
        E_MAX = 0.057
        
    elif species in ['proton', 'protons', 'p']:
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
    
