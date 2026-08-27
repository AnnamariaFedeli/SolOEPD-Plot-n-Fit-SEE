import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from pytimeparse.timeparse import timeparse

def save_info_plot(
    path,
    plot_start,
    plot_end,
    t_inj,
    bgstart,
    bgend,
    bg_distance_from_window,
    bg_period,
    travel_distance,
    travel_distance_second_slope,
    fixed_window,
    data_type,
    averaging_mode,
    averaging,
    masking,
    ion_conta_corr,
    dist,
    spiral_len,
    traveltime_min,
    traveltime_max,
    light_tt,
):
    """Save plot configuration and derived information to a CSV file.

    Args:
        path: Output path for the CSV file.
        plot_start: Start time of the plotting window.
        plot_end: End time of the plotting window.
        t_inj: Injection time at the Sun.
        bgstart: Start of the background interval.
        bgend: End of the background interval.
        bg_distance_from_window: Distance of the background interval from
            the plotting window, in minutes.
        bg_period: Duration of the background interval, in minutes.
        travel_distance: Travel distance for the first slope, in AU.
        travel_distance_second_slope: Travel distance for the second slope,
            in AU.
        fixed_window: Fixed window duration, in minutes.
        data_type: Type of data being processed.
        averaging_mode: Averaging mode.
        averaging: Averaging interval.
        masking: Masking configuration.
        ion_conta_corr: Ion contamination correction setting.
        dist: Spacecraft distance, in AU.
        spiral_len: Parker spiral length, in AU.
        traveltime_min: Travel time for 4 keV particles, in minutes.
        traveltime_max: Travel time for 10 MeV particles, in minutes.
        light_tt: Light travel time at distance ``dist``, in minutes.

    Returns:
        The DataFrame containing the saved information.
    """
    df = pd.DataFrame({
            "Plot Start": plot_start,
            "Plot end": plot_end,
            "Injection time at Sun": t_inj,
            "Background start": bgstart,
            "Background end": bgend,
            "Bg distance from window [min]": bg_distance_from_window,
            "Bg period [min]": bg_period,
            "Travel distance first slope [AU]": travel_distance,
            "Travel distance second slope [AU]": travel_distance_second_slope,
            "Fixed window [min]": fixed_window,
            "data type": data_type,
            "Averaging mode": averaging_mode,
            "Averaging [s]": (
                timeparse(averaging) if averaging is not None else "no_averaging"
            ),
            "Masking": masking,
            "Ion contamination corection": ion_conta_corr,
            "Distance of s/c [AU]": dist,
            "Length of Parker Spiral [AU]": spiral_len,
            "Traveltime 4keV [min]": traveltime_min,
            "Traveltime 10MeV [min]": traveltime_max,
            "Traveltime of light at distance D [min]": light_tt,
            },index=[0],)

    df.to_csv(path, sep=";")

    return df     

def save_info_fit(
    path,
    date_string,
    averaging,
    direction,
    data_product,
    dist,
    step,
    ept,
    het,
    sigma,
    rel_err,
    frac_nan_threshold,
    leave_out_1st_het_chan,
    shift_factor,
    fit_type,
    fit_to,
    which_fit,
    e_min,
    e_max,
    g1_guess,
    g2_guess,
    c1_guess,
    alpha_guess,
    break_guess,
    cut_guess,
    use_random,
    iterations,
    quality_factor_step,
    quality_factor_ept,
    quality_factor_het,
    centre_pixels,
):
    """Save fitting configuration and parameters to a CSV file.

    Args:
        path: Output path for the CSV file.
        date_string: Date associated with the fit.
        averaging: Averaging interval.
        direction: Data direction.
        data_product: Data product/type used for the fit.
        dist: Spacecraft distance, in AU.
        step: Whether STEP data are used.
        ept: Whether EPT data are used.
        het: Whether HET data are used.
        sigma: Sigma value used in the analysis.
        rel_err: Relative error used in the analysis.
        frac_nan_threshold: Maximum allowed fraction of NaN values.
        leave_out_1st_het_chan: Whether the first HET channel is excluded.
        shift_factor: Shift applied to STEP data.
        fit_type: Type of fit.
        fit_to: Data/model component being fitted.
        which_fit: Specific fit selection.
        e_min: Minimum energy used for the fit.
        e_max: Maximum energy used for the fit.
        g1_guess: Initial guess for gamma 1.
        g2_guess: Initial guess for gamma 2.
        c1_guess: Initial guess for c1.
        alpha_guess: Initial guess for alpha.
        break_guess: Initial guess for the break energy, in MeV.
        cut_guess: Initial guess for the cutoff point, in MeV.
        use_random: Whether random initial values are used.
        iterations: Number of fitting iterations.
        quality_factor_step: Quality factor for STEP.
        quality_factor_ept: Quality factor for EPT.
        quality_factor_het: Quality factor for HET.
        centre_pixels: Centre pixel configuration.

    Returns:
        The DataFrame containing the saved fitting information.
    """
    df = pd.DataFrame({
            "Date": date_string,
            "Averaging [s]": (timeparse(averaging) if averaging is not None else "no_averaging"),
            "Direction": direction,
            "Data type": data_product,
            "Distance [AU]": dist,
            "STEP": step,
            "EPT": ept,
            "HET": het,
            "Sigma": sigma,
            "Relative error": rel_err,
            "Fraction of nan": frac_nan_threshold,
            "Leave first HET channel out": leave_out_1st_het_chan,
            "Shift STEP data": shift_factor,
            "Type of fit": fit_type,
            "Fit to": fit_to,
            "Which fit": which_fit,
            "Min energy": e_min,
            "Max energy": e_max,
            "Gamma1 guess": g1_guess,
            "Gamma2 guess": g2_guess,
            "c1 guess": c1_guess,
            "Alpha guess": alpha_guess,
            "Break guess [MeV]": break_guess,
            "Cutoff point guess [MeV]": cut_guess,
            "Use random": use_random,
            "Iterations": iterations,
            "Quality factor average STEP": quality_factor_step,
            "Quality factor average EPT": quality_factor_ept,
            "Quality factor average HET": quality_factor_het,
            "Centre pixels": centre_pixels,
        },index=[0],)

    df.to_csv(path, sep=";")

    return df

def save_quality_factor(path, qf_step, qf_ept, qf_het):
    """Save quality factor information to a CSV file.

    Args:
        path: Output path for the CSV file.
        qf_step: Quality factor information for STEP, or ``None``.
        qf_ept: Quality factor information for EPT, or ``None``.
        qf_het: Quality factor information for HET, or ``None``.

    Returns:
        The DataFrame containing the quality factor information.
    """
    qf_step_av = None
    qf_step_all = None
    qf_ept_av = None
    qf_ept_all = None
    qf_het_av = None
    qf_het_all = None

    if qf_step is not None:
        qf_step_all = qf_step[0]
        qf_step_av = qf_step[1]

    if qf_ept is not None:
        qf_ept_all = qf_ept[0]
        qf_ept_av = qf_ept[1]

    if qf_het is not None:
        qf_het_all = qf_het[0]
        qf_het_av = qf_het[1]

    quality_factors = {
        "QF STEP average": qf_step_av,
        "QF STEP all channels": qf_step_all,
        "QF EPT average": qf_ept_av,
        "QF EPT all channels": qf_ept_all,
        "QF HET average": qf_het_av,
        "QF HET all channels": qf_het_all,
    }

    df = pd.DataFrame({key: pd.Series(value) for key, value in quality_factors.items()})

    df.to_csv(path, sep=";")

    return df