# from solo_epd_loader import epd_load
from matplotlib.ticker import LinearLocator, MultipleLocator, AutoMinorLocator
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from adjustText import adjust_text
from astropy import units as u
import sys
import site
import matplotlib.pyplot as plt
from math import *
from tkinter import *
import astropy.units as u
import numpy as np
import pandas as pd
from solo_epd_loader import epd_load
from sunpy.coordinates import get_horizons_coord
from savecsv import *
from tabulate import tabulate
from seppy.loader.solo import mag_load
from pandas.tseries.frequencies import to_offset
from tqdm.auto import tqdm
import os
import re
import seaborn as sns
from epd_plot_shift import *

def extract_particle_data(
    df_electrons_or_protons,
    df_energies,
    plotstart,
    plotend,
    t_inj,
    species='electron',
    bgstart=None,
    bgend=None,
    bg_distance_from_window='2h',
    bg_period='60min',
    travel_distance=0,
    travel_distance_second_slope=None,
    fixed_window=None,
    instrument='ept',
    data_type='l2',
    averaging=None,
    masking=True,
    ion_conta_corr=False,
    df_protons=None,
    centre_pix=False
):
    """
    Extract electron or proton fluxes and determine energy-dependent
    peak and average information.

    Parameters
    ----------
    df_electrons_or_protons : pandas.DataFrame
        Electron or proton data, depending on ``species``.

    df_energies : pandas.DataFrame
        DataFrame containing the energy-channel information for the
        selected instrument and data product.

    plotstart : str
        Start time of the time interval to analyze.

    plotend : str
        End time of the time interval to analyze.

    t_inj : str
        Solar particle injection time.

    species : str, optional
        Particle species. Accepted values are ``'electron'``,
        ``'electrons'``, ``'e'``, ``'proton'``, ``'protons'``,
        or ``'p'``. Defaults to ``'electron'``.

    bgstart : str, optional
        Start time of a fixed background interval.

    bgend : str, optional
        End time of a fixed background interval.

    bg_distance_from_window : str, optional
        Time between the end of the background interval and the start
        of the energy-dependent search window.

    bg_period : str, optional
        Duration of the moving background interval.

    travel_distance : float, optional
        Travel distance in AU used to determine the start of the
        energy-dependent search window.

    travel_distance_second_slope : float, optional
        Travel distance in AU used to determine the end of the
        energy-dependent search window.

    fixed_window : str, optional
        Fixed duration of the search window. If specified,
        ``travel_distance_second_slope`` is not used.

    instrument : str, optional
        Instrument name: ``'ept'``, ``'het'``, or ``'step'``.

    data_type : str, optional
        Data product type, e.g. ``'ll'`` or ``'l2'``.

    averaging : str, optional
        Pandas resampling interval. If None, no averaging is performed.

    masking : bool, optional
        STEP electron ion-contamination masking.

    ion_conta_corr : bool, optional
        EPT electron ion-contamination correction.

    df_protons : pandas.DataFrame, optional
        Proton data required for EPT ion-contamination correction.

    centre_pix : bool, optional
        Use STEP centre-pixel electron data.

    Returns
    -------
    tuple
        Particle fluxes, information dataframe, search periods,
        energy ranges, and instrument information.
    """

    # ------------------------------------------------------------------
    # Normalise and validate species.
    # ------------------------------------------------------------------

    species = species.lower()

    if species in ['electron', 'electrons', 'e']:
        species = 'electron'

    elif species in ['proton', 'protons', 'p']:
        species = 'proton'

    else:
        raise ValueError(
            "species must be 'electron' or 'proton'."
        )

    particle_name = species.capitalize()

    # Particle mass used by evolt2speed().
    particle_mass = {
        'electron': 2,
        'proton': 1
    }[species]

    # ------------------------------------------------------------------
    # Validate background definition.
    # ------------------------------------------------------------------

    if bgstart is not None or bgend is not None:
        if (
            bg_distance_from_window is not None
            or bg_period is not None
        ):
            raise Exception(
                "Please specify either bg_start and bg_end or "
                "bg_distance_from_window and bg_period."
            )

    if bgstart is None or bgend is None:
        if (
            bg_distance_from_window is None
            or bg_period is None
        ):
            raise Exception(
                "Please specify either bg_start and bg_end or "
                "bg_distance_from_window and bg_period."
            )

    # ------------------------------------------------------------------
    # Extract particle fluxes and uncertainties.
    # ------------------------------------------------------------------

    instrument = instrument.lower()

    if instrument != 'step':
        df_particle_fluxes = (
            df_electrons_or_protons[
                f'{particle_name}_Flux'
            ][plotstart:plotend]
        )

        df_particle_uncertainties = (
            df_electrons_or_protons[
                f'{particle_name}_Uncertainty'
            ][plotstart:plotend]
        )

    # EPT proton data are needed for electron ion-contamination
    # correction.
    if instrument == 'ept' and species == 'electron':

        if ion_conta_corr:
            if df_protons is None:
                raise ValueError(
                    "df_protons must be provided when "
                    "ion_conta_corr=True."
                )

            df_proton_fluxes = (
                df_protons['Proton_Flux'][plotstart:plotend]
            )

            df_proton_uncertainties = (
                df_protons['Proton_Uncertainty'][plotstart:plotend]
            )

    # ------------------------------------------------------------------
    # Determine energy bins and standardise flux column names.
    # ------------------------------------------------------------------

    if instrument in ['ept', 'het']:

        if data_type == 'll':

            channels = range(
                len(df_energies['Electron_Bins_Low_Energy'])
                if species == 'electron'
                else len(df_energies['Proton_Bins_Low_Energy'])
            )

            if species == 'electron':
                e_low = df_energies[
                    'Electron_Bins_Low_Energy'
                ]
            else:
                e_low = df_energies[
                    'Proton_Bins_Low_Energy'
                ]

            e_high = []

            for i in channels:

                if species == 'electron':
                    e_high.append(
                        e_low[i]
                        + df_energies['Electron_Bins_Width'][i]
                    )

                    df_particle_fluxes = (
                        df_particle_fluxes.rename(
                            columns={
                                f'Ele_Flux_{i}':
                                f'Electron_Flux_{i}'
                            }
                        )
                    )

                    df_particle_uncertainties = (
                        df_particle_uncertainties.rename(
                            columns={
                                f'Ele_Flux_Sigma_{i}':
                                f'Electron_Uncertainty_{i}'
                            }
                        )
                    )

                else:
                    e_high.append(
                        e_low[i]
                        + df_energies['Ion_Bins_Width'][i]
                    )

                    df_particle_fluxes = (
                        df_particle_fluxes.rename(
                            columns={
                                f'H_Flux_{i}':
                                f'Proton_Flux_{i}'
                            }
                        )
                    )

                    df_particle_uncertainties = (
                        df_particle_uncertainties.rename(
                            columns={
                                f'H_Flux_Sigma_{i}':
                                f'Proton_Uncertainty_{i}'
                            }
                        )
                    )

        elif data_type == 'l2':

            if species == 'electron':

                e_low = df_energies[
                    'Electron_Bins_Low_Energy'
                ]

            else:

                if instrument == 'ept':
                    e_low = df_energies[
                        'Ion_Bins_Low_Energy'
                    ]
                else:
                    e_low = df_energies[
                        'H_Bins_Low_Energy'
                    ]

            e_high = []

            channels = range(len(e_low))

            for i in channels:

                if instrument == 'ept':

                    width = df_energies[
                        'Electron_Bins_Width'
                        if species == 'electron'
                        else 'Ion_Bins_Width'
                    ][i]

                else:

                    width = df_energies[
                        'Electron_Bins_Width'
                        if species == 'electron'
                        else 'H_Bins_Width'
                    ][i]

                e_high.append(
                    e_low[i] + width
                )

                if species == 'proton':

                    df_particle_fluxes = (
                        df_particle_fluxes.rename(
                            columns={
                                f'H_Flux_{i}':
                                f'Proton_Flux_{i}'
                            }
                        )
                    )

                    df_particle_uncertainties = (
                        df_particle_uncertainties.rename(
                            columns={
                                f'H_Uncertainty_{i}':
                                f'Proton_Uncertainty_{i}'
                            }
                        )
                    )

    # ------------------------------------------------------------------
    # STEP.
    # ------------------------------------------------------------------

    elif instrument == 'step':

        if species == 'electron':

            old_new_data_string = ''

            if (
                'Electron_Sectors_Bins_Text'
                in df_energies.keys()
                and centre_pix
            ):
                old_new_data_string = 'Electron_Sectors_'

            elif 'Electron_Bins_Text' in df_energies.keys():
                old_new_data_string = 'Electron_'

            else:
                raise ValueError(
                    'This is before the data change of October 2021 '
                    'and you are not using center pixels. There is no '
                    'Electron keyword.'
                )

            if data_type == 'l2':

                e_low = df_energies[
                    old_new_data_string
                    + 'Bins_Low_Energy'
                ]

                e_high = []

                channels = range(len(e_low))

                df_particle_fluxes = pd.DataFrame()
                df_particle_uncertainties = pd.DataFrame()

                for i in channels:

                    e_high.append(
                        e_low[i]
                        + df_energies[
                            old_new_data_string
                            + 'Bins_Width'
                        ][i]
                    )

                    if centre_pix:

                        df_particle_fluxes[
                            f'Electron_Flux_{i}'
                        ] = (
                            df_electrons_or_protons[
                                f'Electron_Comb_Flux_{i}'
                            ][plotstart:plotend]
                        )

                        df_particle_uncertainties[
                            f'Electron_Uncertainty_{i}'
                        ] = (
                            df_electrons_or_protons[
                                f'Electron_Comb_Uncertainty_{i}'
                            ][plotstart:plotend]
                        )

                    else:

                        df_particle_fluxes[
                            f'Electron_Flux_{i}'
                        ] = (
                            df_electrons_or_protons[
                                f'Electron_Avg_Flux_{i}'
                            ][plotstart:plotend]
                        )

                        df_particle_uncertainties[
                            f'Electron_Uncertainty_{i}'
                        ] = (
                            df_electrons_or_protons[
                                f'Electron_Avg_Uncertainty_{i}'
                            ][plotstart:plotend]
                        )

        else:

            if data_type == 'l2':

                e_low = df_energies[
                    'Bins_Low_Energy'
                ]

                e_high = []

                channels = range(len(e_low))

                df_particle_fluxes = pd.DataFrame()
                df_particle_uncertainties = pd.DataFrame()

                for i in channels:

                    e_high.append(
                        e_low[i]
                        + df_energies['Bins_Width'][i]
                    )

                    df_particle_fluxes[
                        f'Proton_Flux_{i}'
                    ] = (
                        df_electrons_or_protons[
                            f'Magnet_Avg_Flux_{i}'
                        ][plotstart:plotend]
                    )

                    df_particle_uncertainties[
                        f'Proton_Uncertainty_{i}'
                    ] = (
                        df_electrons_or_protons[
                            f'Magnet_Avg_Uncertainty_{i}'
                        ][plotstart:plotend]
                    )

        # Remove negative STEP fluxes.
        df_particle_fluxes[
            df_particle_fluxes < 0
        ] = np.nan

    # ------------------------------------------------------------------
    # Average the data if requested.
    # ------------------------------------------------------------------

    if averaging is not None:

        if instrument != 'step':

            df_particle_fluxes = (
                df_particle_fluxes
                .resample(averaging)
                .mean()
            )

            df_particle_uncertainties = (
                df_particle_uncertainties
                .resample(averaging)
                .apply(average_flux_error)
            )

        # For STEP, resampling is done independently.
        # The STEP dataframe is already constructed channel by channel.

    # ------------------------------------------------------------------
    # EPT electron ion-contamination correction.
    # ------------------------------------------------------------------

    if (
        species == 'electron'
        and instrument == 'ept'
        and ion_conta_corr
    ):

        ion_cont_corr_matrix = np.loadtxt(
            'EPT_ion_contamination_flux_paco.dat'
        )

        electron_flux_cont = np.zeros(
            np.shape(df_particle_fluxes)
        )

        electron_uncertainty_cont = np.zeros(
            np.shape(df_particle_uncertainties)
        )

        for tt in range(len(df_particle_fluxes)):

            electron_flux_cont[tt, :] = np.sum(
                ion_cont_corr_matrix
                * np.ma.masked_invalid(
                    df_proton_fluxes.values[tt, :]
                ),
                axis=1
            )

            electron_uncertainty_cont[tt, :] = np.sqrt(
                np.sum(
                    ion_cont_corr_matrix**2
                    * np.ma.masked_invalid(
                        df_proton_uncertainties
                        .values[tt, :]**2
                    ),
                    axis=1
                )
            )

        df_particle_fluxes = (
            df_particle_fluxes - electron_flux_cont
        )

        df_particle_uncertainties = np.sqrt(
            df_particle_uncertainties**2
            + electron_uncertainty_cont**2
        )

    # ------------------------------------------------------------------
    # Main information dataframe.
    # ------------------------------------------------------------------

    df_info = pd.DataFrame(
        {
            'Plot_period': [],
            'Averaging': [],
            'Energy_channel': [],
            'Primary_energy': []
        }
    )

    if instrument == 'ept':
        df_info['Ion_contamination_correction'] = (
            [ion_conta_corr]
            + [''] * (len(channels) - 1)
        )

    elif instrument == 'step':
        df_info['Ion_masking'] = (
            [masking]
            + [''] * (len(channels) - 1)
        )

    df_info['Plot_period'] = (
        [plotstart]
        + [plotend]
        + [''] * (len(channels) - 2)
    )

    if averaging is None:

        df_info['Averaging'] = (
            ['No averaging']
            + [''] * (len(channels) - 1)
        )

    else:

        df_info['Averaging'] = (
            ['Mean', 'Resampled to ' + averaging]
            + [''] * (len(channels) - 2)
        )

    # ------------------------------------------------------------------
    # Primary energies and energy uncertainties.
    # ------------------------------------------------------------------

    primary_energies = [
        np.sqrt(e_low[i] * e_high[i])
        for i in range(len(e_low))
    ]

    df_info['Primary_energy'] = [
        primary_energies[i]
        for i in channels
    ]

    energy_error_low = [
        primary_energies[i] - e_low[i]
        for i in range(len(primary_energies))
    ]

    energy_error_high = [
        e_high[i] - primary_energies[i]
        for i in range(len(primary_energies))
    ]

    df_info['Energy_error_low'] = [
        energy_error_low[i]
        for i in channels
    ]

    df_info['Energy_error_high'] = [
        energy_error_high[i]
        for i in channels
    ]

    # ------------------------------------------------------------------
    # Particle velocity and search windows.
    # ------------------------------------------------------------------

    velocity = [
        evolt2speed(energy, particle_mass)
        for energy in primary_energies
    ]

    travel_distance_km = (
        travel_distance * 1.496E8
    )

    DV = [
        travel_distance_km / v
        for v in velocity
    ]

    searchstart = [
        pd.to_datetime(t_inj)
        + pd.Timedelta(seconds=dv)
        for dv in DV
    ]

    searchend = []

    if fixed_window is None:

        travel_distance_second_slope_km = (
            travel_distance_second_slope * 1.496E8
        )

        DV2 = [
            travel_distance_second_slope_km / v
            for v in velocity
        ]

        searchend = [
            pd.to_datetime(t_inj)
            + pd.Timedelta(seconds=dv)
            for dv in DV2
        ]

    else:

        searchend = [
            start + pd.to_timedelta(fixed_window)
            for start in searchstart
        ]

    # ------------------------------------------------------------------
    # Background windows.
    # ------------------------------------------------------------------

    if bg_distance_from_window is None:

        bg_start = bgstart
        bg_end = bgend

        bgstart = [bg_start] * len(searchstart)
        bgend = [bg_end] * len(searchstart)

    else:

        bgstart = []
        bgend = []

        for start in searchstart:

            end = (
                start
                - pd.to_timedelta(
                    bg_distance_from_window
                )
            )

            bgend.append(end)

            bgstart.append(
                end - pd.to_timedelta(bg_period)
            )

    # ------------------------------------------------------------------
    # Calculate channel-dependent information.
    # ------------------------------------------------------------------

    list_bg_fluxes = []
    list_flux_peaks = []
    list_peak_timestamps = []
    list_bg_subtracted_peaks = []
    list_peak_uncertainties = []
    list_average_bg_uncertainties = []
    list_bg_std = []
    list_peak_significance = []
    list_flux_average = []
    list_bg_subtracted_average = []
    list_average_significance = []
    list_frac_nonan = []

    for n, channel in enumerate(channels):

        flux_column = (
            f'{particle_name}_Flux_{channel}'
        )

        uncertainty_column = (
            f'{particle_name}_Uncertainty_{channel}'
        )

        particle_flux = (
            df_particle_fluxes[flux_column]
        )

        particle_uncertainty = (
            df_particle_uncertainties[
                uncertainty_column
            ]
        )

        # --------------------------------------------------------------
        # Peak/search window.
        # --------------------------------------------------------------

        f_p = particle_flux[
            searchstart[n]:searchend[n]
        ]

        if len(f_p) == 0:
            flux_peak = np.nan
            peak_timestamp = np.nan
            frac_nonan = np.nan

        else:

            if f_p.notna().any():
                flux_peak = f_p.max()
                peak_timestamp = f_p.idxmax()
            else:
                flux_peak = np.nan
                peak_timestamp = np.nan

            frac_nonan = f_p.notna().mean()

        list_flux_peaks.append(flux_peak)
        list_peak_timestamps.append(peak_timestamp)
        list_frac_nonan.append(frac_nonan)

        # --------------------------------------------------------------
        # Background flux.
        # --------------------------------------------------------------

        if len(f_p) == 0:

            bg_flux = np.nan

        else:

            bg_flux = particle_flux[
                bgstart[n]:bgend[n]
            ].mean(skipna=True)

        list_bg_fluxes.append(bg_flux)

        # --------------------------------------------------------------
        # Uncertainty at peak.
        # --------------------------------------------------------------

        if (
            pd.isna(peak_timestamp)
            or len(particle_uncertainty) == 0
        ):

            peak_uncertainty = np.nan

        else:

            timestamp_loc = (
                particle_uncertainty.index.get_indexer(
                    [peak_timestamp],
                    method='nearest'
                )[0]
            )

            peak_uncertainty = (
                particle_uncertainty.iloc[
                    timestamp_loc
                ]
            )

        list_peak_uncertainties.append(
            peak_uncertainty
        )

        # --------------------------------------------------------------
        # Average background uncertainty.
        # --------------------------------------------------------------

        bg_uncertainty = particle_uncertainty[
            bgstart[n]:bgend[n]
        ]

        valid_uncertainties = (
            bg_uncertainty.dropna()
        )

        if len(valid_uncertainties) == 0:

            average_bg_uncertainty = np.nan

        else:

            average_bg_uncertainty = (
                np.sqrt(
                    (valid_uncertainties**2).sum()
                )
                / len(valid_uncertainties)
            )

        list_average_bg_uncertainties.append(
            average_bg_uncertainty
        )

        # --------------------------------------------------------------
        # Background standard deviation.
        # --------------------------------------------------------------

        bg_std = particle_flux[
            bgstart[n]:bgend[n]
        ].std()

        list_bg_std.append(bg_std)

        # --------------------------------------------------------------
        # Average flux in search window.
        # --------------------------------------------------------------

        f_a = particle_flux[
            searchstart[n]:searchend[n]
        ]

        if len(f_a) == 0:

            flux_average = np.nan

        else:

            flux_average = f_a.mean(skipna=True)

        list_flux_average.append(flux_average)

    # ------------------------------------------------------------------
    # Background-subtracted values and significances.
    # ------------------------------------------------------------------

    for i in range(len(list_flux_peaks)):

        bg_subtracted_peak = (
            list_flux_peaks[i]
            - list_bg_fluxes[i]
        )

        list_bg_subtracted_peaks.append(
            bg_subtracted_peak
        )

        list_peak_significance.append(
            bg_subtracted_peak
            / list_bg_std[i]
        )

        if bg_subtracted_peak < list_bg_fluxes[i]:
            list_peak_significance[i] = -1

        bg_subtracted_average = (
            list_flux_average[i]
            - list_bg_fluxes[i]
        )

        list_bg_subtracted_average.append(
            bg_subtracted_average
        )

        list_average_significance.append(
            bg_subtracted_average
            / list_bg_std[i]
        )

        if (
            bg_subtracted_average
            < list_bg_fluxes[i]
        ):
            list_average_significance[i] = -1

    # ------------------------------------------------------------------
    # Populate df_info.
    # ------------------------------------------------------------------

    df_info['Energy_channel'] = channels
    df_info['Bg_start'] = bgstart
    df_info['Bg_end'] = bgend
    df_info['Searchstart'] = searchstart
    df_info['Searchend'] = searchend
    df_info['Peak_timestamp'] = list_peak_timestamps

    df_info['Background_flux'] = list_bg_fluxes
    df_info['Flux_peak'] = list_flux_peaks
    df_info['Bg_subtracted_peak'] = (
        list_bg_subtracted_peaks
    )

    df_info[
        f'Peak_{species}_uncertainty'
    ] = list_peak_uncertainties

    df_info[
        f'Bg_{species}_uncertainty'
    ] = list_average_bg_uncertainties

    df_info['Peak_significance'] = (
        list_peak_significance
    )

    df_info['Flux_average'] = list_flux_average

    df_info['Bg_subtracted_average'] = (
        list_bg_subtracted_average
    )

    df_info['Average_significance'] = (
        list_average_significance
    )

    df_info['Backsub_peak_uncertainty'] = np.sqrt(
        df_info[
            f'Peak_{species}_uncertainty'
        ] ** 2
        +
        df_info[
            f'Bg_{species}_uncertainty'
        ] ** 2
    )

    df_info['rel_backsub_peak_err'] = np.abs(
        df_info['Backsub_peak_uncertainty']
        /
        df_info['Bg_subtracted_peak']
    )

    df_info['frac_nonan'] = list_frac_nonan

    # ------------------------------------------------------------------
    # Return.
    # ------------------------------------------------------------------

    return (
        df_particle_fluxes,
        df_info,
        [searchstart, searchend],
        [e_low, e_high],
        [instrument, data_type]
    )


def extract_proton_data(df_protons, df_energies, plotstart, plotend, t_inj, bgstart=None, bgend=None, bg_distance_from_window='120min', bg_period='60min',
    travel_distance=0, travel_distance_second_slope=None, fixed_window=None, instrument='ept', data_type='l2', averaging=None):
    """
    Extract proton fluxes and determine energy-dependent peak information.

    The function determines an energy spectrum from proton time-series data
    for the Solar Orbiter / EPD instruments. Energy-dependent search windows
    are used to determine the flux values for each energy channel. The
    search-window start time is determined from the expected velocity
    dispersion based on the solar injection time (`t_inj`) and the specified
    travel distance.

    A background can either be defined using a fixed time interval or using
    a moving interval whose position is determined relative to the
    energy-dependent search window.

    Parameters
    ----------
    df_protons : pandas.DataFrame
        Proton data containing the fluxes and uncertainties.

    df_energies : pandas.DataFrame
        DataFrame containing the energy-channel information for the selected instrument and data product.

    plotstart : str
        Start time of the time interval to analyze.

    plotend : str
        End time of the time interval to analyze.

    t_inj : str
        Solar particle injection time.

    bgstart : str, optional
        Start time of the background window. If specified, `bgend` must also
        be specified. Leave as None when using a moving background window.

    bgend : str, optional
        End time of the background window. If specified, `bgstart` must also
        be specified. Leave as None when using a moving background window.

    bg_distance_from_window : str, optional
        Distance between the end of the background window and the start of
        the energy-dependent search window. Must be specified together with
        `bg_period` when using a moving background window.

    bg_period : str, optional
        Duration of the background window. Must be specified together with
        `bg_distance_from_window` when using a moving background window.

    travel_distance : float, optional
        Travel distance in AU used to calculate the energy-dependent start
        time of the search window. Defaults to 0.

    travel_distance_second_slope : float, optional
        Travel distance in AU used to calculate the end time of the
        energy-dependent search window. If None, `fixed_window` must be
        specified.

    fixed_window : str, optional
        Length of the search window. If specified,
        `travel_distance_second_slope` is not used.

    instrument : str, optional
        Instrument name: 'ept', 'het', or 'step'. Defaults to 'ept'.

    data_type : str, optional
        Data level, e.g. 'll' or 'l2'. Defaults to 'l2'.

    averaging : str, optional
        Pandas resampling interval. If None, no averaging is performed.

    Raises
    ------
    Exception
        If both a fixed background window and a moving background window
        are specified, or if neither is specified completely.

    Returns
    -------
    tuple
        df_proton_fluxes : pandas.DataFrame
            Proton fluxes for the selected instrument and energy channels.

        df_info : pandas.DataFrame
            DataFrame containing the extracted spectrum and associated
            metadata.

        [searchstart, searchend] : list
            Energy-dependent search-window start and end times.

        [e_low, e_high] : list
            Lower and upper energy boundaries for each energy channel.

        [instrument, data_type] : list
            Instrument and data type.
    """

    if bgstart is not None or bgend is not None:
        if bg_distance_from_window is not None or bg_period is not None:
            raise Exception(
                "Please specify either bg_start and bg_end or bg_distance_from_window and bg_period."
            )

    if bgstart is None or bgend is None:
        if bg_distance_from_window is None or bg_period is None:
            raise Exception(
                "Please specify either bg_start and bg_end or bg_distance_from_window and bg_period."
            )

    # Take proton flux and uncertainty values from the original data.
    if instrument == 'ept':
        df_proton_fluxes = df_protons['Ion_Flux'][plotstart:plotend]
        df_proton_uncertainties = (
            df_protons['Ion_Uncertainty'][plotstart:plotend]
        )

        if data_type == 'll':
            channels = range(len(df_energies['Ion_Bins_Low_Energy']))
            e_low = df_energies['Ion_Bins_Low_Energy']
            e_high = []

            for i in channels:
                e_high.append(
                    e_low[i] + df_energies['Ion_Bins_Width'][i]
                )

                df_proton_fluxes = df_proton_fluxes.rename(
                    columns={
                        f'H_Flux_{i}': f'Ion_Flux_{i}'
                    }
                )

                df_proton_uncertainties = df_proton_uncertainties.rename(
                    columns={
                        f'H_Flux_Sigma_{i}': f'Ion_Uncertainty_{i}'
                    }
                )

        elif data_type == 'l2':
            channels = range(len(df_energies['Ion_Bins_Low_Energy']))
            e_low = df_energies['Ion_Bins_Low_Energy']
            e_high = []

            for i in channels:
                e_high.append(
                    e_low[i] + df_energies['Ion_Bins_Width'][i]
                )

    elif instrument == 'het':
        df_proton_fluxes = df_protons['H_Flux'][plotstart:plotend]
        df_proton_uncertainties = (
            df_protons['H_Uncertainty'][plotstart:plotend]
        )

        if data_type == 'll':
            e_low = df_energies['Ion_Bins_Low_Energy']
            e_high = []
            channels = range(len(df_energies['Ion_Bins_Low_Energy']))

            for i in channels:
                e_high.append(
                    e_low[i] + df_energies['Ion_Bins_Width'][i]
                )

                df_proton_fluxes = df_proton_fluxes.rename(
                    columns={
                        f'H_Flux_{i}': f'Ion_Flux_{i}'
                    }
                )

                df_proton_uncertainties = df_proton_uncertainties.rename(
                    columns={
                        f'H_Uncertainty_{i}': f'Ion_Uncertainty_{i}'
                    }
                )

        elif data_type == 'l2':
            e_low = df_energies['H_Bins_Low_Energy']
            e_high = []
            channels = range(len(df_energies['H_Bins_Low_Energy']))

            for i in channels:
                e_high.append(
                    e_low[i] + df_energies['H_Bins_Width'][i]
                )

                df_proton_fluxes = df_proton_fluxes.rename(
                    columns={
                        f'H_Flux_{i}': f'Ion_Flux_{i}'
                    }
                )

                df_proton_uncertainties = df_proton_uncertainties.rename(
                    columns={
                        f'H_Uncertainty_{i}': f'Ion_Uncertainty_{i}'
                    }
                )

    elif instrument == 'step':
        if data_type == 'l2':
            e_low = df_energies['Bins_Low_Energy']
            e_high = []
            channels = range(len(df_energies['Bins_Low_Energy']))

            df_proton_fluxes = pd.DataFrame()
            df_proton_uncertainties = pd.DataFrame()

            for i in channels:
                e_high.append(
                    e_low[i] + df_energies['Bins_Width'][i]
                )

                df_proton_fluxes[f'Ion_Flux_{i}'] = (
                    df_protons[f'Magnet_Avg_Flux_{i}'][plotstart:plotend]
                )

                df_proton_uncertainties[f'Ion_Uncertainty_{i}'] = (
                    df_protons[
                        f'Magnet_Avg_Uncertainty_{i}'
                    ][plotstart:plotend]
                )

        # Clean up negative flux values in STEP data.
        df_proton_fluxes[df_proton_fluxes < 0] = np.nan

    # Average the data if a resampling interval was provided.
    if averaging is not None:
        if instrument == 'ept':
            df_proton_fluxes = (
                df_proton_fluxes.resample(averaging).mean()
            )

            df_proton_uncertainties = (
                df_proton_uncertainties
                .resample(averaging)
                .apply(average_flux_error)
            )

        # For STEP protons, resampling is done independently.
        if instrument != 'step':
            df_proton_fluxes = (
                df_proton_fluxes.resample(averaging).mean()
            )

            df_proton_uncertainties = (
                df_proton_uncertainties
                .resample(averaging)
                .apply(average_flux_error)
            )

    # Main information DataFrame.
    df_info = pd.DataFrame(
        {
            'Plot_period': [],
            'Averaging': [],
            'Energy_channel': [],
            'Primary_energy': []
        }
    )

    # Add basic metadata.
    df_info['Plot_period'] = (
        [plotstart] + [plotend] + [''] * (len(channels) - 2)
    )

    if averaging is None:
        df_info['Averaging'] = (
            ['No averaging'] + [''] * (len(channels) - 1)
        )

    else:
        df_info['Averaging'] = (
            ['Mean', 'Resampled to ' + averaging]
            + [''] * (len(channels) - 2)
        )

    # Energy bin primary energies; geometric mean.
    primary_energies = []

    for i in range(len(e_low)):
        primary_energies.append(
            np.sqrt(e_low[i] * e_high[i])
        )

    primary_energies_channels = [
        primary_energies[i] for i in channels
    ]

    df_info['Primary_energy'] = primary_energies_channels

    # Calculate energy errors for the spectrum plot.
    energy_error_low = []
    energy_error_high = []

    for i in range(len(primary_energies)):
        energy_error_low.append(
            primary_energies[i] - e_low[i]
        )
        energy_error_high.append(
            e_high[i] - primary_energies[i]
        )

    df_info['Energy_error_low'] = [
        energy_error_low[i] for i in channels
    ]

    df_info['Energy_error_high'] = [
        energy_error_high[i] for i in channels
    ]

    # Calculate particle velocity from the primary energy.
    # The velocity is in km/s.
    velocity = []

    for energy in primary_energies:
        velocity.append(evolt2speed(energy, 1))

    # Calculate the search period using the velocity dispersion.
    # Convert travel distance from AU to km.
    travel_distance = travel_distance * 1.496E8

    DV = []

    for v in velocity:
        DV.append(travel_distance / v)

    searchstart = []

    for i in DV:
        searchstart.append(
            pd.to_datetime(t_inj) + pd.Timedelta(seconds=i)
        )

    searchend = []

    # Calculate search end time using the second slope if no fixed
    # search window is specified.
    if fixed_window is None:
        travel_distance_second_slope = (
            travel_distance_second_slope * 1.496E8
        )

        DV2 = []

        for v in velocity:
            DV2.append(travel_distance_second_slope / v)

        for i in DV2:
            searchend.append(
                pd.to_datetime(t_inj) + pd.Timedelta(seconds=i)
            )

    else:
        for i in searchstart:
            searchend.append(
                i + pd.to_timedelta(fixed_window)
            )

    if bg_distance_from_window is None:
        bg_start = bgstart
        bg_end = bgend

        bgstart = []
        bgend = []

        for i in range(len(searchstart)):
            bgstart.append(bg_start)
            bgend.append(bg_end)

    else:
        bgstart = []
        bgend = []

        for i in range(len(searchstart)):
            bgend.append(
                searchstart[i]
                - pd.to_timedelta(bg_distance_from_window)
            )

            bgstart.append(
                bgend[i] - pd.to_timedelta(bg_period)
            )

    # Calculate information from the data and append it to df_info.
    list_bg_fluxes = []
    list_flux_peaks = []
    list_peak_timestamps = []
    list_bg_subtracted_peaks = []
    list_peak_proton_uncertainties = []
    list_average_bg_uncertainties = []
    list_bg_std = []
    list_peak_significance = []
    list_flux_average = []
    list_bg_subtracted_average = []
    list_average_significance = []
    list_frac_nonan = []

    for n, channel in enumerate(channels):
        proton_flux = df_proton_fluxes[
            f'Ion_Flux_{channel}'
        ]

        proton_uncertainty = df_proton_uncertainties[
            f'Ion_Uncertainty_{channel}'
        ]

        # Background flux.
        b_f = proton_flux[searchstart[n]:searchend[n]]

        if len(b_f) == 0:
            bg_flux = np.nan
        else:
            bg_flux = proton_flux[
                bgstart[n]:bgend[n]
            ].mean(skipna=True)

        list_bg_fluxes.append(bg_flux)

        # Peak flux within the search window.
        f_p = proton_flux[searchstart[n]:searchend[n]]

        if f_p.notna().any():
            flux_peak = f_p.max()
        else:
            flux_peak = np.nan

        list_flux_peaks.append(flux_peak)

        # Fraction of non-NaN data points in the search window.
        if len(f_p) == 0:
            frac_nonan = np.nan
        else:
            frac_nonan = f_p.notna().mean()

        list_frac_nonan.append(frac_nonan)

        # Timestamp of the peak flux.
        if f_p.notna().any():
            peak_timestamp = f_p.idxmax()
        else:
            peak_timestamp = np.nan

        list_peak_timestamps.append(peak_timestamp)

        # Proton uncertainty at the peak timestamp.
        if pd.isna(peak_timestamp):
            list_peak_proton_uncertainties.append(np.nan)

        elif len(proton_uncertainty) == 0:
            list_peak_proton_uncertainties.append(np.nan)

        else:
            timestamp_loc = proton_uncertainty.index.get_indexer(
                [peak_timestamp],
                method='nearest'
            )[0]

            peak_proton_uncertainty = (
                proton_uncertainty.iloc[timestamp_loc]
            )

            list_peak_proton_uncertainties.append(
                peak_proton_uncertainty
            )

        # Average uncertainty in the background window.
        bg_uncertainty = proton_uncertainty[
            bgstart[n]:bgend[n]
        ]

        average_bg_uncertainty = (
            np.sqrt((bg_uncertainty ** 2).sum())
            / len(bg_uncertainty)
        )

        list_average_bg_uncertainties.append(
            average_bg_uncertainty
        )

        # Standard deviation of the background flux.
        bg_std = proton_flux[
            bgstart[n]:bgend[n]
        ].std()

        list_bg_std.append(bg_std)

        # Average flux within the search window.
        f_a = proton_flux[searchstart[n]:searchend[n]]

        if len(f_a) == 0:
            flux_average = np.nan
        else:
            flux_average = f_a.mean(skipna=True)

        list_flux_average.append(flux_average)

    # Calculate background-subtracted values and their significance.
    for i in range(len(list_flux_peaks)):
        list_bg_subtracted_peaks.append(
            list_flux_peaks[i] - list_bg_fluxes[i]
        )

        list_peak_significance.append(
            list_bg_subtracted_peaks[i] / list_bg_std[i]
        )

        # If the background is higher than the peak, mark the
        # significance as -1 so that the value can be excluded later.
        if list_bg_subtracted_peaks[i] < list_bg_fluxes[i]:
            list_peak_significance[i] = -1

        list_bg_subtracted_average.append(
            list_flux_average[i] - list_bg_fluxes[i]
        )

        list_average_significance.append(
            list_bg_subtracted_average[i] / list_bg_std[i]
        )

        # If the background is higher than the average flux, mark the
        # significance as -1 so that the value can be excluded later.
        if list_bg_subtracted_average[i] < list_bg_fluxes[i]:
            list_average_significance[i] = -1

    df_info['Energy_channel'] = channels
    df_info['Bg_start'] = bgstart
    df_info['Bg_end'] = bgend
    df_info['Searchstart'] = searchstart
    df_info['Searchend'] = searchend
    df_info['Peak_timestamp'] = list_peak_timestamps

    df_info['Background_flux'] = list_bg_fluxes
    df_info['Flux_peak'] = list_flux_peaks
    df_info['Bg_subtracted_peak'] = list_bg_subtracted_peaks
    df_info['Peak_proton_uncertainty'] = (
        list_peak_proton_uncertainties
    )
    df_info['Bg_proton_uncertainty'] = (
        list_average_bg_uncertainties
    )
    df_info['Peak_significance'] = list_peak_significance
    df_info['Flux_average'] = list_flux_average
    df_info['Bg_subtracted_average'] = (
        list_bg_subtracted_average
    )
    df_info['Average_significance'] = (
        list_average_significance
    )

    df_info['Backsub_peak_uncertainty'] = np.sqrt(
        df_info['Peak_proton_uncertainty'] ** 2
        + df_info['Bg_proton_uncertainty'] ** 2
    )

    df_info['rel_backsub_peak_err'] = np.abs(
        df_info['Backsub_peak_uncertainty']
        / df_info['Bg_subtracted_peak']
    )

    df_info['frac_nonan'] = list_frac_nonan

    return (
        df_proton_fluxes,
        df_info,
        [searchstart, searchend],
        [e_low, e_high],
        [instrument, data_type]
    )

def plot_channels(
    args,
    species='electron',
    bg_subtraction=False,
    savefig=False,
    sigma=3,
    path='',
    key='',
    frac_nan_threshold=0.4,
    rel_err_threshold=0.5,
    plot_pa=False,
    coverage=None,
    viewing='sun',
    centre_pix=False,
    date=None,
    size=20
):
    """
    Creates a timeseries plot showing the particle flux for each energy
    channel of the instrument (STEP, EPT, HET). The timeseries plot also
    shows the peak search window and background window.

    The function works for both electrons and protons. The particle species
    is selected using the ``species`` keyword.

    The peak is marked with different colored lines:
        green: peak is acceptable
        grey: too many NaNs in search window
        blue: low significance
        orange: high relative error
        purple: no valid background subtraction/significance

    Parameters
    ----------
    args : tuple
        Output of the corresponding extraction function. Contains:

        args[0] : pandas.DataFrame
            Particle fluxes.

        args[1] : pandas.DataFrame
            Spectrum data and metadata.

        args[2] : list
            Search-window start and end times.

        args[3] : list
            Lower and upper energy boundaries for each channel.

        args[4] : list
            Instrument and data type.

    species : str, optional
        Particle species to plot. Accepted values are ``'electron'``,
        ``'electrons'``, ``'e'``, ``'proton'``, ``'protons'``, and ``'p'``.
        Defaults to ``'electron'``.

    bg_subtraction : bool, optional
        If True, subtract the background flux from the particle flux.
        Negative flux values after subtraction are set to NaN.
        Defaults to False.

    savefig : bool, optional
        If True, save the timeseries plot. Defaults to False.

    sigma : int, optional
        Significance threshold used to determine whether the peak is
        significant enough. Defaults to 3.

    path : str, optional
        Path to the folder where the timeseries plot will be saved.
        Defaults to ''.

    key : str, optional
        Additional string appended to the output filename. Defaults to ''.

    frac_nan_threshold : float, optional
        Minimum fraction of non-NaN flux data points required in the search
        interval. Channels below this threshold are considered unreliable.
        Defaults to 0.4.

    rel_err_threshold : float, optional
        Maximum allowed relative error. Channels above this threshold are
        considered unreliable. Defaults to 0.5.

    plot_pa : bool, optional
        If True, include pitch-angle coverage in the plot. Defaults to False.

    coverage : pandas.DataFrame or None, optional
        DataFrame containing the pitch-angle coverage used for plotting.
        Required if ``plot_pa=True``. Defaults to None.

    viewing : str or None, optional
        Viewing direction of EPT or HET, used for plotting pitch angles.
        Ignored for STEP. If None, ``'sun'`` is used. Defaults to ``'sun'``.

    centre_pix : bool, optional
        Refers to STEP data and indicates whether centre-pixel data are
        being used. If True, ``-centre_pix`` is added to the output
        filename. Defaults to False.

    date : str or None, optional
        Date used for the plot title and filename. If None, the date is
        taken from the plot period in df_info. Defaults to None.

    size : int, optional
        Base font size used in the plot. Defaults to 20.

    Raises
    ------
    ValueError
        If an unsupported particle species is provided.

    Notes
    -----
    Electron flux columns are expected to be named
    ``Electron_Flux_{channel}``.

    Proton flux columns are expected to be named
    ``Proton_Flux_{channel}``.

    The latter is intentional: proton data use the ``Proton`` naming
    convention rather than the ``Ion`` naming convention used internally
    by some extraction functions.
    """

    # ------------------------------------------------------------------
    # Normalize particle species.
    # ------------------------------------------------------------------

    species_lower = species.lower()

    if species_lower in ['electron', 'electrons', 'e']:
        species = 'electron'

    elif species_lower in ['proton', 'protons', 'p']:
        species = 'proton'

    else:
        raise ValueError(
            "species must be 'electron' or 'proton'."
        )

    # Column name used by the plotting dataframe.
    flux_column_prefix = (
        'Electron_Flux'
        if species == 'electron'
        else 'Proton_Flux'
    )

    # ------------------------------------------------------------------
    # Extract information from args.
    # ------------------------------------------------------------------

    peak_sig = args[1]['Peak_significance']
    rel_err = args[1]['rel_backsub_peak_err']

    df_fluxes = args[0]
    df_info = args[1]
    search_area = args[2]
    energy_bin = args[3]
    instrument = args[4][0]
    data_type = args[4][1]

    # ------------------------------------------------------------------
    # Date and filename information.
    # ------------------------------------------------------------------

    if date is None:
        date_string = str(df_info['Plot_period'][0][:-5])
        file_date = str(df_info['Plot_period'][0][:-5])

    else:
        date_string = str(date)[:-3]
        file_date = (
            str(date)[:-3]
            .replace(' ', '-')
            .replace(':', '')
        )

    if viewing is None or instrument.lower() == 'step':
        viewing = 'sun'

    title_string = (
        instrument.upper()
        + ', '
        + species.upper()
        + 'S, '
        + data_type.upper()
        + ', '
        + date_string
    )

    filename = (
        species
        + '_channels-'
        + file_date
        + '-'
        + instrument.upper()
        + '-'
        + viewing
        + '-'
        + data_type.upper()
    )

    # ------------------------------------------------------------------
    # Averaging information.
    # ------------------------------------------------------------------

    if df_info['Averaging'][0] == 'Mean':

        averaging = df_info['Averaging'][1].split()[2]

        title_string = (
            title_string
            + ', '
            + averaging
            + ' averaging'
        )

        filename = (
            filename
            + '-'
            + averaging
            + '_averaging'
        )

    elif df_info['Averaging'][0] == 'No averaging':

        title_string = title_string + ', no averaging'
        filename = filename + '-no_averaging'

    # ------------------------------------------------------------------
    # EPT ion-contamination correction.
    # ------------------------------------------------------------------

    if instrument.lower() == 'ept' and species == 'electron':

        if df_info['Ion_contamination_correction'][0]:

            title_string = title_string + ', ion correction on'
            filename = filename + '-ion_corr'

        elif df_info['Ion_contamination_correction'][0] == False:

            title_string = title_string + ', ion correction off'

    # ------------------------------------------------------------------
    # STEP centre-pixel information.
    # ------------------------------------------------------------------

    if instrument.lower() == 'step' and centre_pix:

        filename = filename + '-centre_pix'
        title_string = title_string + ', centre pix'

    # ------------------------------------------------------------------
    # Background subtraction.
    # ------------------------------------------------------------------

    if bg_subtraction:

        title_string = title_string + ', bg subtraction on'
        filename = filename + '-bg_subtr'

        df_fluxes = df_fluxes.sub(
            df_info['Background_flux'].values,
            axis=1
        )

        # Negative background-subtracted fluxes are invalid.
        df_fluxes[df_fluxes < 0] = np.nan

    else:

        title_string = title_string + ', bg subtraction off'

    # ------------------------------------------------------------------
    # Plot configuration.
    # ------------------------------------------------------------------

    color = {
        'sun': 'crimson',
        'asun': 'orange',
        'north': 'darkslateblue',
        'south': 'c'
    }

    npanels = len(df_info['Energy_channel'])

    if plot_pa:
        npanels += 1

    if instrument.lower() == 'step':

        n_channels_step = len(df_info['Energy_channel'])

        if n_channels_step > 8:
            fsize = (20, 60)
        else:
            fsize = (20, 24)

    elif instrument.lower() == 'ept':

        fsize = (20, 48)

    elif instrument.lower() == 'het':

        # Keep the electron behaviour for HET.
        # The original proton function used (20, 48), while the electron
        # function used (20, 12).
        if species == 'electron':
            fsize = (20, 12)
        else:
            fsize = (20, 48)

    fig, axes = plt.subplots(
        npanels,
        sharex=True,
        figsize=fsize
    )

    fig.supylabel(
        "Intensity [1/s cm$^2$ sr MeV]",
        size=size
    )

    axes[0].set_title(
        title_string + "\n",
        size=size
    )

    # ------------------------------------------------------------------
    # Plot each energy channel.
    # ------------------------------------------------------------------

    for n, channel in enumerate(df_info['Energy_channel']):

        ax = axes[n]

        ax.plot(
            df_fluxes.index,
            df_fluxes[
                f'{flux_column_prefix}_{channel}'
            ],
            color=color[viewing],
            drawstyle='steps-mid'
        )

        ax.set_yscale('log')

        plt.text(
            0.025,
            0.7,
            str(energy_bin[0][channel])
            + " - "
            + str(energy_bin[1][channel])
            + " MeV",
            transform=ax.transAxes,
            size=size - 2
        )

        ax.tick_params(
            axis='y',
            which='major',
            labelsize=size - 2
        )

        # --------------------------------------------------------------
        # Search area.
        # --------------------------------------------------------------

        ax.axvline(
            search_area[0][n],
            color='black'
        )

        ax.axvline(
            search_area[1][n],
            color='black'
        )

        ax.set_xlim(
            df_fluxes.index[0],
            df_fluxes.index[-1]
        )

        # --------------------------------------------------------------
        # Peak marker.
        # --------------------------------------------------------------

        if df_info['Peak_timestamp'][n] is not pd.NaT:

            if rel_err[n] > rel_err_threshold:

                ax.axvline(
                    df_info['Peak_timestamp'][n],
                    linestyle=':',
                    linewidth=4,
                    color='orange'
                )

            if df_info['frac_nonan'][n] < frac_nan_threshold:

                ax.axvline(
                    df_info['Peak_timestamp'][n],
                    linestyle='--',
                    linewidth=3,
                    color='gray'
                )

            if peak_sig[n] < sigma:

                ax.axvline(
                    df_info['Peak_timestamp'][n],
                    linestyle='-.',
                    linewidth=2,
                    color='blue'
                )

            if (
                peak_sig[n] >= sigma
                and rel_err[n] <= rel_err_threshold
                and df_info['frac_nonan'][n] > frac_nan_threshold
            ):

                ax.axvline(
                    df_info['Peak_timestamp'][n],
                    color='green'
                )

            if bg_subtraction:

                if (
                    np.isnan(peak_sig[n])
                    and ~np.isnan(
                        df_info['Bg_subtracted_peak'][n]
                    )
                ):

                    ax.axvline(
                        df_info['Peak_timestamp'][n],
                        linestyle='-',
                        linewidth=2,
                        color='purple'
                    )

            else:

                if (
                    np.isnan(peak_sig[n])
                    and df_info['Flux_average'][n] != 0.
                ):

                    ax.axvline(
                        df_info['Peak_timestamp'][n],
                        linestyle='-',
                        linewidth=2,
                        color='purple'
                    )

        # --------------------------------------------------------------
        # Background measurement area.
        # --------------------------------------------------------------

        ax.axvspan(
            df_info['Bg_start'][n],
            df_info['Bg_end'][n],
            color='gray',
            alpha=0.25
        )

        ax.get_xaxis().set_visible(False)

        if (
            n == len(df_info['Energy_channel']) - 1
            and not plot_pa
        ):

            ax.get_xaxis().set_visible(True)

            ax.set_xlabel(
                "Time",
                labelpad=45
            )

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%d-%m-%y\n%H:%M")
            )

    # ------------------------------------------------------------------
    # Pitch-angle panel.
    # ------------------------------------------------------------------

    if plot_pa:

        ax = axes[len(df_info['Energy_channel'])]

        if instrument.lower() in ['het', 'ept']:

            col = color[viewing]

            # Fill the minimum-maximum range of the pitch-angle coverage.
            ax.fill_between(
                coverage.index,
                coverage[viewing]['min'],
                coverage[viewing]['max'],
                alpha=0.5,
                color=col,
                edgecolor=col,
                linewidth=0.0,
                step='mid'
            )

            # Plot the central pitch angle.
            ax.plot(
                coverage.index,
                coverage[viewing]['center'],
                linewidth=0.7,
                label=viewing,
                color=col,
                drawstyle='steps-mid'
            )

        if instrument.lower() == 'step':

            col_list = plt.cm.viridis(
                np.linspace(0., 0.95, 16)
            )

            for p in range(1, 16):

                ax.plot(
                    coverage.index,
                    coverage[f'Pixel_{p}']['center'],
                    color=col_list[p - 1],
                    linewidth=1,
                    label=f'Pixel_{p}',
                    drawstyle='steps-mid'
                )

        ax.axhline(
            y=90,
            color='gray',
            linewidth=0.8,
            linestyle='--'
        )

        ax.axhline(
            y=45,
            color='gray',
            linewidth=0.8,
            linestyle='--'
        )

        ax.axhline(
            y=135,
            color='gray',
            linewidth=0.8,
            linestyle='--'
        )

        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title=instrument
        )

        ax.set_ylim([0, 180])

        ax.yaxis.set_ticks(
            np.arange(0, 180 + 45, 45)
        )

        ax.set_ylabel(
            'PA [°]',
            size=size - 2
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d-%m-%y\n%H:%M")
        )

        plt.tick_params(
            axis='x',
            which='major',
            labelsize=size - 2
        )

        plt.tick_params(
            axis='y',
            which='major',
            labelsize=size - 2
        )

        ax.set_xlabel(
            "Time",
            labelpad=45,
            size=size
        )

    # ------------------------------------------------------------------
    # Save figure.
    # ------------------------------------------------------------------

    if path and not path.endswith('/'):
        path += '/'

    if savefig:

        plt.savefig(
            path + filename + str(key) + '.jpg',
            bbox_inches='tight',
            dpi=300
        )

    plt.show()


def plot_channels_protons(
    args,
    bg_subtraction=False,
    savefig=False,
    sigma=3,
    path='',
    key='',
    frac_nan_threshold=0.4,
    rel_err_threshold=0.5,
    plot_pa=False,
    coverage=None,
    viewing='sun',
    date=None,
    size=20
):
    """
    Creates a timeseries plot showing the proton flux for each energy channel
    of the instrument (STEP, EPT, HET). The timeseries plot also shows the
    peak search window and background window.

    The peak is marked with different colored lines:
        green: peak is acceptable
        grey: too many NaNs in search window
        blue: low significance
        orange: high relative error
        purple: no valid background subtraction/significance

    Args:
        args : tuple
            Output of the extract_proton_data function. Contains:
                df_proton_fluxes: pandas DataFrame
                df_info: pandas DataFrame containing the spectrum data
                    and metadata
                [searchstart, searchend]: search window start and end times
                [e_low, e_high]: lowest and highest energy for each channel
                [instrument, data_type]: instrument and data type

        bg_subtraction (bool, optional):
            Subtract the background flux from the data. Defaults to False.

        savefig (bool, optional):
            If True, save the timeseries plot. Defaults to False.

        sigma (int, optional):
            Significance threshold used to determine whether the peak is
            significant enough. Defaults to 3.

        path (str, optional):
            Path to the folder where the timeseries plot will be saved.
            Defaults to ''.

        key (str, optional):
            Additional string appended to the output filename. Defaults to ''.

        frac_nan_threshold (float, optional):
            Minimum fraction of non-NaN flux data points required in the
            search interval. Channels below this threshold are considered
            unreliable. Defaults to 0.4.

        rel_err_threshold (float, optional):
            Maximum allowed relative error. Channels above this threshold
            are considered unreliable. Defaults to 0.5.

        plot_pa (bool, optional):
            If True, include pitch-angle coverage in the plot.
            Defaults to False.

        coverage (pandas DataFrame or None, optional):
            DataFrame containing the pitch-angle coverage used for plotting.
            Defaults to None.

        viewing (str, optional):
            Viewing direction of EPT or HET, used for plotting pitch angles.
            Ignored for STEP. Defaults to 'sun'.

        date (str, optional):
            Date used for the plot title and filename. If None, the date is
            taken from the plot period in df_info. Defaults to None.

        size (int, optional):
            Base font size used in the plot. Defaults to 20.
    """

    peak_sig = args[1]['Peak_significance']
    rel_err = args[1]['rel_backsub_peak_err']

    df_proton_fluxes = args[0]
    df_info = args[1]
    search_area = args[2]
    energy_bin = args[3]
    instrument = args[4][0]
    data_type = args[4][1]

    date_string = ''
    file_date = ''

    if date is None:
        date_string = str(df_info['Plot_period'][0][:-5])
        file_date = str(df_info['Plot_period'][0][:-5])

    else:
        date_string = str(date)[:-3]
        file_date = str(date)[:-3].replace(' ', '-').replace(':', '')

    if viewing is None or instrument.lower() == 'step':
        viewing = 'sun'

    title_string = (
        instrument.upper()
        + ', PROTONS, '
        + data_type.upper()
        + ', '
        + date_string
    )

    filename = (
        'proton_channels-'
        + file_date
        + '-'
        + instrument.upper()
        + '-'
        + viewing
        + '-'
        + data_type.upper()
    )

    if df_info['Averaging'][0] == 'Mean':

        title_string = (
            title_string
            + ', '
            + df_info['Averaging'][1].split()[2]
            + ' averaging'
        )

        filename = (
            filename
            + '-'
            + df_info['Averaging'][1].split()[2]
            + '_averaging'
        )

    elif df_info['Averaging'][0] == 'No averaging':

        title_string = title_string + ', no averaging'
        filename = filename + '-no_averaging'

    if bg_subtraction:

        title_string = title_string + ', bg subtraction on'
        filename = filename + '-bg_subtr'

    else:

        title_string = title_string + ', bg subtraction off'

    # If background subtraction is enabled, subtract background flux from
    # all observations. Negative flux values are set to NaN.
    if bg_subtraction:

        df_proton_fluxes = df_proton_fluxes.sub(
            df_info['Background_flux'].values,
            axis=1
        )

        df_proton_fluxes[df_proton_fluxes < 0] = np.nan

    # Plotting.
    color = {
        'sun': 'crimson',
        'asun': 'orange',
        'north': 'darkslateblue',
        'south': 'c'
    }

    npanels = len(df_info['Energy_channel'])

    if plot_pa:
        npanels += 1

    if instrument.lower() == 'step':

        n_channels_step = len(df_info['Energy_channel'])

        if n_channels_step > 8:
            fsize = (20, 60)
        else:
            fsize = (20, 24)

    elif instrument.lower() in ['ept', 'het']:

        fsize = (20, 48)

    fig, axes = plt.subplots(
        npanels,
        sharex=True,
        figsize=fsize
    )

    fig.supylabel(
        "Intensity [1/s cm$^2$ sr MeV]",
        size=size
    )

    axes[0].set_title(
        title_string + "\n",
        size=size
    )

    # Loop through selected energy channels and create a subplot for each.
    n = 0

    for channel in df_info['Energy_channel']:

        ax = axes[n]

        ax.plot(
            df_proton_fluxes.index,
            df_proton_fluxes['Ion_Flux_{}'.format(channel)],
            color=color[viewing],
            drawstyle='steps-mid'
        )

        ax.set_yscale('log')

        plt.text(
            0.025,
            0.7,
            str(energy_bin[0][channel])
            + " - "
            + str(energy_bin[1][channel])
            + " MeV",
            transform=ax.transAxes,
            size=size - 2
        )

        ax.tick_params(
            axis='y',
            which='major',
            labelsize=size - 2
        )

        # Search area vertical lines.
        ax.axvline(
            search_area[0][n],
            color='black'
        )

        ax.axvline(
            search_area[1][n],
            color='black'
        )

        ax.set_xlim(
            df_proton_fluxes.index[0],
            df_proton_fluxes.index[-1]
        )

        # Peak vertical line.
        if df_info['Peak_timestamp'][n] is not pd.NaT:

            if rel_err[n] > rel_err_threshold:

                ax.axvline(
                    df_info['Peak_timestamp'][n],
                    linestyle=':',
                    linewidth=4,
                    color='orange'
                )

            if df_info['frac_nonan'][n] < frac_nan_threshold:

                ax.axvline(
                    df_info['Peak_timestamp'][n],
                    linestyle='--',
                    linewidth=3,
                    color='gray'
                )

            if peak_sig[n] < sigma:

                ax.axvline(
                    df_info['Peak_timestamp'][n],
                    linestyle='-.',
                    linewidth=2,
                    color='blue'
                )

            if (
                peak_sig[n] >= sigma
                and rel_err[n] <= rel_err_threshold
                and df_info['frac_nonan'][n] > frac_nan_threshold
            ):

                ax.axvline(
                    df_info['Peak_timestamp'][n],
                    color='green'
                )

            if bg_subtraction:

                if (
                    np.isnan(peak_sig[n])
                    and ~np.isnan(df_info['Bg_subtracted_peak'][n])
                ):

                    ax.axvline(
                        df_info['Peak_timestamp'][n],
                        linestyle='-',
                        linewidth=2,
                        color='purple'
                    )

            else:

                if (
                    np.isnan(peak_sig[n])
                    and df_info['Flux_average'][n] != 0.
                ):

                    ax.axvline(
                        df_info['Peak_timestamp'][n],
                        linestyle='-',
                        linewidth=2,
                        color='purple'
                    )

        # Background measurement area.
        ax.axvspan(
            df_info['Bg_start'][n],
            df_info['Bg_end'][n],
            color='gray',
            alpha=0.25
        )

        ax.get_xaxis().set_visible(False)

        if (
            n == len(df_info['Energy_channel']) - 1
            and not plot_pa
        ):

            ax.get_xaxis().set_visible(True)

            ax.set_xlabel(
                "Time",
                labelpad=45
            )

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%d-%m-%y\n%H:%M")
            )

        n += 1

    if plot_pa:

        # Add a panel that shows the pitch angle of the telescope.
        ax = axes[n]

        if instrument.lower() in ['het', 'ept']:

            col = color[viewing]

            # Fill the minimum-maximum range of the pitch-angle coverage.
            ax.fill_between(
                coverage.index,
                coverage[viewing]['min'],
                coverage[viewing]['max'],
                alpha=0.5,
                color=col,
                edgecolor=col,
                linewidth=0.0,
                step='mid'
            )

            # Plot the central pitch angle as a thin line.
            ax.plot(
                coverage.index,
                coverage[viewing]['center'],
                linewidth=0.7,
                label=viewing,
                color=col,
                drawstyle='steps-mid'
            )

        if instrument.lower() == 'step':

            col_list = plt.cm.viridis(
                np.linspace(0., 0.95, 16)
            )

            for p in range(1, 16):

                # Plot the central pitch angle as a thin line.
                ax.plot(
                    coverage.index,
                    coverage[f'Pixel_{p}']['center'],
                    color=col_list[p - 1],
                    linewidth=1,
                    label=f'Pixel_{p}',
                    drawstyle='steps-mid'
                )

        ax.axhline(
            y=90,
            color='gray',
            linewidth=0.8,
            linestyle='--'
        )

        ax.axhline(
            y=45,
            color='gray',
            linewidth=0.8,
            linestyle='--'
        )

        ax.axhline(
            y=135,
            color='gray',
            linewidth=0.8,
            linestyle='--'
        )

        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title=instrument
        )

        ax.set_ylim([0, 180])

        ax.yaxis.set_ticks(
            np.arange(0, 180 + 45, 45)
        )

        ax.set_ylabel(
            'PA [°]',
            size=size - 2
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d-%m-%y\n%H:%M")
        )

        plt.tick_params(
            axis='x',
            which='major',
            labelsize=size - 2
        )

        plt.tick_params(
            axis='y',
            which='major',
            labelsize=size - 2
        )

        ax.set_xlabel(
            "Time",
            labelpad=45,
            size=size
        )

    # Save figure, if enabled.
    if path[len(path) - 1] != '/':
        path = path + '/'

    if savefig:

        plt.savefig(
            path + filename + str(key) + '.jpg',
            bbox_inches='tight',
            dpi=300
        )

    plt.show()


def plot_some_channels(
    args,
    bg_subtraction=False,
    savefig=False,
    sigma=3,
    path='',
    key='',
    plot_pa=False,
    coverage=None,
    viewing='sun',
    frac_nan_threshold=0.9,
    rel_err_threshold=0.5,
    channels=None,
    figsize_x=15,
    figsize_y=8,
    f_scale=1,
    f_size=12,
    species='electron',
    centre_pix=False
):
    """
    Creates a timeseries plot for selected energy channels of the
    instrument (STEP, EPT, HET).

    The function can be used for both electron and proton data. The
    particle species is selected using ``species``.

    The timeseries plots show the peak search window and background
    window. The peak is marked with different color lines:

        green: peak is ok
        grey: too many NaNs in window
        blue: low significance
        orange: high relative error

    If ``channels`` is None, all available energy channels are plotted.
    Otherwise, ``channels`` must be a list containing valid energy channel
    numbers for the supplied data.

    Args:
        args: Output of the extract_particle_data function. Contains:
            df_fluxes: pandas DataFrame containing particle fluxes.
            df_info: pandas DataFrame containing spectrum data and metadata.
            [searchstart, searchend]: search window start and end times.
            [e_low, e_high]: lowest and highest energy for each channel.
            [instrument, data_type]: instrument and data type.

        bg_subtraction (bool, optional):
            Subtract background from the data. Defaults to False.

        savefig (bool, optional):
            Save the timeseries plot. Defaults to False.

        sigma (int, optional):
            Significance threshold used to check whether the peak is
            significant. Defaults to 3.

        path (str, optional):
            Path to the folder where the timeseries plot is saved.
            Defaults to ''.

        key (str, optional):
            Additional string added to the filename. Defaults to ''.

        plot_pa (bool, optional):
            Include a pitch-angle panel. Defaults to False.

        coverage (pandas.DataFrame or None, optional):
            DataFrame used to plot the pitch-angle coverage.
            Defaults to None.

        viewing (str or None, optional):
            Viewing direction of EPT or HET used for plotting pitch angles.
            Defaults to 'sun'.

        frac_nan_threshold (float, optional):
            Threshold for the fraction of non-NaN data points in the
            search interval. Defaults to 0.9.

        rel_err_threshold (float, optional):
            Maximum allowed relative error. Defaults to 0.5.

        channels (list or None, optional):
            Energy channels to plot. If None, all available channels are
            plotted. Defaults to None.

        figsize_x (float, optional):
            Figure width. Defaults to 15.

        figsize_y (float, optional):
            Figure height. Defaults to 8.

        f_scale (float, optional):
            Font scaling factor. Defaults to 1.

        f_size (int, optional):
            Base font size. Defaults to 12.

        species (str, optional):
            Particle species to plot. Accepted values are
            'electron', 'electrons', 'e', 'proton', 'protons', or 'p'.
            Defaults to 'electron'.

        centre_pix (bool, optional):
            Refers to STEP data and indicates whether centre-pixel data
            are being used. Defaults to False.

    Raises:
        ValueError:
            If an unsupported particle species is supplied.

        ValueError:
            If one or more requested channels are not available in the
            supplied data.
    """

    # Normalize species name.
    species_lower = species.lower()

    if species_lower in ['electron', 'electrons', 'e']:
        species = 'electron'

    elif species_lower in ['proton', 'protons', 'p']:
        species = 'proton'

    else:
        raise ValueError(
            "Unsupported species. Use 'electron' or 'proton'."
        )

    # Particle-specific column prefix.
    flux_prefix = (
        'Electron_Flux'
        if species == 'electron'
        else 'Proton_Flux'
    )

    peak_sig = args[1]['Peak_significance']
    rel_err = args[1]['rel_backsub_peak_err']

    df_fluxes = args[0]
    df_info = args[1]
    search_area = args[2]
    energy_bin = args[3]
    instrument = args[4][0]
    data_type = args[4][1]

    # Available channels are taken directly from the supplied data.
    available_channels = list(df_info['Energy_channel'])

    # If no channels are specified, plot all available channels.
    if channels is None:

        channels = available_channels.copy()

    else:

        invalid_channels = [
            channel for channel in channels
            if channel not in available_channels
        ]

        if invalid_channels:

            raise ValueError(
                f"Invalid channel(s): {invalid_channels}. "
                f"Available channels are: {available_channels}."
            )

    # Make sure viewing is defined for STEP, where it is not used.
    if viewing is None or instrument.lower() == 'step':
        viewing = 'sun'

    title_string = (
        instrument.upper()
        + ', '
        + species.upper()
        + ', '
        + data_type.upper()
        + ', '
        + str(df_info['Plot_period'][0][:-5])
    )

    filename = (
        species
        + '_channels-'
        + str(df_info['Plot_period'][0][:-5])
        + '-'
        + instrument.upper()
        + '-'
        + data_type.upper()
    )

    if df_info['Averaging'][0] == 'Mean':

        title_string = (
            title_string
            + ', '
            + df_info['Averaging'][1].split()[2]
            + ' averaging'
        )

        filename = (
            filename
            + '-'
            + df_info['Averaging'][1].split()[2]
            + '_averaging'
        )

    elif df_info['Averaging'][0] == 'No averaging':

        title_string = title_string + ', no averaging'
        filename = filename + '-no_averaging'

    if bg_subtraction:

        title_string = title_string + ', bg subtraction on'
        filename = filename + '-bg_subtr'

    else:

        title_string = title_string + ', bg subtraction off'

    # EPT ion-contamination correction is relevant only for electrons.
    if (
        instrument.lower() == 'ept'
        and species == 'electron'
    ):

        if df_info['Ion_contamination_correction'][0]:

            title_string = title_string + ', ion correction on'
            filename = filename + '-ion_corr'

        elif df_info['Ion_contamination_correction'][0] is False:

            title_string = title_string + ', ion correction off'

    # Add centre-pixel information for STEP.
    if instrument.lower() == 'step' and centre_pix:

        filename = filename + '-centre_pix'
        title_string = title_string + ', centre pix'

    # If background subtraction is enabled, subtract background from
    # all observations. Negative flux values are set to NaN.
    if bg_subtraction:

        df_fluxes = df_fluxes.sub(
            df_info['Background_flux'].values,
            axis=1
        )

        df_fluxes[df_fluxes < 0] = np.nan

    # Plotting part.
    sns.set_theme(
        style="white",
        font_scale=f_scale
    )

    # One panel for each selected channel, plus one optional
    # pitch-angle panel.
    npanels = len(channels)

    if plot_pa:
        npanels += 1

    fig = plt.figure(
        figsize=(figsize_x, figsize_y)
    )

    plt.xticks(
        [],
        fontsize=f_size
    )

    plt.yticks(
        [],
        fontsize=f_size
    )

    plt.ylabel(
        "Intensity \n [1/s cm$^2$ sr MeV] \n \n",
        size=f_size
    )

    plt.xlabel(
        "\n \n Time",
        size=f_size
    )

    plt.title(
        title_string,
        size=f_size
    )

    # Loop through selected energy channels.
    for n, channel in enumerate(channels, start=1):

        if plot_pa:
            ax = fig.add_subplot(
                npanels,
                1,
                n
            )

        else:
            ax = fig.add_subplot(
                len(channels),
                1,
                n
            )

        ax = df_fluxes[
            '{}_{}'.format(flux_prefix, channel)
        ].plot(
            logy=True,
            figsize=(figsize_x, figsize_y),
            color='red',
            drawstyle='steps-mid'
        )

        plt.text(
            0.025,
            0.7,
            str(energy_bin[0][channel])
            + " - "
            + str(energy_bin[1][channel])
            + " MeV",
            transform=ax.transAxes,
            size=f_size
        )

        # Search area vertical lines.
        channel_index = available_channels.index(channel)

        ax.axvline(
            search_area[0][channel_index],
            color='black'
        )

        ax.axvline(
            search_area[1][channel_index],
            color='black'
        )

        # Peak vertical line.
        if df_info['Peak_timestamp'][channel_index] is not pd.NaT:

            if rel_err[channel_index] > rel_err_threshold:

                ax.axvline(
                    df_info['Peak_timestamp'][channel_index],
                    linestyle=':',
                    linewidth=4,
                    color='orange'
                )

            if df_info['frac_nonan'][channel_index] < frac_nan_threshold:

                ax.axvline(
                    df_info['Peak_timestamp'][channel_index],
                    linestyle='--',
                    linewidth=3,
                    color='gray'
                )

            if peak_sig[channel_index] < sigma:

                ax.axvline(
                    df_info['Peak_timestamp'][channel_index],
                    linestyle='-.',
                    linewidth=2,
                    color='blue'
                )

            if (
                peak_sig[channel_index] >= sigma
                and rel_err[channel_index] <= rel_err_threshold
                and df_info['frac_nonan'][channel_index] > frac_nan_threshold
            ):

                ax.axvline(
                    df_info['Peak_timestamp'][channel_index],
                    color='green'
                )

        # Background measurement area.
        ax.axvspan(
            df_info['Bg_start'][channel_index],
            df_info['Bg_end'][channel_index],
            color='gray',
            alpha=0.25
        )

        ax.get_xaxis().set_visible(False)

        # Show the time axis only on the last timeseries panel
        # if there is no pitch-angle panel.
        if n == len(channels) and not plot_pa:

            ax.get_xaxis().set_visible(True)

            plt.xlabel("")

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%d-%m-%y\n%H:%M")
            )

    # Optional pitch-angle panel.
    if plot_pa:

        ax = fig.add_subplot(
            npanels,
            1,
            npanels
        )

        color = {
            'sun': 'crimson',
            'asun': 'orange',
            'north': 'darkslateblue',
            'south': 'c'
        }

        if instrument.lower() in ['het', 'ept']:

            col = color[viewing]

            ax.fill_between(
                coverage.index,
                coverage[viewing]['min'],
                coverage[viewing]['max'],
                alpha=0.5,
                color=col,
                edgecolor=col,
                linewidth=0.0,
                step='mid'
            )

            ax.plot(
                coverage.index,
                coverage[viewing]['center'],
                linewidth=0.7,
                label=viewing,
                color=col,
                drawstyle='steps-mid'
            )

        if instrument.lower() == 'step':

            col_list = plt.cm.viridis(
                np.linspace(0., 0.95, 16)
            )

            for p in range(1, 16):

                ax.plot(
                    coverage.index,
                    coverage[f'Pixel_{p}']['center'],
                    color=col_list[p - 1],
                    linewidth=1,
                    label=f'Pixel_{p}',
                    drawstyle='steps-mid'
                )

        ax.axhline(
            y=90,
            color='gray',
            linewidth=0.8,
            linestyle='--'
        )

        ax.axhline(
            y=45,
            color='gray',
            linewidth=0.8,
            linestyle='--'
        )

        ax.axhline(
            y=135,
            color='gray',
            linewidth=0.8,
            linestyle='--'
        )

        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title=instrument
        )

        ax.set_ylim([0, 180])

        ax.yaxis.set_ticks(
            np.arange(0, 180 + 45, 45)
        )

        ax.set_ylabel(
            'PA / °',
            size=f_size
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d-%m-%y\n%H:%M")
        )

        plt.tick_params(
            axis='x',
            which='major',
            labelsize=f_size
        )

        plt.tick_params(
            axis='y',
            which='major',
            labelsize=f_size
        )

        ax.set_xlabel(
            "Time",
            labelpad=45,
            size=f_size
        )

    # Save figure, if enabled.
    if path and path[-1] != '/':
        path = path + '/'

    if savefig:

        plt.savefig(
            path + filename + str(key) + '.jpg',
            bbox_inches='tight',
            dpi=300
        )

    plt.show()

def plot_spectrum_peak(
    args,
    species,
    bg_subtraction=True,
    savefig=False,
    path='',
    key='',
    sigma=3,
    frac_nan_threshold=0.4,
    rel_err_threshold=0.5,
    direction=None,
    centre_pix=False,
    date=None
):
    """
    Creates an energy spectrum plot using the peak flux values from each
    energy channel for electrons or protons.

    The plot can show either background-subtracted or raw peak intensities.
    Error bars include the corresponding flux uncertainty and the lower and
    upper energy-bin uncertainties. The background intensity is also shown
    for comparison.

    Energy channels that do not satisfy the specified data-quality criteria
    are marked separately according to the reason for exclusion:
        - grey: too many NaN values in the search interval
        - blue: peak significance below the sigma threshold
        - orange: relative error above the specified threshold

    Args:
        args (tuple):
            Output of the corresponding extract_data function. Contains:
                df_fluxes: pandas DataFrame containing particle fluxes.
                df_info: pandas DataFrame containing spectrum data and
                    metadata.
                [searchstart, searchend]: search-window start and end times.
                [e_low, e_high]: lower and upper energies for each energy
                    channel.
                [instrument, data_type]: instrument and data-product type.

        species (str):
            Particle species to plot. Accepted values are 'electron',
            'electrons', 'e', 'proton', 'protons', or 'p'.

        bg_subtraction (bool, optional):
            If True, plot background-subtracted peak intensities.
            If False, plot the raw peak intensities. Defaults to True.

        savefig (bool, optional):
            If True, save the generated figure. Defaults to False.

        path (str, optional):
            Path to the directory where the figure should be saved.
            Defaults to ''.

        key (str, optional):
            Optional string appended to the output filename. Defaults to ''.

        sigma (int, optional):
            Minimum peak-significance threshold used to identify significant
            peaks. Defaults to 3.

        frac_nan_threshold (float, optional):
            Minimum fraction of non-NaN data points required in the search
            interval. Channels below this threshold are marked as excluded.
            Defaults to 0.4.

        rel_err_threshold (float, optional):
            Maximum allowed relative uncertainty of the background-subtracted
            peak. Channels above this threshold are marked as excluded.
            Defaults to 0.5.

        direction (str or None, optional):
            Telescope viewing direction ('sun', 'asun', 'north', or 'south').
            For STEP, the direction is always set to 'sun'. If None is
            provided for EPT or HET, 'sun' is used. Defaults to None.

        centre_pix (bool, optional):
            Indicates whether centre-pixel STEP data are being used. This
            information is included in the plot title and filename.
            Defaults to False.

        date (str or pandas.Timestamp or None, optional):
            Date used in the plot title and filename. If None, the date is
            taken from the Plot_period entry in df_info. Defaults to None.

    Returns:
        None
            Displays the spectrum plot and optionally saves it to disk.
    """

    color = {
        'sun': 'crimson',
        'asun': 'orange',
        'north': 'darkslateblue',
        'south': 'c'
    }

    # Normalize species name.
    species_lower = species.lower()

    if species_lower in ['electron', 'electrons', 'e']:
        species = 'electron'
    elif species_lower in ['proton', 'protons', 'p']:
        species = 'proton'

    df_info = args[1]
    instrument = args[4][0]
    data_type = args[4][1]

    instrument_lower = instrument.lower()

    # Determine viewing direction.
    if direction is None or instrument_lower == 'step':
        direction = 'sun'

    viewing = '' if instrument_lower == 'step' else f'-{direction}'

    # Determine date strings for title and filename.
    if date is None:
        date_string = str(df_info['Plot_period'][0][:-5])
        file_date = date_string
    else:
        date_string = str(date)[:-3]
        file_date = date_string.replace(' ', '-').replace(':', '')

    title_string = (
        f"{instrument.upper()}, {data_type.upper()}, {date_string}"
    )

    filename = (
        f"{species}_spectrum-{file_date}-{instrument.upper()}"
        f"{viewing}-{data_type.upper()}"
    )

    # Add averaging information.
    if df_info['Averaging'][0] == 'Mean':
        averaging = df_info['Averaging'][1].split()[2]
        title_string += f", {averaging} averaging"
        filename += f"-{averaging}_averaging"

    elif df_info['Averaging'][0] == 'No averaging':
        title_string += ", no averaging"
        filename += "-no_averaging"

    # Add background-subtraction information.
    if bg_subtraction:
        title_string += ", bg subtraction on"
        filename += "-bg_subtr"
    else:
        title_string += ", bg subtraction off"

    # Add ion-contamination correction information for EPT electrons.
    if instrument_lower == 'ept' and species == 'electron':
        if df_info['Ion_contamination_correction'][0]:
            title_string += ", ion correction on"
            filename += "-ion_corr"
        else:
            title_string += ", ion correction off"

    # Add centre-pixel information for STEP.
    if instrument_lower == 'step' and centre_pix:
        filename += "-centre_pix"
        title_string += ", centre pix"

    # Identify channels excluded for different reasons.
    df_nan = df_info.where(
        df_info['frac_nonan'] < frac_nan_threshold,
        np.nan
    )

    df_no_sig = df_info.where(
        df_info['Peak_significance'] < sigma,
        np.nan
    )

    df_rel_err = df_info.where(
        df_info['rel_backsub_peak_err'] > rel_err_threshold,
        np.nan
    )

    # Plot either background-subtracted or raw peak fluxes.
    fig, ax = plt.subplots(figsize=(13, 10))

    if bg_subtraction:
        ax.errorbar(
            x=df_info['Primary_energy'],
            y=df_info['Bg_subtracted_peak'],
            yerr=df_info['Backsub_peak_uncertainty'],
            xerr=[
                df_info['Energy_error_low'],
                df_info['Energy_error_high']
            ],
            color=color[direction],
            fmt='o',
            ecolor=color[direction],
            zorder=0,
            label='Flux peaks'
        )

        ax.plot(
            df_nan.Primary_energy,
            df_nan.Bg_subtracted_peak,
            'o',
            markersize=15,
            c='gray',
            label='excluded (NaNs)'
        )

        ax.plot(
            df_no_sig.Primary_energy,
            df_no_sig.Bg_subtracted_peak,
            'o',
            markersize=11,
            c='blue',
            label='excluded (sigma)'
        )

        ax.plot(
            df_rel_err.Primary_energy,
            df_rel_err.Bg_subtracted_peak,
            'o',
            markersize=6,
            c='orange',
            label='excluded (rel error)'
        )

    else:
        ax.errorbar(
            x=df_info['Primary_energy'],
            y=df_info['Flux_peak'],
            yerr=df_info[f'Peak_{species}_uncertainty'],
            xerr=[
                df_info['Energy_error_low'],
                df_info['Energy_error_high']
            ],
            fmt='o',
            color=color[direction],
            ecolor=color[direction],
            zorder=0,
            label='Intensity peaks'
        )

        ax.plot(
            df_nan.Primary_energy,
            df_nan.Flux_peak,
            'o',
            markersize=15,
            c='gray',
            label='excluded (NaNs)'
        )

        ax.plot(
            df_no_sig.Primary_energy,
            df_no_sig.Flux_peak,
            'o',
            markersize=11,
            c='blue',
            label='excluded (sigma)'
        )

        ax.plot(
            df_rel_err.Primary_energy,
            df_rel_err.Flux_peak,
            'o',
            markersize=6,
            c='orange',
            label='excluded (rel error)'
        )

    # Plot background intensity and its uncertainty.
    ax.errorbar(
        x=df_info['Primary_energy'],
        y=df_info['Background_flux'],
        yerr=df_info[f'Bg_{species}_uncertainty'],
        xerr=[
            df_info['Energy_error_low'],
            df_info['Energy_error_high']
        ],
        fmt='o',
        color=color[direction],
        ecolor=color[direction],
        alpha=0.15,
        label='Background intensity'
    )

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel('Energy [MeV]', size=20)
    ax.set_ylabel('Intensity \n [1/s cm$^2$ sr MeV]', size=20)

    plt.tick_params(axis='x', which='minor', labelsize=16)

    ax.xaxis.set_minor_formatter(
        FormatStrFormatter("%.2f")
    )

    plt.legend(prop={'size': 18})
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.grid()
    plt.title(title_string)

    # Prevent every minor x-axis label from being displayed.
    for label in ax.xaxis.get_ticklabels(which='minor')[1::2]:
        label.set_visible(False)

    # Save figure if requested.
    if savefig:
        if path and not path.endswith('/'):
            path += '/'

        plt.savefig(
            path + filename + str(key) + '.jpg',
            dpi=300,
            bbox_inches='tight'
        )

    plt.show()

def plot_spectrum_average(
    args,
    species,
    bg_subtraction=True,
    savefig=False,
    path='',
    key='',
    sigma=3,
    frac_nan_threshold=0.4,
    rel_err_threshold=0.5,
    direction=None,
    centre_pix=False,
    date=None
):
    """
    Creates an energy spectrum plot using the average flux values from each
    energy channel for electrons or protons.

    The plot can show either background-subtracted or raw average intensities.
    Error bars include the corresponding flux uncertainty and the lower and
    upper energy-bin uncertainties. The background intensity is also shown
    for comparison.

    Energy channels that do not satisfy the specified data-quality criteria
    are marked separately according to the reason for exclusion:
        - grey: too many NaN values in the search interval
        - blue: average significance below the sigma threshold
        - orange: relative error above the specified threshold

    Args:
        args (tuple):
            Output of the corresponding extract_data function. Contains:
                df_fluxes: pandas DataFrame containing particle fluxes.
                df_info: pandas DataFrame containing spectrum data and
                    metadata.
                [searchstart, searchend]: search-window start and end times.
                [e_low, e_high]: lower and upper energies for each energy
                    channel.
                [instrument, data_type]: instrument and data-product type.

        species (str):
            Particle species to plot. Accepted values are 'electron',
            'electrons', 'e', 'proton', 'protons', or 'p'.

        bg_subtraction (bool, optional):
            If True, plot background-subtracted average intensities.
            If False, plot the raw average intensities. Defaults to True.

        savefig (bool, optional):
            If True, save the generated figure. Defaults to False.

        path (str, optional):
            Path to the directory where the figure should be saved.
            Defaults to ''.

        key (str, optional):
            Optional string appended to the output filename. Defaults to ''.

        sigma (int, optional):
            Minimum average-significance threshold used to identify
            significant channels. Defaults to 3.

        frac_nan_threshold (float, optional):
            Minimum fraction of non-NaN data points required in the search
            interval. Channels below this threshold are marked as excluded.
            Defaults to 0.4.

        rel_err_threshold (float, optional):
            Maximum allowed relative uncertainty of the background-subtracted
            average. Channels above this threshold are marked as excluded.
            Defaults to 0.5.

        direction (str or None, optional):
            Telescope viewing direction ('sun', 'asun', 'north', or 'south').
            For STEP, the direction is always set to 'sun'. If None is
            provided for EPT or HET, 'sun' is used. Defaults to None.

        centre_pix (bool, optional):
            Indicates whether centre-pixel STEP data are being used. This
            information is included in the plot title and filename.
            Defaults to False.

        date (str or pandas.Timestamp or None, optional):
            Date used in the plot title and filename. If None, the date is
            taken from the Plot_period entry in df_info. Defaults to None.

    Returns:
        None
            Displays the spectrum plot and optionally saves it to disk.
    """

    color = {
        'sun': 'crimson',
        'asun': 'orange',
        'north': 'darkslateblue',
        'south': 'c'
    }

    # Normalize species name.
    species_lower = species.lower()

    if species_lower in ['electron', 'electrons', 'e']:
        species = 'electron'
    elif species_lower in ['proton', 'protons', 'p']:
        species = 'proton'

    df_info = args[1]
    instrument = args[4][0]
    data_type = args[4][1]

    instrument_lower = instrument.lower()

    # Determine viewing direction.
    if direction is None or instrument_lower == 'step':
        direction = 'sun'

    viewing = '' if instrument_lower == 'step' else f'-{direction}'

    # Determine date strings for title and filename.
    if date is None:
        date_string = str(df_info['Plot_period'][0][:-5])
        file_date = date_string
    else:
        date_string = str(date)[:-3]
        file_date = date_string.replace(' ', '-').replace(':', '')

    title_string = (
        f"{instrument.upper()}, {data_type.upper()}, {date_string}"
    )

    filename = (
        f"{species}_spectrum-{file_date}-{instrument.upper()}"
        f"{viewing}-{data_type.upper()}"
    )

    # Add averaging information.
    if df_info['Averaging'][0] == 'Mean':
        averaging = df_info['Averaging'][1].split()[2]
        title_string += f", {averaging} averaging"
        filename += f"-{averaging}_averaging"

    elif df_info['Averaging'][0] == 'No averaging':
        title_string += ", no averaging"
        filename += "-no_averaging"

    # Add background-subtraction information.
    if bg_subtraction:
        title_string += ", bg subtraction on"
        filename += "-bg_subtr"
    else:
        title_string += ", bg subtraction off"

    # Add ion-contamination correction information for EPT electrons.
    if instrument_lower == 'ept' and species == 'electron':
        if df_info['Ion_contamination_correction'][0]:
            title_string += ", ion correction on"
            filename += "-ion_corr"
        else:
            title_string += ", ion correction off"

    # Add centre-pixel information for STEP.
    if instrument_lower == 'step' and centre_pix:
        filename += "-centre_pix"
        title_string += ", centre pix"

    # Identify channels excluded for different reasons.
    df_nan = df_info.where(
        df_info['frac_nonan'] < frac_nan_threshold,
        np.nan
    )

    df_no_sig = df_info.where(
        df_info['Average_significance'] < sigma,
        np.nan
    )

    df_rel_err = df_info.where(
        df_info['rel_backsub_peak_err'] > rel_err_threshold,
        np.nan
    )

    # Plot either background-subtracted or raw average fluxes.
    fig, ax = plt.subplots(figsize=(13, 10))

    if bg_subtraction:
        ax.errorbar(
            x=df_info['Primary_energy'],
            y=df_info['Bg_subtracted_average'],
            yerr=df_info['Backsub_peak_uncertainty'],
            xerr=[
                df_info['Energy_error_low'],
                df_info['Energy_error_high']
            ],
            color=color[direction],
            fmt='o',
            ecolor=color[direction],
            zorder=0,
            label='Intensity average'
        )

        ax.plot(
            df_nan.Primary_energy,
            df_nan.Bg_subtracted_average,
            'o',
            markersize=15,
            c='gray',
            label='excluded (NaNs)'
        )

        ax.plot(
            df_no_sig.Primary_energy,
            df_no_sig.Bg_subtracted_average,
            'o',
            markersize=11,
            c='blue',
            label='excluded (sigma)'
        )

        ax.plot(
            df_rel_err.Primary_energy,
            df_rel_err.Bg_subtracted_average,
            'o',
            markersize=6,
            c='orange',
            label='excluded (rel error)'
        )

    else:
        ax.errorbar(
            x=df_info['Primary_energy'],
            y=df_info['Flux_average'],
            yerr=df_info[f'Peak_{species}_uncertainty'],
            xerr=[
                df_info['Energy_error_low'],
                df_info['Energy_error_high']
            ],
            fmt='o',
            color=color[direction],
            ecolor=color[direction],
            zorder=0,
            label='Intensity average'
        )

        ax.plot(
            df_nan.Primary_energy,
            df_nan.Flux_average,
            'o',
            markersize=15,
            c='gray',
            label='excluded (NaNs)'
        )

        ax.plot(
            df_no_sig.Primary_energy,
            df_no_sig.Flux_average,
            'o',
            markersize=11,
            c='blue',
            label='excluded (sigma)'
        )

        ax.plot(
            df_rel_err.Primary_energy,
            df_rel_err.Flux_average,
            'o',
            markersize=6,
            c='orange',
            label='excluded (rel error)'
        )

    # Plot background intensity and its uncertainty.
    ax.errorbar(
        x=df_info['Primary_energy'],
        y=df_info['Background_flux'],
        yerr=df_info[f'Bg_{species}_uncertainty'],
        xerr=[
            df_info['Energy_error_low'],
            df_info['Energy_error_high']
        ],
        fmt='o',
        color=color[direction],
        ecolor=color[direction],
        alpha=0.15,
        label='Background intensity'
    )

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel('Energy [MeV]', size=20)
    ax.set_ylabel('Intensity \n [1/s cm$^2$ sr MeV]', size=20)

    plt.tick_params(axis='x', which='minor', labelsize=16)

    ax.xaxis.set_minor_formatter(
        FormatStrFormatter("%.2f")
    )

    plt.legend(prop={'size': 18})
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.grid()
    plt.title(title_string)

    # Prevent every minor x-axis label from being displayed.
    for label in ax.xaxis.get_ticklabels(which='minor')[1::2]:
        label.set_visible(False)

    # Save figure if requested.
    if savefig:
        if path and not path.endswith('/'):
            path += '/'

        plt.savefig(
            path + filename + str(key) + '.jpg',
            dpi=300,
            bbox_inches='tight'
        )

    plt.show()

def write_to_csv(
    args,
    date,
    species,
    path='',
    key='',
    direction=None,
    centre_pix=False
):
    """
    Saves the spectrum information dataframe to a CSV file.

    The output filename contains the particle species, date, instrument,
    viewing direction, data-product type, averaging information, and
    relevant instrument-specific processing information.

    Args:
        args (tuple):
            Output of the extract_data function. Contains:
                df_fluxes: pandas DataFrame containing particle fluxes.
                df_info: pandas DataFrame containing spectrum data and
                    metadata.
                [searchstart, searchend]: search-window start and end times.
                [e_low, e_high]: lower and upper energies for each energy
                    channel.
                [instrument, data_type]: instrument and data-product type.

        date (str):
            Date used in the output filename.

        species (str):
            Particle species to save. Accepted values are 'electron',
            'electrons', 'e', 'proton', 'protons', or 'p'.

        path (str, optional):
            Path to the directory where the CSV file should be saved.
            Defaults to ''.

        key (str, optional):
            Optional string appended to the output filename. Defaults to ''.

        direction (str or None, optional):
            Telescope viewing direction. If None, 'sun' is used.
            Defaults to None.

        centre_pix (bool, optional):
            Indicates whether centre-pixel STEP data are being used.
            Defaults to False.

    Returns:
        None
            Saves the df_info dataframe as a semicolon-separated CSV file.
    """

    df_info = args[1]
    instrument = args[4][0]
    data_type = args[4][1]

    instrument_lower = instrument.lower()

    # Normalize species name.
    species_lower = species.lower()

    if species_lower in ['electron', 'electrons', 'e']:
        species = 'electron'
    elif species_lower in ['proton', 'protons', 'p']:
        species = 'proton'

    # Use sun as the default viewing direction.
    viewing = 'sun' if direction is None else direction

    filename = (
        f'{species}_data-{date}-{instrument.upper()}-'
        f'{viewing}-{data_type.upper()}'
    )

    # Add averaging information.
    if df_info['Averaging'][0] == 'Mean':
        averaging = df_info['Averaging'][1].split()[2]
        filename += f'-{averaging}_averaging'

    elif df_info['Averaging'][0] == 'No averaging':
        filename += '-no_averaging'

    # Add ion-contamination correction information for EPT electrons.
    if instrument_lower == 'ept' and species == 'electron':
        if df_info['Ion_contamination_correction'][0]:
            filename += '-ion_corr'

    # Add centre-pixel information for STEP.
    if instrument_lower == 'step' and centre_pix:
        filename += '-centre_pix'

    # Save dataframe.
    if path and not path.endswith('/'):
        path += '/'

    df_info.to_csv(
        path + filename + str(key) + '.csv',
        sep=';',
        index=False
    )

