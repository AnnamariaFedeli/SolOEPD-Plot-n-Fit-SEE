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

# Add folder for data and one for plots
def create_new_path(path,date, threshold_folders=False, contamination_threshold=None, plots_n_data=True,):
    """
    Create the directory structure for a given event.

    Parameters
    ----------
    path : str
        Base path where the new event folder will be created.

    date : str
        Event date used as the event-folder name. For this software,
        the expected format is ``'YYYY-mm-dd-hhMM'``.

    threshold_folders : bool, optional
        If True, create an additional folder named according to the
        contamination threshold. This option is intended for comparing
        results obtained with different thresholds.

    contamination_threshold : int or float or None, optional
        Contamination threshold included in the folder name when
        ``threshold_folders`` is True.

    plots_n_data : bool, optional
        If True, create separate ``plots`` and ``data`` subdirectories
        inside the event (or threshold) folder.
    """

    newpath = path + date

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    print("Creating new directory " + newpath)

    if threshold_folders:
        newpath = (newpath + "/contamination_threshold_" + str(contamination_threshold))

        if not os.path.exists(newpath):
            os.makedirs(newpath)
            print("Creating new directory  " + newpath)

    if plots_n_data:
        plots_path = newpath + "/plots"
        data_path = newpath + "/data"

        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
            print("Creating new directory  " + plots_path)

        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print("Creating new directory  " + data_path)

                        

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """
    Return the angle in radians between two vectors.

    Examples
    --------
    angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    angle_between((1, 0, 0), (1, 0, 0))
    0.0
    angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """

    v1_unit = unit_vector(v1)
    v2_unit = unit_vector(v2)

    dot_product = np.dot(v1_unit, v2_unit)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))


def calc_pa_coverage(instrument, mag_data):
    """
    Calculate pitch-angle coverage for an instrument.

    Parameters
    ----------
    instrument : str
        Instrument name. Supported values are ``'EPT'``, ``'HET'``,
        and ``'STEP'`` (case-insensitive).

    mag_data : pandas.DataFrame
        Magnetic-field data containing ``Bx``, ``By``, and ``Bz`` columns.

    Returns
    -------
    pandas.DataFrame
        Pitch-angle coverage with ``min``, ``center``, and ``max`` values
        for each instrument direction or STEP pixel.
    """

    print(f'Calculating PA coverage for {instrument}...')

    valid_instruments = ['ept', 'het', 'step']

    if instrument.lower() not in valid_instruments:
        print("instrument not known, select 'EPT', 'HET', or 'STEP' ")
        return pd.DataFrame(mag_data.index)

    instrument = instrument.lower()

    # Instrument opening angle.
    if instrument == 'ept':
        opening = 30
    elif instrument == 'het':
        opening = 43
    else:
        print(
            "Opening of STEP just a placeholder! Replace with real value! "
            "This affects the 'min' and 'max' values of the pitch-angle, "
            "not the 'center' ones.")
        opening = 10

    mag_vec = np.array([mag_data.Bx.values, mag_data.By.values, mag_data.Bz.values,])

    # EPT / HET
    # ----------
    if instrument in ['ept', 'het']:
        # Pointing directions in XYZ/SRF coordinates.
        # Arrows point into the sensor.
        pointing_sun = np.array([-0.81915206, 0.57357645, 0.])
        pointing_asun = np.array([0.81915206, -0.57357645, 0.])
        pointing_north = np.array([0.30301532, 0.47649285, -0.8253098])
        pointing_south = np.array([-0.30301532, -0.47649285, 0.8253098])

        pa_sun = np.ones(len(mag_data.Bx.values)) * np.nan
        pa_asun = np.ones(len(mag_data.Bx.values)) * np.nan
        pa_north = np.ones(len(mag_data.Bx.values)) * np.nan
        pa_south = np.ones(len(mag_data.Bx.values)) * np.nan

        for i in tqdm(range(len(mag_data.Bx.values))):
            pa_sun[i] = np.rad2deg(angle_between(pointing_sun, mag_vec[:, i]))
            pa_asun[i] = np.rad2deg(angle_between(pointing_asun, mag_vec[:, i]))
            pa_north[i] = np.rad2deg(angle_between(pointing_north, mag_vec[:, i]))
            pa_south[i] = np.rad2deg(angle_between(pointing_south, mag_vec[:, i]))

        sun_min = pa_sun - opening / 2
        sun_max = pa_sun + opening / 2
        asun_min = pa_asun - opening / 2
        asun_max = pa_asun + opening / 2
        north_min = pa_north - opening / 2
        north_max = pa_north + opening / 2
        south_min = pa_south - opening / 2
        south_max = pa_south + opening / 2

        cov_sun = pd.DataFrame({'min': sun_min, 'center': pa_sun, 'max': sun_max}, index=mag_data.index,)
        cov_asun = pd.DataFrame({'min': asun_min, 'center': pa_asun, 'max': asun_max}, index=mag_data.index,)
        cov_north = pd.DataFrame({'min': north_min, 'center': pa_north, 'max': north_max}, index=mag_data.index,)
        cov_south = pd.DataFrame( {'min': south_min, 'center': pa_south, 'max': south_max}, index=mag_data.index,)

        keys = ['sun', 'asun', 'north', 'south']
        coverage = pd.concat([cov_sun, cov_asun, cov_north, cov_south], keys=keys, axis=1, )

    # STEP
    # ------------------------------------------------------------------
    elif instrument == 'step':
        # Particle flow direction (unit vector) in spacecraft XYZ
        # coordinates for each STEP pixel ('XYZ_Pixels').
        pointing_step = np.array([
            [-0.8412, 0.4396,  0.3149],
            [-0.8743, 0.4570,  0.1635],
            [-0.8862, 0.4632, -0.0000],
            [-0.8743, 0.4570, -0.1635],
            [-0.8412, 0.4396, -0.3150],
            [-0.7775, 0.5444,  0.3149],
            [-0.8082, 0.5658,  0.1635],
            [-0.8191, 0.5736,  0.0000],
            [-0.8082, 0.5659, -0.1634],
            [-0.7775, 0.5444, -0.3149],
            [-0.7008, 0.6401,  0.3149],
            [-0.7284, 0.6653,  0.1634],
            [-0.7384, 0.6744, -0.0000],
            [-0.7285, 0.6653, -0.1635],
            [-0.7008, 0.6401, -0.3150],
        ])

        pa_step = (np.ones((len(mag_data.Bx.values), pointing_step.shape[0])) * np.nan)

        for i in tqdm(range(len(mag_data.Bx.values))):
            for j in range(pointing_step.shape[0]):
                pa_step[i, j] = np.rad2deg(angle_between(pointing_step[j], mag_vec[:, i]))

        pa_step_min = pa_step - opening / 2
        pa_step_max = pa_step + opening / 2

        cov = {}

        for i in range(pa_step.shape[1]):
            cov[f'Pixel_{i + 1}'] = pd.DataFrame({
                    'min': pa_step_min[:, i],
                    'center': pa_step[:, i],
                    'max': pa_step_max[:, i],}, index=mag_data.index,)

        coverage = pd.concat(cov, keys=cov.keys(), axis=1)

    # Preserve the original final clipping step.
    coverage[coverage > 180] = 180
    coverage[coverage < 0] = 0

    return coverage


def solo_mag_loader(sdate, edate, level='l2', type='normal', frame='rtn', av=None, path=None,):
    """
    Load Solar Orbiter/MAG data and optionally average it.
    Load SolO/MAG data from SOAR using the ``mag_load()`` function from Jan.

    ``mag_load()`` autodownloads the data files if they are not already
    available.
    
    TODO
    ----
    Implement higher-resolution averaging (``'1S'`` / seconds) for burst data.

        Parameters
    ----------
    sdate : int
        Start date, e.g. ``20210417``.

    edate : int
        End date, e.g. ``20210418``.

    level : str, optional
        MAG data level, by default ``'l2'``.

    type : str, optional
        MAG data type, e.g. ``'normal'`` or ``'burst'``,
        by default ``'normal'``.

    frame : str, optional
        Coordinate frame, by default ``'rtn'``.
        Supported frames here are ``'rtn'`` and ``'srf'``.

    av : str or None, optional
        Pandas resampling frequency used for averaging.
        For example, ``'10min'``. If ``None``, no averaging
        is performed.

    path : str or None, optional
        Path passed to ``mag_load()``.

    Returns
    -------
    pandas.DataFrame
        Loaded MAG data, optionally resampled and averaged.
    """

    print('Loading MAG...')

    mag_data = mag_load(sdate, edate, level=level, data_type=type, frame=frame, path=path,)

    # Rename magnetic-field components according to the selected frame.
    if frame == 'rtn':
        mag_data.rename(columns={'B_RTN_0': 'B_r', 'B_RTN_1': 'B_t', 'B_RTN_2': 'B_n',}, inplace=True,)
    elif frame == 'srf':
        mag_data.rename(columns={'B_SRF_0': 'Bx', 'B_SRF_1': 'By', 'B_SRF_2': 'Bz',}, inplace=True,)

    # Average the data if a resampling interval was provided.
    if av is not None:
        mav = av

        m_int = int(re.search(r'\d+', av).group()) / 2
        m_string = ''.join(i for i in av if not i.isdigit())
        mag_offset = str(m_int) + m_string

        mag_data = mag_data.resample(mav, label='left').mean()
        mag_data.index = mag_data.index + to_offset(mag_offset)

    return mag_data


def evolt2beta(ekin, which):
    """
    Calculate the plasma beta for particles with a given kinetic energy.

    Parameters
    ----------
    ekin : float
        Particle kinetic energy in MeV.

    which : int
        Particle type: ``1`` for protons, ``2`` for electrons.

    Returns
    -------
    float
        Particle beta (v/c).
    """

    c = 299792458.0       # m/s, speed of light
    me0 = 9.109e-31        # kg, mass of electron
    mp0 = 1.67e-27         # kg, mass of proton
    q = 1.60217646e-19     # C, charge

    # Convert kinetic energy from MeV to Joules.
    ekin = ekin * 1.0e6 * q

    betae = np.sqrt(
        1 - (me0 * c**2 / (ekin + me0 * c**2))**2
    )
    betap = np.sqrt(
        1 - (mp0 * c**2 / (ekin + mp0 * c**2))**2
    )

    if which == 1:
        return betap

    if which == 2:
        return betae

def evolt2speed(ekin, which):
    """
    Calculate the velocity of a particle with a given kinetic energy.

    Parameters
    ----------
    ekin : float
        Particle kinetic energy in MeV.

    which : int
        Particle type: ``1`` for protons, ``2`` for electrons.

    Returns
    -------
    float
        Particle velocity in km/s.
    """

    c = 299792458.0  # m/s, speed of light

    beta = evolt2beta(ekin, which)

    velocity = beta * c
    velocity = velocity / 1000.0  # Convert from m/s to km/s.

    return velocity

#searchstart, searchend,

def len_of_spiral(vsw, dist):
    """
    Calculate the length of a Parker spiral.

    Parameters
    ----------
    vsw : float
        Solar-wind speed in km/s.

    dist : float
        Spacecraft distance from the Sun in AU.

    Returns
    -------
    float
        Length of the Parker spiral in AU.
    """

    # Solar rotation rate in radians per second.
    omega = np.deg2rad(360.0 / (25.38 * 24.0 * 60.0 * 60.0))

    AU = 1 * u.au

    # Spacecraft distance in km.
    r = AU.to(u.km).value * dist

    # Solar radius in km.
    solar_radius = 695700.0

    # Length of the spiral in km.
    R_s = (
        0.5 * omega / vsw
        * (r - solar_radius)
        * np.sqrt((r - solar_radius)**2 + (vsw / omega)**2)
        + 0.5 * vsw / omega
        * asinh((r - solar_radius) / vsw * omega)
    )

    # Convert spiral length from km to AU.
    new_R_s = R_s / AU.to(u.km).value

    return new_R_s


def traveltime_los(los, energy, which):
    """
    Calculate particle travel time along a length-of-spiral path.

    Parameters
    ----------
    los : float
        Length of the spiral in AU. The distance from the Sun to the
        spacecraft is already included in this value.

    energy : float
        Particle kinetic energy in MeV.

    which : int
        Particle type: ``1`` for protons, ``2`` for electrons.

    Returns
    -------
    float
        Travel time in seconds.
    """

    velocity = evolt2speed(energy, which)

    # Convert the length of the spiral from AU to km.
    R_s = los * 149597870.691

    # Travel time in seconds.
    travel_time = R_s / velocity

    return travel_time


def light_tt(dist):
    """
    Calculate the light travel time for a given distance.

    Parameters
    ----------
    dist : float
        Distance in AU.

    Returns
    -------
    float
        Light travel time in seconds.
    """

    # Speed of light in m/s.
    c = 299792458.0

    # Astronomical unit in meters.
    au2m = 149597870691.0

    distance = dist * au2m
    travel_time = distance / c

    return travel_time

def format_traveltime(seconds):
    """
    Format a travel time given in seconds as hours, minutes, and seconds.
    This function obviously works also to just change seconds to h, m, s.

    
    Parameters
    ----------
    seconds : float
        Travel time in seconds.

    Returns
    -------
    str
        Formatted travel time, e.g. ``'2 h 15 min 34.5 s'``.
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    return f"{hours} h {minutes} min {remaining_seconds:.1f} s"

def position_and_traveltime(date, species):
    """
    Calculate the Solar Orbiter distance, Parker spiral length, and
    particle/light travel times for a given date and particle species.

    Parameters
    ----------
    date : str
        Date used to obtain the Solar Orbiter position from Horizons.

    species : str
        Particle species. Accepted values are ``'electron'``,
        ``'electrons'``, ``'e'``, ``'proton'``, ``'protons'``, and ``'p'``.

    Returns
    -------
    list
        Table data containing the Solar Orbiter distance, Parker spiral
        length, particle travel times, and light travel time.
    """

    species = species.lower()

    if species in ['electron', 'electrons', 'e']:
        which = 2
    elif species in ['proton', 'protons', 'p']:
        which = 1

    # aug26 updated get coord
    pos = get_horizons_coord('Solar Orbiter', date)

    dist = np.round(pos.radius.value, 2)

    spiral_len = len_of_spiral(400, dist)

    traveltime_min = traveltime_los(spiral_len, 0.004, which)
    traveltime_max = traveltime_los(spiral_len, 10, which)

    if species in ['proton', 'protons', 'p']:
        traveltime_100mev = traveltime_los(spiral_len, 100, which)

    light_t = light_tt(dist)
    traveltime_min = format_traveltime(traveltime_min)
    traveltime_max = format_traveltime(traveltime_max)
    light_t = format_traveltime(light_t)

    if species in ['proton', 'protons', 'p']:
        traveltime_100mev = format_traveltime(traveltime_100mev)


    if species in ['electron', 'electrons', 'e']:
        table_data = [["Distance of SolO from the Sun", "[AU]", dist],
            ["Length of the Parker Spiral for 400 km/s sw ", "[AU]", spiral_len],
            ["Travel time of 4 keV electrons ", "", traveltime_min],
            ["Travel time of 10 MeV electrons ", "", traveltime_max],
            ["Travel time of light ", "", light_t],]

    elif species in ['proton', 'protons', 'p']:
        table_data = [["Distance of SolO from the Sun", "[AU]", dist],
            ["Length of the Parker Spiral for 400 km/s sw ", "[AU]", spiral_len],
            ["Travel time of 4 keV protons ", "", traveltime_min],
            ["Travel time of 10 MeV protons ", "", traveltime_max],
            ["Travel time of 100 MeV protons ", "", traveltime_100mev],
            ["Travel time of light ", "", light_t],]
    print(tabulate(table_data))

    return table_data


def extract_electron_data(df_electrons, df_energies, plotstart, plotend, t_inj,
        bgstart=None, bgend=None, bg_distance_from_window='2h', bg_period='60min', travel_distance=0,
        travel_distance_second_slope=None, fixed_window=None, instrument='ept', data_type='l2',
        averaging=None, masking=True, ion_conta_corr=False, df_protons=None, centre_pix=False ):
    """
    Extract electron fluxes and determine energy-dependent peak information.

    The function determines an energy spectrum from time-series data for the
    Solar Orbiter / EPD instruments. Energy-dependent search windows are used
    to determine the flux values for each energy channel. The search-window
    start time is determined from an expected velocity dispersion based on
    the solar injection time (`t_inj`) and a specified travel distance.

    A background can either be defined using a fixed time interval or using
    a moving interval whose position is determined relative to the
    energy-dependent search window.

    For EPT data, ion contamination can optionally be corrected using proton
    fluxes. For STEP data, intervals affected by significant ion
    contamination can optionally be masked.

    Parameters
    ----------
    df_electrons : pandas.DataFrame
        Electron data containing the fluxes and uncertainties.

    df_energies : pandas.DataFrame
        DataFrame containing the energy-channel information for the selected
        instrument and data product.

    plotstart : str
        Start time of the time interval to analyze, e.g.
        ``'2020-11-18-0000'``.

    plotend : str
        End time of the time interval to analyze, e.g.
        ``'2020-11-18-2230'``.

    t_inj : str
        Solar injection time, e.g. ``'2020-11-18-1230'``.

    bgstart : str, optional
        Start time of a fixed background interval. If specified, `bgend`
        must also be specified. Do not specify this together with
        `bg_distance_from_window` or `bg_period`.

    bgend : str, optional
        End time of a fixed background interval. If specified, `bgstart`
        must also be specified. Do not specify this together with
        `bg_distance_from_window` or `bg_period`.

    bg_distance_from_window : str, optional
        Time between the end of the background interval and the start of
        the energy-dependent search window. The value is interpreted as a
        pandas time offset, e.g. ``'2h'``. Must be specified together with
        `bg_period` when using a moving background.

    bg_period : str, optional
        Duration of the background interval. The value is interpreted as a
        pandas time offset, e.g. ``'60min'``. Must be specified together
        with `bg_distance_from_window` when using a moving background.

    travel_distance : float, optional
        Travel distance in AU used to determine the start of the
        energy-dependent search window. Defaults to 0.

    travel_distance_second_slope : float, optional
        Travel distance in AU used to determine the end of the
        energy-dependent search window. If specified, the search-window
        end is calculated using a second velocity-dispersion slope.
        If `None`, `fixed_window` must be specified.

    fixed_window : str, optional
        Duration of the search window, interpreted as a pandas time offset,
        e.g. ``'30min'``. If specified, the search-window end is determined
        by adding this duration to the energy-dependent search-window start.

        Use either `travel_distance_second_slope` or `fixed_window` to
        determine the search-window end.

    instrument : str, optional
        EPD instrument to use. Supported values are ``'ept'``, ``'het'``,
        and ``'step'``. Defaults to ``'ept'``.

    data_type : str, optional
        Data product level/type used for the electron data, e.g. ``'ll'``
        or ``'l2'``. Defaults to ``'l2'``.

    averaging : str, optional
        Pandas resampling frequency used to average the data before the
        peak and background calculations. If `None`, no averaging is
        performed. Defaults to `None`.

    masking : bool, optional
        Refers to STEP data. If `True`, time intervals with significant
        ion contamination are masked. Defaults to `True`.

    ion_conta_corr : bool, optional
        Refers to EPT data. If `True`, ion contamination is corrected using
        the proton flux data supplied in `df_protons`. Defaults to `False`.

    df_protons : pandas.DataFrame, optional
        Proton/ion data containing the fluxes and uncertainties required
        for the EPT ion-contamination correction. Defaults to `None`.

    centre_pix : bool, optional
        Refers to STEP data. If `True`, the STEP center-pixel electron flux
        and uncertainty data are used. Defaults to `False`.

    Raises
    ------
    Exception
        If a fixed background interval is specified together with a moving
        background definition, or if only one of the two required
        parameters for either background definition is specified.

    Returns
    -------
    df_electron_fluxes : pandas.DataFrame
        Electron fluxes for the selected instrument and energy channels
        over the requested plot interval.

    df_info : pandas.DataFrame
        DataFrame containing the energy-channel information, background and
        search-window intervals, peak and average fluxes, uncertainties,
        significances, and related metadata.

    search_periods : list of list
        A two-element list containing the search-window start times and
        search-window end times for each energy channel.

    energy_ranges : list of list
        A two-element list containing the lower and upper energy boundaries
        for each energy channel.

    instrument_info : list of str
        Two-element list containing the instrument and data type.
    """

    if bgstart is not None or bgend is not None:
        if bg_distance_from_window is not None or bg_period is not None:
            raise Exception("Please specify either bg_start and bg_end or bg_distance_from_window and bg_period.")

    if bgstart is None or bgend is None:
        if bg_distance_from_window is None or bg_period is None:
            raise Exception("Please specify either bg_start and bg_end or bg_distance_from_window and bg_period.")

    # Take proton and electron flux and uncertainty values from the original data.
    if instrument != 'step':
        df_electron_fluxes = df_electrons['Electron_Flux'][plotstart:plotend]
        df_electron_uncertainties = df_electrons['Electron_Uncertainty'][plotstart:plotend]

    if instrument == 'ept':
        df_proton_fluxes = df_protons['Ion_Flux'][plotstart:plotend]
        df_proton_uncertainties = df_protons['Ion_Uncertainty'][plotstart:plotend]

    
    if instrument in ['ept', 'het']:
        if data_type == 'll':
            channels = range(len(df_energies['Electron_Bins_Low_Energy']))
            e_low = df_energies['Electron_Bins_Low_Energy']
            e_high = []

            for i in channels:
                e_high.append(e_low[i] + df_energies['Electron_Bins_Width'][i])
                df_electron_fluxes = df_electron_fluxes.rename(columns={f'Ele_Flux_{i}': f'Electron_Flux_{i}'})
                df_electron_uncertainties = df_electron_uncertainties.rename(columns={f'Ele_Flux_Sigma_{i}': f'Electron_Uncertainty_{i}'})

        elif data_type == 'l2':
            channels = range(len(df_energies['Electron_Bins_Low_Energy']))
            e_low = df_energies['Electron_Bins_Low_Energy']
            e_high = []

            for i in channels:
                e_high.append(e_low[i] + df_energies['Electron_Bins_Width'][i])

    elif instrument == 'step':
        # STEP data changed in October 2021.
        # For center-pixel data, use sector energy information when available.
        energy_prefix = ''

        if centre_pix:
            if 'Electron_Sectors_Bins_Text' in df_energies.keys():
                energy_prefix = 'Electron_Sectors_'
                print('Sectors!!')
            else:
                raise ValueError(
                    'Centre-pixel electron STEP energy information '
                    'could not be found.')

        else:
            if 'Electron_Bins_Text' in df_energies.keys():
                energy_prefix = 'Electron_'
                print('E')

            elif 'Electron_Avg_Bins_Text' in df_energies.keys():
                # Pre-October-2021 electron data.
                energy_prefix = 'Electron_Avg_'

            else:
                raise ValueError(
                    'Could not identify the electron STEP energy '
                    'information.')

        if data_type == 'l2':
            e_low = df_energies[energy_prefix + 'Bins_Low_Energy']
            e_high = []

            channels = range(len(df_energies[energy_prefix + 'Bins_Low_Energy']))

            df_electron_fluxes = pd.DataFrame()
            df_electron_uncertainties = pd.DataFrame()

            for i in channels:
                e_high.append(e_low[i]+ df_energies[energy_prefix + 'Bins_Width'][i])

                if centre_pix:
                    df_electron_fluxes[f'Electron_Flux_{i}'] = (df_electrons[f'Electron_Comb_Flux_{i}'][plotstart:plotend])
                    df_electron_uncertainties[f'Electron_Uncertainty_{i}'] = (df_electrons[f'Electron_Comb_Uncertainty_{i}'][plotstart:plotend])

                else:
                    df_electron_fluxes[f'Electron_Flux_{i}'] = (df_electrons[f'Electron_Avg_Flux_{i}'][plotstart:plotend])
                    df_electron_uncertainties[f'Electron_Uncertainty_{i}'] = (df_electrons[f'Electron_Avg_Uncertainty_{i}'][plotstart:plotend])

        # Cleans up negative flux values in STEP data.
        df_electron_fluxes[df_electron_fluxes < 0] = np.nan


       
    if averaging is not None:
        if instrument == 'ept':
            df_proton_fluxes = df_proton_fluxes.resample(averaging).mean()
            df_proton_uncertainties = (df_proton_uncertainties.resample(averaging).apply(average_flux_error))

        # For STEP electrons, resampling is done independently,
        # e.g. solo_epd_loader.calc_electrons(df, resample='1min').
        if instrument != 'step':
            df_electron_fluxes = df_electron_fluxes.resample(averaging).mean()
            df_electron_uncertainties = (df_electron_uncertainties.resample(averaging).apply(average_flux_error))
            

    if ion_conta_corr and instrument == 'ept':
        ion_cont_corr_matrix = np.loadtxt('EPT_ion_contamination_flux_paco.dat')

        Electron_Flux_cont = np.zeros(np.shape(df_electron_fluxes))
        Electron_Uncertainty_cont = np.zeros(np.shape(df_electron_uncertainties))

        for tt in range(len(df_electron_fluxes)):
            Electron_Flux_cont[tt, :] = np.sum(ion_cont_corr_matrix* np.ma.masked_invalid(df_proton_fluxes.values[tt, :]),axis=1)

            # Matrix multiplication does not work with NaN values because
            # np.matmul has no built-in option to ignore them. Using
            # masked_invalid() ignores both NaN and infinite values.
            Electron_Uncertainty_cont[tt, :] = np.sqrt(np.sum(ion_cont_corr_matrix**2* np.ma.masked_invalid(df_proton_uncertainties.values[tt, :]**2),axis=1))

        df_electron_fluxes = df_electron_fluxes - Electron_Flux_cont
        df_electron_uncertainties = np.sqrt(df_electron_uncertainties**2+ Electron_Uncertainty_cont**2)
        
       
    if instrument == 'ept':
        ion_string = 'Ion_contamination_correction'
    elif instrument == 'step':
        ion_string = 'Ion_masking'
    elif instrument == 'het':
        ion_string = ''

    # Main information dataframe containing most of the required data.
    df_info = pd.DataFrame({'Plot_period': [], 'Averaging': [], '{}'.format(ion_string): [], 'Energy_channel': [],
            'Primary_energy': []})

    # Add basic metadata to the main info DataFrame.
    df_info['Plot_period'] = ([plotstart] + [plotend] + [''] * (len(channels) - 2))

    if instrument == 'ept':
        df_info['Ion_contamination_correction'] = ([ion_conta_corr] + [''] * (len(channels) - 1))

    elif instrument == 'step':
        df_info['Ion_masking'] = ([masking] + [''] * (len(channels) - 1))

    if averaging is None:
        df_info['Averaging'] = (['No averaging'] + [''] * (len(channels) - 1))

    elif averaging is not None:
        df_info['Averaging'] = (['Mean', 'Resampled to ' + averaging] + [''] * (len(channels) - 2))
 
    # Energy bin primary energies; geometric mean.
    # These are used to calculate particle beta and velocity.
    primary_energies = []

    for i in range(len(e_low)):
        primary_energies.append(np.sqrt(e_low[i] * e_high[i]))

    primary_energies_channels = [primary_energies[i] for i in channels]

    df_info['Primary_energy'] = primary_energies_channels

    # Calculate energy errors for the spectrum plot.
    energy_error_low = []
    energy_error_high = []

    for i in range(len(primary_energies)):
        energy_error_low.append(primary_energies[i] - e_low[i])
        energy_error_high.append(e_high[i] - primary_energies[i])

    energy_error_low_channels = [energy_error_low[i] for i in channels]
    energy_error_high_channels = [energy_error_high[i] for i in channels]

    
    df_info['Energy_error_low'] = energy_error_low_channels
    df_info['Energy_error_high'] = energy_error_high_channels

    # Calculate particle velocity from the primary energy.
    # The velocity is in km/s.
    velocity = []

    for energy in primary_energies:
        velocity.append(evolt2speed(energy, 2))

    # Calculate the search period using the velocity dispersion.
    # Convert travel distance from AU to km.
    travel_distance = travel_distance * 1.496E8

    DV = []

    for v in velocity:
        DV.append(travel_distance / v)

    searchstart = []

    for i in DV:
        searchstart.append(pd.to_datetime(t_inj) + pd.Timedelta(seconds=i))

    searchend = []

    # Calculate search end time using the second slope if no fixed
    # search window is specified.
    if fixed_window is None:
        travel_distance_second_slope = travel_distance_second_slope * 1.496E8

        DV2 = []

        for v in velocity:
            DV2.append(travel_distance_second_slope / v)

        for i in DV2:
            searchend.append(pd.to_datetime(t_inj) + pd.Timedelta(seconds=i))

    if fixed_window is not None:
        for i in searchstart:
            searchend.append(i + pd.to_timedelta(fixed_window))

    if bg_distance_from_window is None:
        bg_start = bgstart
        bg_end = bgend

        bgstart = []
        bgend = []

        for i in range(len(searchstart)):
            bgstart.append(bg_start)
            bgend.append(bg_end)

    if bg_distance_from_window is not None:
        bgstart = []
        bgend = []

        for i in range(len(searchstart)):
            bgend.append(searchstart[i] - pd.to_timedelta(bg_distance_from_window))
            bgstart.append(bgend[i] - pd.to_timedelta(bg_period))


    # Calculate information from the data and append it to the main info DataFrame.
    list_bg_fluxes = []
    list_flux_peaks = []
    list_peak_timestamps = []
    list_bg_subtracted_peaks = []
    list_peak_electron_uncertainties = []
    list_average_bg_uncertainties = []
    list_bg_std = []
    list_peak_significance = []
    list_flux_average = []
    list_bg_subtracted_average = []
    list_average_significance = []
    list_frac_nonan = []

    n = 0

    for channel in channels:
        electron_flux = df_electron_fluxes[f'Electron_Flux_{channel}']
        electron_uncertainty = (df_electron_uncertainties[f'Electron_Uncertainty_{channel}'])

        # Background flux.
        b_f = electron_flux[searchstart[n]:searchend[n]]

        # Check if the search window is empty.
        if len(b_f) == 0:
            bg_flux = np.nan
        else:
            bg_flux = electron_flux[bgstart[n]:bgend[n]].mean(skipna=True)

        list_bg_fluxes.append(bg_flux)

        # Peak flux within the search window.
        f_p = electron_flux[searchstart[n]:searchend[n]]

        if f_p.notna().any():
            flux_peak = f_p.max()
        else:
            flux_peak = np.nan

        list_flux_peaks.append(flux_peak)

        # Fraction of non-NaN data points in the search window.
        # This can be used to exclude channels with too much missing data.
        if len(f_p) == 0:
            frac_nonan = np.nan
        else:
            frac_nonan = f_p.notna().mean()

        list_frac_nonan.append(frac_nonan)

        # Timestamp of the peak flux.
        peak_timestamp = (f_p.idxmax() if f_p.notna().any() else np.nan)
        list_peak_timestamps.append(peak_timestamp)

        # Electron uncertainty at the peak timestamp.
        # Find the nearest timestamp in the uncertainty DataFrame.
        if pd.isna(peak_timestamp):
            list_peak_electron_uncertainties.append(np.nan)

        if len(electron_uncertainty) == 0:
            list_peak_electron_uncertainties.append(np.nan)

        if len(electron_uncertainty) != 0 and pd.isna(peak_timestamp) == False:
            timestamp_loc = electron_uncertainty.index.get_indexer([peak_timestamp],method='nearest')[0]

            peak_electron_uncertainty = electron_uncertainty.iloc[timestamp_loc]
            list_peak_electron_uncertainties.append(peak_electron_uncertainty)

        # Average uncertainty in the background window.
        bg_uncertainty = electron_uncertainty[bgstart[n]:bgend[n]]
        valid_uncertainties = bg_uncertainty.dropna()

        if len(valid_uncertainties) == 0:
            average_bg_uncertainty = np.nan
        else:
            average_bg_uncertainty = (np.sqrt((valid_uncertainties**2).sum()) / len(valid_uncertainties))

        list_average_bg_uncertainties.append(average_bg_uncertainty)

        # Standard deviation of the background flux.
        bg_std = electron_flux[bgstart[n]:bgend[n]].std()
        list_bg_std.append(bg_std)

        # Average flux within the search window.
        f_a = electron_flux[searchstart[n]:searchend[n]]

        if len(f_a) == 0:
            flux_average = np.nan
        else:
            flux_average = f_a.mean(skipna=True)

        list_flux_average.append(flux_average)

        n += 1

    # Calculate background-subtracted values and their significance.
    for i in range(len(list_flux_peaks)):
        list_bg_subtracted_peaks.append(list_flux_peaks[i] - list_bg_fluxes[i])

        list_peak_significance.append(list_bg_subtracted_peaks[i] / list_bg_std[i])

        # If the background is higher than the peak, mark the significance
        # as -1 so that the value can be excluded later.
        if list_bg_subtracted_peaks[i] < list_bg_fluxes[i]:
            list_peak_significance[i] = -1

        list_bg_subtracted_average.append(list_flux_average[i] - list_bg_fluxes[i])

        list_average_significance.append(list_bg_subtracted_average[i] / list_bg_std[i])

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
    df_info['Peak_electron_uncertainty'] = list_peak_electron_uncertainties
    df_info['Bg_electron_uncertainty'] = list_average_bg_uncertainties
    df_info['Peak_significance'] = list_peak_significance
    df_info['Flux_average'] = list_flux_average
    df_info['Bg_subtracted_average'] = list_bg_subtracted_average
    df_info['Average_significance'] = list_average_significance

    df_info['Backsub_peak_uncertainty'] = np.sqrt(
        df_info['Peak_electron_uncertainty']**2
        + df_info['Bg_electron_uncertainty']**2)

    df_info['rel_backsub_peak_err'] = np.abs(
        df_info['Backsub_peak_uncertainty']
        / df_info['Bg_subtracted_peak'])

    df_info['frac_nonan'] = list_frac_nonan

    return (df_electron_fluxes, df_info, [searchstart, searchend], [e_low, e_high], [instrument, data_type] )

# Aug 2026 I think this function is now redundant...
def make_step_electron_flux(stepdata, mask_conta=True):
    """
    Calculate STEP electron fluxes and uncertainties from integral and
    magnet fluxes.

    The electron flux is calculated from the difference between the
    integral and magnet fluxes using calibration factors from Paco
    (Alcala). If `mask_conta` is True, intervals where the difference
    between the integral and magnet flux is not sufficiently large are
    masked as likely ion-contaminated data.

    The current masking criterion is:

        F_INT - F_MAG > 2 * U_INT

    where `F_INT` and `F_MAG` are the integral and magnet fluxes,
    respectively, and `U_INT` is the uncertainty of the integral flux.

    Parameters
    ----------
    stepdata : pandas.DataFrame
        STEP data containing the integral and magnet fluxes and their
        corresponding uncertainties. The required columns are
        ``'Integral_Flux'``, ``'Magnet_Flux'``,
        ``'Integral_Uncertainty'``, and ``'Magnet_Uncertainty'``.

    mask_conta : bool, optional
        If True, intervals that do not satisfy the contamination
        threshold are masked. Defaults to True.

    Returns
    -------
    df_electron_fluxes : pandas.DataFrame
        Calculated STEP electron fluxes for each energy channel. Columns
        are named ``'Electron_Flux_0'``, ``'Electron_Flux_1'``, etc.

    df_electron_uncertainties : pandas.DataFrame
        Calculated uncertainties of the STEP electron fluxes. Columns
        are named ``'Electron_Uncertainty_0'``,
        ``'Electron_Uncertainty_1'``, etc.

    paco.E_low : pandas.Series
        Lower energy boundaries of the STEP energy channels in keV.

    paco.E_hi : pandas.Series
        Upper energy boundaries of the STEP energy channels in keV.
    """

    # Load calibration factors from Paco (Alcala).
    colnames = ["ch_num", "E_low", "E_hi", "factors"]
    paco = pd.read_csv('step_electrons_calibration.csv', names=colnames, skiprows=1)

    paco.E_low = round(paco.E_low / 1000, 5)
    paco.E_hi = round(paco.E_hi / 1000, 5)

    # Calculate electron flux from integral and magnet fluxes.
    integral_flux = stepdata['Integral_Flux']
    magnet_flux = stepdata['Magnet_Flux']

    step_flux = ((integral_flux - magnet_flux) * paco.factors.values)

    # Calculate electron flux uncertainties.
    integral_uncertainty = stepdata['Integral_Uncertainty']
    magnet_uncertainty = stepdata['Magnet_Uncertainty']

    step_uncertainty = (
        np.sqrt(integral_uncertainty**2 + magnet_uncertainty**2) * paco.factors.values)

    # Mask intervals where the electron signal is not sufficiently
    # larger than the magnet signal.
    if mask_conta:
        clean = ((integral_flux - magnet_flux) > 2 * integral_uncertainty)

        step_flux = step_flux.mask(~clean)
        step_uncertainty = step_uncertainty.mask(~clean)

    # Combine fluxes and uncertainties into separate DataFrames.
    step_data = pd.concat([step_flux, step_uncertainty], axis=1, keys=['Electron_Flux', 'Electron_Uncertainty'])

    df_electron_fluxes = step_data['Electron_Flux']
    df_electron_uncertainties = step_data['Electron_Uncertainty']

    # Add the expected column names for each energy channel.
    df_electron_fluxes.columns = [f'Electron_Flux_{channel}' for channel in df_electron_fluxes.columns]

    df_electron_uncertainties.columns = [f'Electron_Uncertainty_{channel}' for channel in df_electron_uncertainties.columns]

    return (df_electron_fluxes, df_electron_uncertainties, paco.E_low, paco.E_hi)

def average_flux_error(flux_err: pd.DataFrame) -> pd.Series:
    """
    Calculate the average flux uncertainty for each energy channel.

    Parameters
    ----------
    flux_err : pandas.DataFrame
        DataFrame containing flux uncertainties for each energy channel.

    Returns
    -------
    pandas.Series
        Average uncertainty for each energy channel.
    """
    return np.sqrt((flux_err ** 2).sum(axis=0)) / len(flux_err.values)

def plot_channels(args, bg_subtraction=False, savefig=False, sigma=3, path='', key='', frac_nan_threshold=0.4, 
    rel_err_threshold=0.5, plot_pa=False, coverage=None, viewing='sun', centre_pix=False, date=None, size=20 ):
    """
    Creates a timeseries plot showing the particle flux for each energy channel of
    the instrument (STEP, EPT, HET). The timeseries plot also shows the peak window
    and background window.

    The peak is marked with different colored lines:
        green: peak is acceptable
        grey: too many NaNs in window
        blue: low significance
        orange: high relative error
        purple: no valid background subtraction/significance

    Args:
        args : tuple
            Output of the extract_electron_data function. Contains:
                df_electron_fluxes: pandas DataFrame
                df_info: pandas DataFrame containing the spectrum data and metadata
                [searchstart, searchend]: search window start and end times
                [e_low, e_high]: lowest and highest energy for each channel
                [instrument, data_type]: instrument and data type

        bg_subtraction (bool, optional):
            Subtract the background flux from the data. Defaults to False.

        savefig (bool, optional):
            If True, save the timeseries plot. Defaults to False.

        sigma (int, optional):
            Significance threshold used to determine whether the peak is significant
            enough. Defaults to 3.

        path (str, optional):
            Path to the folder where the timeseries plot will be saved.
            Defaults to ''.

        key (str, optional):
            Additional string appended to the output filename. Defaults to ''.

        frac_nan_threshold (float, optional):
            Minimum fraction of non-NaN flux data points required in the search
            interval. Channels below this threshold are considered unreliable.
            Defaults to 0.4.

        rel_err_threshold (float, optional):
            Maximum allowed relative error. Channels above this threshold are
            considered unreliable. Defaults to 0.5.

        plot_pa (bool, optional):
            If True, include pitch-angle coverage in the plot. Defaults to False.

        coverage (pandas DataFrame or None, optional):
            DataFrame containing the pitch-angle coverage used for plotting.
            Defaults to None.

        viewing (str, optional):
            Viewing direction of EPT or HET, used for plotting pitch angles.
            Ignored for STEP. Defaults to 'sun'.

        centre_pix (bool, optional):
            Refers to STEP data and indicates whether centre-pixel data are used.
            Defaults to False.

        date (str, optional):
            Date used for the plot title and filename. If None, the date is taken
            from the plot period in df_info. Defaults to None.

        size (int, optional):
            Base font size used in the plot. Defaults to 20.
    """

    peak_sig = args[1]['Peak_significance']
    rel_err = args[1]['rel_backsub_peak_err']

    df_electron_fluxes = args[0]
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

    title_string = instrument.upper() + ', ' + data_type.upper() + ', ' + date_string

    filename = ('electron_channels-' + file_date+ '-' + instrument.upper() + '-' + viewing+ '-' + data_type.upper())

    if df_info['Averaging'][0] == 'Mean':

        title_string = (title_string+ ', ' + df_info['Averaging'][1].split()[2] + ' averaging')

        filename = (filename+ '-' + df_info['Averaging'][1].split()[2] + '_averaging')

    elif df_info['Averaging'][0] == 'No averaging':

        title_string = title_string + ', no averaging'
        filename = filename + '-no_averaging'

    if bg_subtraction:

        title_string = title_string + ', bg subtraction on'
        filename = filename + '-bg_subtr'

    else:

        title_string = title_string + ', bg subtraction off'

    if instrument.lower() == 'ept':

        if df_info['Ion_contamination_correction'][0]:

            title_string = title_string + ', ion correction on'
            filename = filename + '-ion_corr'

        elif df_info['Ion_contamination_correction'][0] == False:

            title_string = title_string + ', ion correction off'

    if instrument.lower() == 'step' and centre_pix:

        filename = filename + '-centre_pix'
        title_string = title_string + ', centre pix'

    # If background subtraction is enabled, subtract background flux from all
    # observations. Negative flux values are set to NaN.
    if bg_subtraction == False:
        pass

    elif bg_subtraction == True:

        df_electron_fluxes = df_electron_fluxes.sub(df_info['Background_flux'].values, axis=1)
        df_electron_fluxes[df_electron_fluxes < 0] = np.nan

    # Plotting part.
    color = {'sun': 'crimson', 'asun': 'orange', 'north': 'darkslateblue', 'south': 'c'}

    npanels = len(df_info['Energy_channel'])

    if plot_pa:
        npanels = npanels + 1

    if instrument.lower() == 'step':

        n_channels_step = len(args[1]['Energy_channel'])

        if n_channels_step > 8:
            fsize = (20, 60)

        else:
            fsize = (20, 24)

    if instrument.lower() == 'ept':
        fsize = (20, 48)

    if instrument.lower() == 'het':
        fsize = (20, 12)

    fig, axes = plt.subplots(npanels, sharex=True, figsize=fsize)

    fig.supylabel("Intensity [1/s cm$^2$ sr MeV]", size=size)

    axes[0].set_title(title_string + "\n", size=size)

    # Loop through selected energy channels and create a subplot for each.
    n = 0

    for channel in df_info['Energy_channel']:

        ax = axes[n]

        ax.plot(df_electron_fluxes.index, df_electron_fluxes['Electron_Flux_{}'.format(channel)], color=color[viewing], drawstyle='steps-mid')

        ax.set_yscale('log')

        plt.text(0.025, 0.7, str(energy_bin[0][channel])+ " - " + str(energy_bin[1][channel]) + " MeV", transform=ax.transAxes, size=size - 2)

        ax.tick_params(axis='y', which='major', labelsize=size - 2)

        # Search area vertical lines.
        ax.axvline(search_area[0][n], color='black')

        ax.axvline(search_area[1][n], color='black')

        ax.set_xlim(df_electron_fluxes.index[0], df_electron_fluxes.index[-1])

        # Peak vertical line.
        if df_info['Peak_timestamp'][n] is not pd.NaT:

            if rel_err[n] > rel_err_threshold:
                ax.axvline(df_info['Peak_timestamp'][n], linestyle=':', linewidth=4, color='orange')

            if df_info['frac_nonan'][n] < frac_nan_threshold:
                ax.axvline(df_info['Peak_timestamp'][n], linestyle='--', linewidth=3,color='gray')

            if peak_sig[n] < sigma:
                ax.axvline(df_info['Peak_timestamp'][n], linestyle='-.', linewidth=2,color='blue')

            if (peak_sig[n] >= sigma and rel_err[n] <= rel_err_threshold and df_info['frac_nonan'][n] > frac_nan_threshold):
                ax.axvline(df_info['Peak_timestamp'][n], color='green')

            if bg_subtraction == True:

                if (np.isnan(peak_sig[n]) and ~np.isnan(df_info['Bg_subtracted_peak'][n])):
                    ax.axvline(df_info['Peak_timestamp'][n], linestyle='-', linewidth=2, color='purple')

            if bg_subtraction == False:

                if (np.isnan(peak_sig[n]) and df_info['Flux_average'][n] != 0.):
                    ax.axvline(df_info['Peak_timestamp'][n], linestyle='-', linewidth=2, color='purple')

        # Background measurement area.
        ax.axvspan(df_info['Bg_start'][n], df_info['Bg_end'][n], color='gray', alpha=0.25)
        ax.get_xaxis().set_visible(False)

        if (n == len(df_info['Energy_channel']) - 1 and plot_pa == False):

            ax.get_xaxis().set_visible(True)
            ax.set_xlabel("Time", labelpad=45)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d\n%H:%M"))
            ax.tick_params(axis="x", labelrotation=45)

        n += 1

    if plot_pa:

        # Add a panel that shows the pitch angle of the telescope.
        ax = axes[n]

        if instrument.lower() in ['het', 'ept']:

            col = color[viewing]

            # Fill the minimum-maximum range of the pitch-angle coverage.
            ax.fill_between( coverage.index, coverage[viewing]['min'], coverage[viewing]['max'], alpha=0.5, color=col, edgecolor=col, linewidth=0.0, step='mid')

            # Plot the central pitch angle as a thin line.
            ax.plot(coverage.index, coverage[viewing]['center'], linewidth=0.7, label=viewing, color=col, drawstyle='steps-mid')

        if instrument.lower() == 'step':

            col_list = plt.cm.viridis(np.linspace(0., 0.95, 16))

            for p in range(1, 16):

                # Plot the central pitch angle as a thin line.
                ax.plot(coverage.index, coverage[f'Pixel_{p}']['center'], color=col_list[p - 1], linewidth=1, label=f'Pixel_{p}', drawstyle='steps-mid')

        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=instrument)

        ax.set_ylim([0, 180])

        ax.yaxis.set_ticks(np.arange(0, 180 + 45, 45))

        ax.set_ylabel('PA [°]', size=size - 2)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%y\n%H:%M"))

        plt.tick_params(axis='x', which='major', labelsize=size - 2, labelrotation=45)

        plt.tick_params(axis='y', which='major', labelsize=size - 2)

        ax.set_xlabel("Time", labelpad=45, size=size)

    # Saves figure, if enabled.
    if path[len(path) - 1] != '/':
        path = path + '/'

    if savefig:
        plt.savefig(path + filename + str(key) + '.jpg', bbox_inches='tight', dpi=300)

    plt.show()


def plot_some_channels(args, bg_subtraction=False, savefig=False, sigma=3, path='', key='', plot_pa=False, coverage=None, 
                       viewing='sun', frac_nan_threshold=0.9, 
                       rel_err_threshold=0.5, channels=None, figsize_x=15, figsize_y=8, f_scale=1, f_size=12):
    """Creates a timeseries plot for selected energy channels of the
    instrument (STEP, EPT, HET).

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
        args: Output of the extract_electron_data function. Contains:
            df_electron_fluxes: pandas DataFrame containing electron fluxes.
            df_info: pandas DataFrame containing spectrum data and metadata.
            [searchstart, searchend]: search window start and end times.
            [e_low, e_high]: lowest and highest energy for each channel.
            [instrument, data_type]: instrument and data type.

        bg_subtraction (bool, optional): Subtract background from the data.
            Defaults to False.

        savefig (bool, optional): Save the timeseries plot.
            Defaults to False.

        sigma (int, optional): Significance threshold used to check whether
            the peak is significant. Defaults to 3.

        path (str, optional): Path to the folder where the timeseries plot
            is saved. Defaults to ''.

        key (str, optional): Additional string added to the filename.
            Defaults to ''.

        plot_pa (bool, optional): Include a pitch-angle panel.
            Defaults to False.

        coverage (pandas DataFrame or None, optional): DataFrame used to
            plot the pitch-angle coverage. Defaults to None.

        viewing (str, optional): Viewing direction of EPT or HET used for
            plotting pitch angles. Defaults to 'sun'.

        frac_nan_threshold (float, optional): Threshold for the fraction
            of non-NaN data points in the search interval. Defaults to 0.9.

        rel_err_threshold (float, optional): Maximum allowed relative
            error. Defaults to 0.5.

        channels (list or None, optional): Energy channels to plot.
            If None, all available channels are plotted. Defaults to None.

        figsize_x (float, optional): Figure width. Defaults to 15.

        figsize_y (float, optional): Figure height. Defaults to 8.

        f_scale (float, optional): Font scaling factor. Defaults to 1.

        f_size (int, optional): Base font size. Defaults to 12.

    Raises:
        ValueError: If one or more requested channels are not available
            in the supplied data.
    """

    peak_sig = args[1]['Peak_significance']
    rel_err = args[1]['rel_backsub_peak_err']

    df_electron_fluxes = args[0]
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
        invalid_channels = [channel for channel in channels if channel not in available_channels]

        if invalid_channels:
            raise ValueError(f"Invalid channel(s): {invalid_channels}. Available channels are: {available_channels}.")

    # Make sure viewing is defined for STEP, where it is not used.
    if viewing is None or instrument.lower() == 'step':
        viewing = 'sun'

    title_string = (instrument.upper() + ', ' + data_type.upper() + ', ' + str(df_info['Plot_period'][0][:-5]))

    filename = ('channels-' + str(df_info['Plot_period'][0][:-5]) + '-' + instrument.upper() + '-' + data_type.upper())

    if df_info['Averaging'][0] == 'Mean':
        title_string = (title_string + ', ' + df_info['Averaging'][1].split()[2] + ' averaging')

        filename = (filename + '-' + df_info['Averaging'][1].split()[2] + '_averaging')

    elif df_info['Averaging'][0] == 'No averaging':

        title_string = title_string + ', no averaging'
        filename = filename + '-no_averaging'

    if bg_subtraction:

        title_string = title_string + ', bg subtraction on'
        filename = filename + '-bg_subtr'

    else:

        title_string = title_string + ', bg subtraction off'

    if instrument.lower() == 'ept':

        if df_info['Ion_contamination_correction'][0]:

            title_string = title_string + ', ion correction on'
            filename = filename + '-ion_corr'

        elif df_info['Ion_contamination_correction'][0] is False:

            title_string = title_string + ', ion correction off'

    # If background subtraction is enabled, subtract background flux from
    # all observations. Negative flux values are set to NaN.
    if bg_subtraction:
        df_electron_fluxes = df_electron_fluxes.sub(df_info['Background_flux'].values,axis=1)
        df_electron_fluxes[df_electron_fluxes < 0] = np.nan

    # Plotting part.
    sns.set_theme(style="white", font_scale=f_scale)

    # One panel for each selected channel, plus one optional pitch-angle panel.
    npanels = len(channels)

    if plot_pa:
        npanels += 1

    fig = plt.figure(figsize=(figsize_x, figsize_y))

    plt.xticks([], fontsize=f_size)
    plt.yticks([], fontsize=f_size)
    plt.ylabel("Intensity \n [1/s cm$^2$ sr MeV] \n \n", size=f_size)
    plt.xlabel("\n \n Time", size=f_size)
    plt.title(title_string, size=f_size)

    # Loop through selected energy channels.
    for n, channel in enumerate(channels, start=1):

        if plot_pa:
            ax = fig.add_subplot(npanels, 1, n)
        else:
            ax = fig.add_subplot(len(channels), 1, n)

        ax = df_electron_fluxes['Electron_Flux_{}'.format(channel)].plot(logy=True, figsize=(figsize_x, figsize_y), color='red',drawstyle='steps-mid')

        plt.text(0.025, 0.7, str(energy_bin[0][channel]) + " - " + str(energy_bin[1][channel]) + " MeV", transform=ax.transAxes, size=f_size)

        # Search area vertical lines.
        channel_index = available_channels.index(channel)

        ax.axvline(search_area[0][channel_index], color='black')
        ax.axvline(search_area[1][channel_index], color='black')

        # Peak vertical line.
        if df_info['Peak_timestamp'][channel_index] is not pd.NaT:

            if rel_err[channel_index] > rel_err_threshold:
                ax.axvline(df_info['Peak_timestamp'][channel_index], linestyle=':', linewidth=4, color='orange')

            if df_info['frac_nonan'][channel_index] < frac_nan_threshold:
                ax.axvline(df_info['Peak_timestamp'][channel_index], linestyle='--', linewidth=3, color='gray')

            if peak_sig[channel_index] < sigma:
                ax.axvline(df_info['Peak_timestamp'][channel_index], linestyle='-.', linewidth=2, color='blue')

            if (peak_sig[channel_index] >= sigma and rel_err[channel_index] <= rel_err_threshold  
                and df_info['frac_nonan'][channel_index] > frac_nan_threshold):
                ax.axvline(df_info['Peak_timestamp'][channel_index], color='green')

        # Background measurement area.
        ax.axvspan(df_info['Bg_start'][channel_index], df_info['Bg_end'][channel_index], color='gray', alpha=0.25)
        ax.get_xaxis().set_visible(False)

        # Show the time axis only on the last timeseries panel if there
        # is no pitch-angle panel.
        if n == len(channels) and not plot_pa:
            ax.get_xaxis().set_visible(True)
            plt.xlabel("")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d\n%H:%M"))
            ax.tick_params(axis="x", labelrotation=45)
            


    # Optional pitch-angle panel.
    if plot_pa:

        ax = fig.add_subplot(npanels, 1, npanels)

        color = {'sun': 'crimson', 'asun': 'orange', 'north': 'darkslateblue', 'south': 'c'}

        if instrument.lower() in ['het', 'ept']:

            col = color[viewing]

            ax.fill_between(coverage.index, coverage[viewing]['min'], coverage[viewing]['max'], alpha=0.5, color=col, edgecolor=col, linewidth=0.0, step='mid')
            ax.plot(coverage.index, coverage[viewing]['center'], linewidth=0.7, label=viewing, color=col,  drawstyle='steps-mid')

        if instrument.lower() == 'step':
            col_list = plt.cm.viridis(np.linspace(0., 0.95, 16))

            for p in range(1, 16):
                ax.plot(coverage.index, coverage[f'Pixel_{p}']['center'], color=col_list[p - 1], linewidth=1, label=f'Pixel_{p}', drawstyle='steps-mid')

        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=instrument)

        ax.set_ylim([0, 180])
        ax.yaxis.set_ticks(np.arange(0, 180 + 45, 45))
        ax.set_ylabel('PA / °', size=f_size)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d\n%H:%M"))

        plt.tick_params(axis='x', which='major', labelsize=f_size, labelrotation=45)
        plt.tick_params(axis='y', which='major', labelsize=f_size)

        ax.set_xlabel("Time", labelpad=45, size=f_size)

    # Saves figure, if enabled.
    if path and path[-1] != '/':
        path = path + '/'

    if savefig:
        plt.savefig(path + filename + str(key) + '.jpg', bbox_inches='tight', dpi=300)

    plt.show()

def plot_spectrum_peak(args, bg_subtraction=True,
    savefig=False, path='', key='',
    sigma=3, frac_nan_threshold=0.4,
    rel_err_threshold=0.5, direction=None,
    centre_pix=False, date=None):
    """
    Creates an energy spectrum plot using the peak flux values from each
    energy channel.

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
            Output of the extract_data function. Contains:
                df_electron_fluxes: pandas DataFrame containing electron fluxes.
                df_info: pandas DataFrame containing spectrum data and metadata.
                [searchstart, searchend]: search-window start and end times.
                [e_low, e_high]: lower and upper energies for each energy channel.
                [instrument, data_type]: instrument and data-product type.

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
    color = {'sun': 'crimson', 'asun': 'orange', 'north': 'darkslateblue', 'south': 'c'}


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

    title_string = (f"{instrument.upper()}, {data_type.upper()}, {date_string}")

    filename = (f"electron_spectrum-{file_date}-{instrument.upper()}{viewing}-{data_type.upper()}")

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

    # Add ion-contamination correction information for EPT.
    if instrument_lower == 'ept':
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
    df_nan = df_info.where(df_info['frac_nonan'] < frac_nan_threshold, np.nan)

    df_no_sig = df_info.where(df_info['Peak_significance'] < sigma, np.nan)

    df_rel_err = df_info.where(df_info['rel_backsub_peak_err'] > rel_err_threshold, np.nan)

    # Plot either background-subtracted or raw peak fluxes.
    fig, ax = plt.subplots(figsize=(13, 10))

    if bg_subtraction:
        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Bg_subtracted_peak'],
                    yerr=df_info['Backsub_peak_uncertainty'], xerr=[ df_info['Energy_error_low'],df_info['Energy_error_high']], 
                    color=color[direction], fmt='o', ecolor=color[direction], zorder=0, label='Flux peaks')

        ax.plot(df_nan.Primary_energy, df_nan.Bg_subtracted_peak, 'o', markersize=15, c='gray', label='excluded (NaNs)')

        ax.plot(df_no_sig.Primary_energy, df_no_sig.Bg_subtracted_peak, 'o', markersize=11, c='blue', label='excluded (sigma)')

        ax.plot(df_rel_err.Primary_energy, df_rel_err.Bg_subtracted_peak, 'o', markersize=6, c='orange', label='excluded (rel error)')

    else:
        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Flux_peak'],
                    yerr=df_info['Peak_electron_uncertainty'], xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']],
                    fmt='o', color=color[direction], ecolor=color[direction], zorder=0, label='Intensity peaks')

        ax.plot(df_nan.Primary_energy, df_nan.Flux_peak, 'o',  markersize=15, c='gray', label='excluded (NaNs)')
        ax.plot(df_no_sig.Primary_energy, df_no_sig.Flux_peak, 'o', markersize=11, c='blue', label='excluded (sigma)')
        ax.plot(df_rel_err.Primary_energy, df_rel_err.Flux_peak, 'o', markersize=6, c='orange', label='excluded (rel error)')

    # Plot background intensity and its uncertainty.
    ax.errorbar(x=df_info['Primary_energy'], y=df_info['Background_flux'],
                yerr=df_info['Bg_electron_uncertainty'], xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']],
                fmt='o', color=color[direction], ecolor=color[direction], alpha=0.15, label='Background intensity')

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel('Energy [MeV]', size=20)
    ax.set_ylabel('Intensity \n [1/s cm$^2$ sr MeV]', size=20)

    plt.tick_params(axis='x', which='minor', labelsize=16)

    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))

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

        plt.savefig(path + filename + str(key) + '.jpg', dpi=300, bbox_inches='tight')

    plt.show()

def plot_spectrum_average(args, bg_subtraction=True, savefig=False, path='', key='', sigma=3, frac_nan_threshold=0.4, 
                          rel_err_threshold=0.5, direction=None, centre_pix=False, date=None):
    """
    Creates an energy spectrum plot using the average flux values from each
    energy channel.

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
            Output of the extract_data function. Contains:
                df_electron_fluxes: pandas DataFrame containing electron fluxes.
                df_info: pandas DataFrame containing spectrum data and metadata.
                [searchstart, searchend]: search-window start and end times.
                [e_low, e_high]: lower and upper energies for each energy channel.
                [instrument, data_type]: instrument and data-product type.

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

    color = {'sun': 'crimson', 'asun': 'orange', 'north': 'darkslateblue', 'south': 'c'}
    

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

    title_string = (f"{instrument.upper()}, {data_type.upper()}, {date_string}")

    filename = (f"electron_spectrum-{file_date}-{instrument.upper()}{viewing}-{data_type.upper()}")

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

    # Add ion-contamination correction information for EPT.
    if instrument_lower == 'ept':
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
    df_nan = df_info.where(df_info['frac_nonan'] < frac_nan_threshold, np.nan)

    df_no_sig = df_info.where(df_info['Average_significance'] < sigma, np.nan)

    df_rel_err = df_info.where(df_info['rel_backsub_peak_err'] > rel_err_threshold, np.nan)

    # Plot either background-subtracted or raw average fluxes.
    fig, ax = plt.subplots(figsize=(13, 10))

    if bg_subtraction:
        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Bg_subtracted_average'],
            yerr=df_info['Backsub_peak_uncertainty'], xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']],
            color=color[direction], fmt='o', ecolor=color[direction], zorder=0, label='Intensity average')

        ax.plot(df_nan.Primary_energy, df_nan.Bg_subtracted_average, 'o', markersize=15, c='gray', label='excluded (NaNs)')

        ax.plot(df_no_sig.Primary_energy, df_no_sig.Bg_subtracted_average, 'o', markersize=11, c='blue', label='excluded (sigma)')

        ax.plot(df_rel_err.Primary_energy, df_rel_err.Bg_subtracted_average, 'o', markersize=6, c='orange', label='excluded (rel error)')

    else:
        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Flux_average'],
                    yerr=df_info['Peak_electron_uncertainty'], xerr=[df_info['Energy_error_low'],df_info['Energy_error_high']],
                    fmt='o', color=color[direction], ecolor=color[direction], zorder=0, label='Intensity average')

        ax.plot(df_nan.Primary_energy, df_nan.Flux_average, 'o', markersize=15, c='gray', label='excluded (NaNs)')
        ax.plot(df_no_sig.Primary_energy, df_no_sig.Flux_average, 'o', markersize=11, c='blue', label='excluded (sigma)')
        ax.plot(df_rel_err.Primary_energy, df_rel_err.Flux_average, 'o', markersize=6, c='orange', label='excluded (rel error)')

    # Plot background intensity and its uncertainty.
    ax.errorbar(x=df_info['Primary_energy'], y=df_info['Background_flux'],
                yerr=df_info['Bg_electron_uncertainty'], xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']],
                fmt='o', color=color[direction], ecolor=color[direction], alpha=0.15, label='Background intensity')

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel('Energy [MeV]', size=20)
    ax.set_ylabel('Intensity \n [1/s cm$^2$ sr MeV]', size=20)

    plt.tick_params(axis='x', which='minor', labelsize=16)

    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))

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

        plt.savefig(path + filename + str(key) + '.jpg', dpi=300, bbox_inches='tight')

    plt.show()

def write_to_csv(args, date, path='', key='', direction=None, centre_pix=False):
    """
    Saves the spectrum information dataframe to a CSV file.

    The output filename contains the date, instrument, viewing direction,
    data-product type, averaging information, and relevant instrument-specific
    processing information.

    Args:
        args (tuple):
            Output of the extract_data function. Contains:
                df_electron_fluxes: pandas DataFrame containing electron fluxes.
                df_info: pandas DataFrame containing spectrum data and metadata.
                [searchstart, searchend]: search-window start and end times.
                [e_low, e_high]: lower and upper energies for each energy channel.
                [instrument, data_type]: instrument and data-product type.

        date (str):
            Date used in the output filename.

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

    # Use sun as the default viewing direction.
    viewing = 'sun' if direction is None else direction

    filename = (f'electron_data-{date}-{instrument.upper()}-{viewing}-{data_type.upper()}')

    # Add averaging information.
    if df_info['Averaging'][0] == 'Mean':
        averaging = df_info['Averaging'][1].split()[2]
        filename += f'-{averaging}_averaging'

    elif df_info['Averaging'][0] == 'No averaging':
        filename += '-no_averaging'

    # Add ion-contamination correction information for EPT.
    if instrument_lower == 'ept':
        if df_info['Ion_contamination_correction'][0]:
            filename += '-ion_corr'

    # Add centre-pixel information for STEP.
    if instrument_lower == 'step' and centre_pix:
        filename += '-centre_pix'

    # Save dataframe.
    if path and not path.endswith('/'):
        path += '/'

    df_info.to_csv(path + filename + str(key) + '.csv', sep=';', index=False)


def centre_pix_average_comparison_spec(args, args_pix, bg_subtraction=True, savefig=False, path='', key='', sigma=3, 
                                       frac_nan_threshold=0.4, rel_err_threshold=0.5, date=None):
    """
    Plot and compare the STEP electron spectrum obtained from all-pixel
    averaged data with the spectrum obtained from the centre pixel.

    The plot shows either the background-subtracted or raw flux peaks for
    both datasets, including their energy uncertainties and flux
    uncertainties. Background fluxes are also shown for both datasets.

    Data points that do not meet the quality criteria are marked according
    to the reason for exclusion:
        - grey: too many NaNs in the search interval
        - blue: significance below the sigma threshold
        - orange: relative error above the threshold

    This function is specific to STEP, for which the viewing direction is
    always the sun direction.

    Args:
        args (tuple): Output of the data extraction function for the
            all-pixel averaged STEP data. Contains electron fluxes,
            spectrum information, search windows, energy bins, instrument
            information, and data type.
        args_pix (tuple): Output of the data extraction function for the
            centre-pixel STEP data.
        bg_subtraction (bool, optional): If True, plot background-subtracted
            fluxes. If False, plot raw fluxes. Defaults to True.
        savefig (bool, optional): If True, save the generated figure.
            Defaults to False.
        path (str, optional): Path to the folder where the figure is saved.
            Defaults to ''.
        key (str, optional): Additional string appended to the output
            filename. Defaults to ''.
        sigma (int, optional): Significance threshold used to identify
            channels with insufficiently significant flux peaks.
            Defaults to 3.
        frac_nan_threshold (float, optional): Minimum fraction of
            non-NaN data points required in the search interval.
            Defaults to 0.4.
        rel_err_threshold (float, optional): Maximum allowed relative
            uncertainty of the background-subtracted peak.
            Defaults to 0.5.
        date (optional): Date used for the plot title and filename. If None,
            the date is taken from ``df_info['Plot_period']``.
            Defaults to None.

    Returns:
        None
    """

    color = {'sun': 'crimson', 'sun_pix': 'purple'}

    df_info = args[1]
    df_info_pix = args_pix[1]

    instrument = 'STEP'
    data_type = args[4][1]
    direction = 'sun'

    date_string = ''
    file_date = ''

    if date is None:
        date_string = str(df_info['Plot_period'][0][:-5])
        file_date = str(df_info['Plot_period'][0][:-5])
    else:
        date_string = str(date)[:-3]
        file_date = str(date)[:-3].replace(' ', '-').replace(':', '')

    title_string = instrument + ', ' + data_type.upper() + ', ' + date_string
    filename = ('spectrum-pix-comparison-' + file_date + '-' + instrument + '-' + direction + '-' + data_type.upper())

    if df_info['Averaging'][0] == 'Mean':

        title_string = (title_string + ', ' + df_info['Averaging'][1].split()[2] + ' averaging')
        filename = (filename + '-' + df_info['Averaging'][1].split()[2] + '_averaging')

    elif df_info['Averaging'][0] == 'No averaging':

        title_string = title_string + ', no averaging'
        filename = filename + '-no_averaging'

    if bg_subtraction:

        title_string = title_string + ', bg subtraction on'
        filename = filename + '-bg_subtr'

    else:

        title_string = title_string + ', bg subtraction off'

    # Identify points excluded due to the different quality criteria.
    df_nan = df_info.where(df_info['frac_nonan'] < frac_nan_threshold, np.nan)
    df_no_sig = df_info.where(df_info['Peak_significance'] < sigma, np.nan)
    df_rel_err = df_info.where(df_info['rel_backsub_peak_err'] > rel_err_threshold, np.nan)

    df_nan_pix = df_info_pix.where(df_info_pix['frac_nonan'] < frac_nan_threshold, np.nan)
    df_no_sig_pix = df_info_pix.where(df_info_pix['Peak_significance'] < sigma, np.nan)
    df_rel_err_pix = df_info_pix.where(df_info_pix['rel_backsub_peak_err'] > rel_err_threshold, np.nan)

    # Plot either the background-subtracted or raw flux peaks.
    if bg_subtraction:

        f, ax = plt.subplots(figsize=(13, 10))

        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Bg_subtracted_peak'],
                    yerr=df_info['Backsub_peak_uncertainty'], xerr=[df_info['Energy_error_low'],df_info['Energy_error_high']],
                    color=color['sun'], fmt='o', ecolor=color['sun'], zorder=0, label='Intensity peaks all pix avg')

        ax.plot(df_nan.Primary_energy, df_nan.Bg_subtracted_peak, 'o', markersize=15, c='gray', label='excluded (NaNs)')
        ax.plot(df_no_sig.Primary_energy, df_no_sig.Bg_subtracted_peak, 'o', markersize=11, c='blue', label='excluded (sigma)')
        ax.plot(df_rel_err.Primary_energy, df_rel_err.Bg_subtracted_peak, 'o', markersize=6, c='orange', label='excluded (rel error)')

        ax.errorbar(x=df_info_pix['Primary_energy'], y=df_info_pix['Bg_subtracted_peak'],
                    yerr=df_info_pix['Backsub_peak_uncertainty'], xerr=[df_info_pix['Energy_error_low'],df_info_pix['Energy_error_high']],
                    color=color['sun_pix'], fmt='o', ecolor=color['sun_pix'], zorder=0, label='Intensity peaks centre pix')

        ax.plot(df_nan_pix.Primary_energy, df_nan_pix.Bg_subtracted_peak, 'o', markersize=15, c='gray')
        ax.plot(df_no_sig_pix.Primary_energy, df_no_sig_pix.Bg_subtracted_peak, 'o', markersize=11, c='blue')
        ax.plot(df_rel_err_pix.Primary_energy, df_rel_err_pix.Bg_subtracted_peak, 'o', markersize=6, c='orange')

    else:

        f, ax = plt.subplots(figsize=(13, 10))

        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Flux_peak'],
                    yerr=df_info['Peak_electron_uncertainty'], xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']],
                    fmt='o', color=color['sun'], ecolor=color['sun'], zorder=0, label='Intensity peaks all pix avg')

        ax.plot(df_nan.Primary_energy, df_nan.Flux_peak, 'o', markersize=15, c='gray', label='excluded (NaNs)')
        ax.plot(df_no_sig.Primary_energy, df_no_sig.Flux_peak, 'o', markersize=11, c='blue', label='excluded (sigma)')
        ax.plot(df_rel_err.Primary_energy, df_rel_err.Flux_peak, 'o', markersize=6, c='orange', label='excluded (rel error)')

        ax.errorbar(x=df_info_pix['Primary_energy'], y=df_info_pix['Flux_peak'], 
                    yerr=df_info_pix['Peak_electron_uncertainty'], xerr=[df_info_pix['Energy_error_low'], df_info_pix['Energy_error_high']],
                    fmt='o', color=color['sun_pix'], ecolor=color['sun_pix'], zorder=0, label='Intensity peaks centre pix')

        ax.plot(df_nan_pix.Primary_energy, df_nan_pix.Flux_peak, 'o', markersize=15, c='gray')
        ax.plot(df_no_sig_pix.Primary_energy, df_no_sig_pix.Flux_peak, 'o', markersize=11, c='blue')
        ax.plot(df_rel_err_pix.Primary_energy, df_rel_err_pix.Flux_peak, 'o', markersize=6, c='orange')

    # Plot background flux and background uncertainties for both datasets.
    ax.errorbar(x=df_info['Primary_energy'], y=df_info['Background_flux'],
                yerr=df_info['Bg_electron_uncertainty'], xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']],
                fmt='o', color=color['sun'], ecolor=color['sun'], alpha=0.15, label='Background intensity all pix avg')

    ax.errorbar(x=df_info_pix['Primary_energy'], y=df_info_pix['Background_flux'],
                yerr=df_info_pix['Bg_electron_uncertainty'], xerr=[df_info_pix['Energy_error_low'], df_info_pix['Energy_error_high']],
                fmt='o', color=color['sun_pix'], ecolor=color['sun_pix'], alpha=0.15, label='Background intensity centre pix')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Energy [MeV]', size=20)
    ax.set_ylabel('Intensity \n [1/s cm$^2$ sr MeV]', size=20)

    plt.tick_params(axis='x', which='minor', labelsize=16)
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))

    plt.legend(prop={'size': 18})
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.grid()
    plt.title(title_string)

    for label in ax.xaxis.get_ticklabels(which='minor')[1::2]:
        label.set_visible(False)

    if path[len(path) - 1] != '/':
        path = path + '/'

    if savefig:
        plt.savefig(path + filename + str(key) + '.jpg', dpi=300, bbox_inches='tight')

    plt.show()

# aug2026 no idea...    
# This acc_flux function is not really finished, just something I put together quickly.
def acc_flux(args, time=[]):
    """_summary_

    Args:
        args (_type_): _description_
        time (list, optional): _description_. Defaults to [].
    """

    df_electron_fluxes = args[0]
    df_info = args[1]

    # If no timeframe specified, use search area.
    if(time==[]):

        time = args[2]

    # Calculates average fluxes for each enery channel from given timeframe and appends to list.
    list_flux_averages = []

    for channel in df_info['Energy_channel']:

        list_flux_averages.append(df_electron_fluxes['Electron_Flux_{}'.format(channel)][time[0]:time[1]].mean())

    df_acc = pd.DataFrame({'Primary_energy':[], 'Acc_flux':[]})
    df_acc['Primary_energy'] = df_info['Primary_energy']
    df_acc['Acc_flux'] = list_flux_averages

    ax = df_acc.plot(kind='scatter', x='Primary_energy', y='Acc_flux', logy=True, logx=True, color='green', figsize=(13,10))



# This plot_check function is not finished, but it does produce cool rainbow coloured plots.
# aug26 do not remember what this was supposed to do...
def plot_check(args, bg_subtraction=False, savefig=False, key=''):
    """_summary_

    Args:
        args (_type_): _description_
        bg_subtraction (bool, optional): _description_. Defaults to False.
        savefig (bool, optional): _description_. Defaults to False.
        key (str, optional): _description_. Defaults to ''.
    """

    hours = mdates.HourLocator(interval = 1)
    df_electron_fluxes = args[0]
    df_info = args[1]
    search_area = args[2]
    energy_bin = args[3]
    instrument = args[4][0]
    data_type = args[4][1]

    fig = plt.figure()
    colors = iter(plt.cm.jet(np.linspace(0, 1, len(df_info['Energy_channel']))))

    #for channel in df_info['Energy_channel']:
    #    ax = df_electron_fluxes['Electron_Flux_{}'.format(channel)].plot(logy=True, figsize=(20,25), color='red', drawstyle='steps-mid')

    for channel in df_info['Energy_channel']:

        col = next(colors)
        ax = df_electron_fluxes['Electron_Flux_{}'.format(channel)].plot(logy=True, figsize=(13,10), color=col, drawstyle='steps-mid')

    plt.show()

