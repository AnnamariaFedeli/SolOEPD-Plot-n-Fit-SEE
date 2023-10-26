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

# Add folder for data and one for plots
def create_new_path(path, date, threshold_folders = False, contamination_threshold = None, plots_n_data = True):
    """This function creates new folders to a given path based on different options.
    Args:
        path (str): Path to where the new folders will be created.
        date (str): Date of an event in string form. In theory the string can be of any kind, but for the purposes of this sofware, it should be 'YYYY-mm-dd-hhMM'.
        threshold_folders (bool, optional): In case you wish to create new folders based on different contamination thresholds set to True. This option was added only to 
                make checking the difference between files easier. But after deciding on a certain threshold this shold be se to False. Also, if True some other options such as 
                plots_n data will not work. Defaults to False.
        contamination_threshold (int, optional): The threshold value will be added to the folder name when, and only if, threshold folders = True. Defaults to None.
        plots_n_data (bool, optional): If True, the plots (e.g. spectrum, timeseries, fits) will also be saved to a separate folder called 'plots' inside the event folder. 
                Defaults to True.
    """
    newpath = path+date
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    print("Creating new directory "+newpath)

    if threshold_folders :
        nnewpath = newpath+'/'+'contamination_threshold_'+str(contamination_threshold)

        if not os.path.exists(nnewpath):
            os.makedirs(nnewpath)
            print('Creating new directory  '+ nnewpath)

        if plots_n_data:
            plots_path = nnewpath+'/plots'

            if not os.path.exists(plots_path):
                os.makedirs(plots_path)
                print('Creating new directory  '+ plots_path)
            #if not os.path.exists(data_path):
             #   os.makedirs(data_path)
              #  print('Creating new directory  '+ data_path)
    else:
        if plots_n_data:
            plots_path = newpath+'/plots'
            
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)
                print('Creating new directory  '+ plots_path)
            #if not os.path.exists(data_path):
             #   os.makedirs(data_path)
              #  print('Creating new directory  '+ data_path)

            
        


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def calc_pa_coverage(instrument, mag_data):
    """_summary_

    Args:
        instrument (_type_): _description_
        mag_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    print(f'Calculating PA coverage for {instrument}...')
    if instrument not in ['ept', 'EPT', 'het', 'HET', 'step', 'STEP']:
        print("instrument not known, select 'EPT', 'HET', or 'STEP' ")
        coverage = pd.DataFrame(mag_data.index)
    else:
        if instrument.lower() == 'ept':
            opening = 30
        if instrument.lower() == 'het':
            opening = 43
        if instrument.lower() == 'step':
            print("Opening of STEP just a placeholder! Replace with real value! This affects the 'min' and 'max' values of the pitch-angle, not the 'center' ones.")
            opening = 10
        mag_vec = np.array([mag_data.Bx.values, mag_data.By.values, mag_data.Bz.values])

        if instrument.lower() in ['ept', 'het']:
            # pointing directions of EPT in XYZ/SRF coordinates (!) (arrows point into the sensor)
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

        if instrument.lower() == 'step':
            # Particle flow direction (unit vector) in spacecraft XYZ coordinates for each STEP pixel ('XYZ_Pixels')
            pointing_step = np.array([[-0.8412, 0.4396,  0.3149],
                                      [-0.8743, 0.457 ,  0.1635],
                                      [-0.8862, 0.4632, -0.    ],
                                      [-0.8743, 0.457 , -0.1635],
                                      [-0.8412, 0.4396, -0.315 ],
                                      [-0.7775, 0.5444,  0.3149],
                                      [-0.8082, 0.5658,  0.1635],
                                      [-0.8191, 0.5736,  0.    ],
                                      [-0.8082, 0.5659, -0.1634],
                                      [-0.7775, 0.5444, -0.3149],
                                      [-0.7008, 0.6401,  0.3149],
                                      [-0.7284, 0.6653,  0.1634],
                                      [-0.7384, 0.6744, -0.    ],
                                      [-0.7285, 0.6653, -0.1635],
                                      [-0.7008, 0.6401, -0.315 ]])
            pa_step = np.ones((len(mag_data.Bx.values), pointing_step.shape[0])) * np.nan

            for i in tqdm(range(len(mag_data.Bx.values))):
                for j in range(pointing_step.shape[0]):
                    pa_step[i, j] = np.rad2deg(angle_between(pointing_step[j], mag_vec[:, i]))

    if instrument.lower() in ['ept', 'het']:
        sun_min = pa_sun - opening/2
        sun_max = pa_sun + opening/2
        asun_min = pa_asun - opening/2
        asun_max = pa_asun + opening/2
        north_min = pa_north - opening/2
        north_max = pa_north + opening/2
        south_min = pa_south - opening/2
        south_max = pa_south + opening/2
        cov_sun = pd.DataFrame({'min': sun_min, 'center': pa_sun, 'max': sun_max}, index=mag_data.index)
        cov_asun = pd.DataFrame({'min': asun_min, 'center': pa_asun, 'max': asun_max}, index=mag_data.index)
        cov_north = pd.DataFrame({'min': north_min, 'center': pa_north, 'max': north_max}, index=mag_data.index)
        cov_south = pd.DataFrame({'min': south_min, 'center': pa_south, 'max': south_max}, index=mag_data.index)
        keys = [('sun'), ('asun'), ('north'), ('south')]
        coverage = pd.concat([cov_sun, cov_asun, cov_north, cov_south], keys=keys, axis=1)

    if instrument.lower() == 'step':
        pa_step_min = pa_step - opening/2
        pa_step_max = pa_step + opening/2

        cov = {}
        for i in range(pa_step.shape[1]):
            cov[f'Pixel_{i+1}'] = pd.DataFrame({'min': pa_step_min[:, i], 'center': pa_step[:, i], 'max': pa_step_max[:, i]}, index=mag_data.index)
        coverage = pd.concat(cov, keys=cov.keys(), axis=1)

    coverage[coverage > 180] = 180
    coverage[coverage < 0] = 0
    return coverage

def solo_mag_loader(sdate, edate, level='l2', type='normal', frame='rtn', av=None, path=None):
    """
    to do: implement higher resultion averaging ('1S' (seconds)) for burst data
    loads SolO/MAG data from soar using function mag_load() from Jan: 
    autodownloads if files are not there

    Parameters
    ----------
    sdate : int
        20210417
    edate : int
        20210418
    level : str, optional
        by default 'l2'
    type : str, optional
        'burst', 'normal-1-minute', by default 'normal'
    frame : str, optional
        'srf', by default 'rtn'
    av : int or None, optional
        use pandas strings CHANGE
        number of minutes to average, by default None

    Returns
    -------
    [type]
        [description]
    """    
    print('Loading MAG...')
    mag_data = mag_load(sdate, edate, level=level, data_type=type, frame=frame, path=path)
    #mag_data = mag_load(sdate, edate, level=level, frame=frame, path=path)
    if frame == 'rtn':
        mag_data.rename(columns={'B_RTN_0':'B_r', 'B_RTN_1':'B_t', 'B_RTN_2':'B_n'}, inplace=True)
    if frame == 'srf':
        mag_data.rename(columns={'B_SRF_0':'Bx', 'B_SRF_1':'By', 'B_SRF_2':'Bz'}, inplace=True)
    if av is not None:
        mav = av 
        m_int = int(re.search(r'\d+', av).group())/2
        m_string = ''.join([i for i in av if not i.isdigit()])
        mag_offset = str(m_int)+m_string
    
        mag_data = mag_data.resample(mav,label='left').mean()
        mag_data.index = mag_data.index + to_offset(mag_offset)

    return mag_data






def evolt2beta(ekin, which):
    """ This function calculates the plasma beta for particles 
    (electrons or protons) with a certain energy ekin.

    Args:
        ekin (float): input energy in MeV
        which (int): 1 for protons, 2 for electrons

    Returns:
        float: plasma beta
    """
    c = 299792458.  # m/s,  speed of light
    me0 = 9.109e-31 # kg,   mass of electron
    mp0 = 1.67e-27
    q = 1.60217646e-19  # C  charge
    ekin = ekin*1.0e6*q  # E in MeV to eV and then to Joule
    
    betae = np.sqrt(1-(me0*c**2/(ekin+me0*c**2))**2)
    betap = np.sqrt(1-(mp0*c**2/(ekin+mp0*c**2))**2)
    
    if (which == 1):
        beta = betap

    if (which == 2):
        beta = betae
    #print 'beta: ', beta
    return beta

def evolt2speed(ekin, which):
    """ This function calculates the velocity of particles
    (protons or electrons) with a certain energy ekin
    Args:
        ekin (float): energy in MeV
        which (int): 1 for protons, 2 for electrons

    Returns:
        float: particle velocity in km/s
    """
    c = 299792458.  # m/s,  speed of light
    
    beta = evolt2beta(ekin, which)
    
    v = beta*c
    v = v/1000. # in km/s
    return v

#searchstart, searchend,

def len_of_spiral(vsw, dist):
    """_summary_

    Args:
        vsw (_type_): _description_
        dist (_type_): _description_

    Returns:
        _type_: _description_
    """
    # dist = np.float(dist)
    #print('dist:', dist, type(dist))
    #print('')
    omega = np.deg2rad(360./(25.38*24.*60.*60.)) #in deg per sec
    AU = 1*u.au
    r     = AU.to(u.km).value * dist    # ~ 1AU sc Distance in km multiplied by dist in AU
    r_a   = 695700.      # 1R_sun in km
    #r_a = 0.05 * AU.to(u.km).value


    # length of spiral in kmN
    R_s = 0.5*omega/vsw*(r-r_a)*np.sqrt((r-r_a)**2+(vsw/omega)**2)+0.5*vsw/omega*asinh((r-r_a)/vsw*omega)
    new_R_s = R_s/AU.to(u.km).value            # '  length of spiral in AU'

    return new_R_s


def traveltime_los(los, energy, which):
    """_summary_

    Args:
        los (_type_): _description_
        energy (_type_): _description_
        which (_type_): _description_
        dist (_type_): _description_

    Returns:
        _type_: _description_
    """

    v_e = evolt2speed(energy, which)

    R_s = los * 149597870.691 #  dist is already taken into account in los (length of spiral)

    # traveltime of electrons with given keV energy
    t = R_s/v_e
    #print(t)
    #t is now in seconds

    return t

def light_tt(dist):
    """_summary_

    Args:
        dist (_type_): _description_

    Returns:
        _type_: _description_
    """
    # dist in AU
    v = 299792458  # m/s
    au2m = 149597870691# m

    dist = dist*au2m
    t = dist/v  # in sec
    #t = t/60.
    return t

def position_and_traveltime(date, species):
    """_summary_

    Args:
        date (_type_): _description_
    """
    if species.lower() in ['electron', 'electrons', 'e']:
        which = 2
    if species.lower() in ['proton', 'protons', 'p']:
        which = 1
    
    pos = get_horizons_coord('Solar Orbiter', date, 'id')
    dist = np.round(pos.radius.value, 2)
    spiral_len = len_of_spiral(400,dist)
    traveltime_min = traveltime_los(spiral_len, 0.004, which)
    traveltime_max = traveltime_los(spiral_len, 10, which)
    light_t = light_tt(dist)
    min_sec = 's'

    if traveltime_min > 90:
        traveltime_min = traveltime_min/60.
        traveltime_max = traveltime_max/60.
        light_t = light_t/60
        min_sec = 'min'
    
    
    if species.lower() in ['electron', 'electrons', 'e']:
        table_data = [["Distance of SolO from the Sun", "[AU]", dist],
                    ["Length of the Parker Spiral for 400 km/s sw ", "[AU]", spiral_len],
                    ["Travel time of 4 KeV electrons ", "["+min_sec+"]", traveltime_min],
                    ["Travel time of 10 MeV electrons ", "["+min_sec+"]", traveltime_max],
                    ["Travel time of light ", "["+min_sec+"]", light_t]]
    if species.lower() in ['proton', 'protons', 'p']:
        table_data = [["Distance of SolO from the Sun", "[AU]", dist],
                    ["Length of the Parker Spiral for 400 km/s sw ", "[AU]", spiral_len],
                    ["Travel time of 4 keV protons ", "[min]", traveltime_min],
                    ["Travel time of 100 MeV protons ", "[min]", traveltime_max],
                    ["Travel time of light ", "[min]", light_t]]        
    print(tabulate(table_data))
    return(table_data)

def extract_electron_data(df_electrons, df_energies, plotstart, plotend,  t_inj, bgstart = None, bgend = None, bg_distance_from_window = '2h', bg_period = '60min', travel_distance = 0,  travel_distance_second_slope = None, fixed_window = None, instrument = 'ept', data_type = 'l2', averaging=None, masking=True, ion_conta_corr=False, df_protons = None, centre_pix = False):
    """This function determines an energy spectrum from time series data for any of the Solar Orbiter / EPD 
    sensors uses energy-dependent time windows to determine the flux points for the spectrum. 
    The dependence is determined according to an expected velocity dispersion assuming a certain 
    solar injection time (t_inj) and a traval distance (travel_distance).

    CHANGES 1.08: THE DISTANCE IN MINUTES AND RESOLUTION FUNCTIONS CHANGED TO PANDAS TO_TIMEDELTA AND RESAMPLE. THESE WORK WITH PANDAS TIME STRINGS.
    CHANGE ALSO DOCTRINGS.

    ALSO NEED TO DELETE ROLLING OPTION AND MAKE SURE THE NONE (NO AVERAGING) OPTION WORKS
    AVERAGING MODE KEY WORD SHOULD BE DELETED ENTIRELY. EITHER MEAN ALWAYS WHEN THE AVERAGING IS DONE OR IF AVERAGING IS NONE THEN NO AVERAGING WILL BE DONE


    Args:
        df_electrons (pandas DataFrame): contains electron data 
        df_energies (pandas DataFrame): contains information about the energy channels of both proton (for EPT and HET) and electron data
        plotstart (string): start time of the time series plot, e.g. '2020-11-18-0000'
        plotend (string): end time of the time series plot, e.g., '2020-11-18-2230'
        t_inj (string): solar injection time e.g. '2020-11-18-1230'
        bgstart (string, optional): start time of the background window. e.g., '2020-11-18-1030'
                If specified, specify also bgend. By specifying bgstart and bgend the bg window 
                will be fixed. Defaults to None. Leave to None for a moving bg and specify 
                bg_distance_from_window and bg_period.
        bgend (string, optional): end time of the background window. e.g., '2020-11-18-1130'
                If specified, specify also bgstart. By specifying bgstart and bgend the bg window 
                will be fixed. Defaults to None. Leave to None for a moving bg and specify 
                bg_distance_from_window and bg_period.
        bg_distance_from_window (str, optional):  This is the distance of the 
                end time of the background window from the start of the peak search window.
                Follows the velocity dispersion (first slope). If specified, specify also bg_period.
                Defaults to None. Leave to None for a fixed window and specify bgstart and bgend. 
        bg_period (str, optional): This is the duration of the backdround window 
                in minutes. If specified, specify also bg_distance_from_window. Defaults to None.
                Leave to None for a fixed window and specify bgstart and bgend. 
                Possible values: 
                'W'
                'D' / 'days' / 'day'
                'hours' / 'hour' / 'hr' / 'h'
                'm' / 'minute' / 'min' / 'minutes' / 'T'
                'S' / 'seconds' / 'sec' / 'second'
                'ms' / 'milliseconds' / 'millisecond' / 'milli' / 'millis'/ 'L'
                'us' / 'microseconds' / 'microsecond' / 'nicro' / 'micros' / 'U'
                'ns' / 'nanoseconds' / 'nano' / 'nanos' / 'nanosecond' / 'N'

        travel_distance (float, optional): input in AU. The travel distance calculated with
                velocity dispersion analysis. This value is used to calculate the 
                peak search window starting time which is different for each energy channel.
                Follows the velocity dispersion. Defaults to 0. If left to 0 the search 
                window will be fixed.
        travel_distance_second_slope (float, optional): input in AU. Travel distance to calculate 
                a second slope for the peak window search end time. The flux peak can get broader
                at lower energies and with a fixed time window it can be hard to determine
                the flux peak if two events are close to each other with strong velocity dispersion.
                Defaults to None. If None the peak search window will have a fixed time period. 
                Either specify travel_distance_second_slope or fixed_window.
        fixed_window (str, optional): This is the length of the search window
                in minutes. Defaults to None. Either specify travel_distance_second_slope 
                or fixed_window.
                Possible values: 
                'W'
                'D' / 'days' / 'day'
                'hours' / 'hour' / 'hr' / 'h'
                'm' / 'minute' / 'min' / 'minutes' / 'T'
                'S' / 'seconds' / 'sec' / 'second'
                'ms' / 'milliseconds' / 'millisecond' / 'milli' / 'millis'/ 'L'
                'us' / 'microseconds' / 'microsecond' / 'nicro' / 'micros' / 'U'
                'ns' / 'nanoseconds' / 'nano' / 'nanos' / 'nanosecond' / 'N'

        instrument (str, optional): 'ept', 'het', or 'step'. Defaults to 'ept'.
        data_type (str, optional): which data level (e.g., low latency (ll) or level2 (l2)) is used. 
                This affects the number of energy channels. Defaults to 'l2'.
        averaging (str, optional): Defaults to None.
        Possible values: 
                'W'
                'D' / 'days' / 'day'
                'hours' / 'hour' / 'hr' / 'h'
                'm' / 'minute' / 'min' / 'minutes' / 'T'
                'S' / 'seconds' / 'sec' / 'second'
                'ms' / 'milliseconds' / 'millisecond' / 'milli' / 'millis'/ 'L'
                'us' / 'microseconds' / 'microsecond' / 'nicro' / 'micros' / 'U'
                'ns' / 'nanoseconds' / 'nano' / 'nanos' / 'nanosecond' / 'N'

        masking (bool, optional): Refers only to STEP data. If true, time intervals with significant 
                (5 sigma) ion contamination are masked. Defaults to False.
        ion_conta_corr (bool, optional): Refers only to EPT data. If true, ion contamination
                correction is applied. Defaults to False.
        df_protons (pandas DataFrame, optional): contains proton (ion) data. Use only with EPT and HET data.
                Defauts to None. 

    Raises:
        Exception: If either bgstart or bgend are not None (so a value has been specified)
                and also bg_distance_from_window or bg_period are not None, this will raise an error.
                Either specify bgstart and bgend for a fixed background OR specify bg_distance_from_window
                and bg_period for a shifting background.

    Returns:
        df_electron_fluxes: pandas DataFrame
        df_info : pandas DataFrame. This data frame contains the spectrum data 
                and all its metadata (which is saved to csv in the function write_to_csv())
        [searchstart, searchend]: list of strings. The search window start and end times.
        [e_low, e_high] : list of float. The lowest and highest energy corresponding to 
                each energy channel.
        [instrument, data_type] : list of strings.

    """

    if bgstart is not None or bgend is not None: 
        if bg_distance_from_window is not None or bg_period is not None:
            raise Exception("Please specify either bg_start and bg_end or bg_distance_from_window and bg_period.")
        
    if bgstart is None or bgend is None: 
        if bg_distance_from_window is None or bg_period is None:
            raise Exception("Please specify either bg_start and bg_end or bg_distance_from_window and bg_period.")
    
        
    
    # Takes proton and electron flux and uncertainty values from original data.
    if(instrument != 'step'):
        df_electron_fluxes = df_electrons['Electron_Flux'][plotstart:plotend]
        df_electron_uncertainties = df_electrons['Electron_Uncertainty'][plotstart:plotend]

    if(instrument == 'ept'):
        df_proton_fluxes = df_protons['Ion_Flux'][plotstart:plotend]
        df_proton_uncertainties = df_protons['Ion_Uncertainty'][plotstart:plotend]

        if(data_type == 'll'):
            channels = range(len(df_energies['Electron_Bins_Low_Energy']))
            e_low = df_energies['Electron_Bins_Low_Energy']
            e_high = []

            for i in channels:
                e_high.append(e_low[i]+df_energies['Electron_Bins_Width'][i])
                df_electron_fluxes = df_electron_fluxes.rename(columns={'Ele_Flux_{}'.format(i):'Electron_Flux_{}'.format(i)})
                df_electron_uncertainties = df_electron_uncertainties.rename(columns={'Ele_Flux_Sigma_{}'.format(i):'Electron_Uncertainty_{}'.format(i)})


        elif(data_type == 'l2'):
            channels = range(len(df_energies['Electron_Bins_Low_Energy']))
            e_low = df_energies['Electron_Bins_Low_Energy']
            e_high = []
            
            for i in channels:
                e_high.append(e_low[i]+df_energies['Electron_Bins_Width'][i])
                

            
    elif(instrument == 'het'):

        if(data_type == 'll'):

            e_low = df_energies['Electron_Bins_Low_Energy']
            e_high = []

            channels = range(len(df_energies['Electron_Bins_Low_Energy']))
            
            for i in channels:
                e_high.append(e_low[i]+df_energies['Electron_Bins_Width'][i])
                df_electron_fluxes = df_electron_fluxes.rename(columns={'Ele_Flux_{}'.format(i):'Electron_Flux_{}'.format(i)})
                df_electron_uncertainties = df_electron_uncertainties.rename(columns={'Ele_Flux_Sigma_{}'.format(i):'Electron_Uncertainty_{}'.format(i)})


        elif(data_type == 'l2'):
            channels = range(len(df_energies['Electron_Bins_Low_Energy']))
            e_low = df_energies['Electron_Bins_Low_Energy']
            e_high = []

            for i in channels:
                e_high.append(e_low[i]+df_energies['Electron_Bins_Width'][i])

    
    elif(instrument == 'step'):
        if(data_type == 'l2'):
            e_low = df_energies['Bins_Low_Energy']
            e_high = []

            channels = range(len(df_energies['Bins_Low_Energy']))
            for i in channels:
                    e_high.append(e_low[i]+df_energies['Bins_Width'][i])
                    

    
            if 'Electron_Avg_Flux_0' in df_electrons.columns:
                df_electron_fluxes = pd.DataFrame()
                df_electron_uncertainties = pd.DataFrame()

                for i in channels:
                    e_high.append(e_low[i]+df_energies['Bins_Width'][i])

                    if centre_pix:
                        df_electron_fluxes['Electron_Flux_'+str(i)] = df_electrons['Electron_Comb_Flux_'+str(i)][plotstart:plotend]
                        df_electron_uncertainties['Electron_Uncertainty_'+str(i)] = df_electrons['Electron_Comb_Uncertainty_'+str(i)][plotstart:plotend]


                    else:
                        df_electron_fluxes['Electron_Flux_'+str(i)] = df_electrons['Electron_Avg_Flux_'+str(i)][plotstart:plotend]
                        df_electron_uncertainties['Electron_Uncertainty_'+str(i)] = df_electrons['Electron_Avg_Uncertainty_'+str(i)][plotstart:plotend]


            else:
                # check how the pix works for old events
                step_data = make_step_electron_flux(df_electrons, mask_conta=masking)
                
                df_electron_fluxes = step_data[0][plotstart:plotend]
                df_electron_uncertainties = step_data[1][plotstart:plotend]



        # Cleans up negative flux values in STEP data.
        df_electron_fluxes[df_electron_fluxes<0] = np.NaN

    if(averaging != None ):
        if(instrument=='ept'):
            df_proton_fluxes =df_proton_fluxes.resample(averaging).mean()
            df_proton_uncertainties = df_proton_uncertainties.resample(averaging).apply(average_flux_error)
            
# The data product changed so the first energy channel was set to nan. That messes with the matrix calculation of the ion contamination correction.
# The issue was fixed by not using matmul and using a few extra steps and masking the nan data.
         
        # for STEP electrons, the resampling is done independently, e.g. solo_epd_loader.calc_electrons(df, resamle='1min')
        if(instrument!='step'):
            df_electron_fluxes = df_electron_fluxes.resample(averaging).mean()
            df_electron_uncertainties = df_electron_uncertainties.resample(averaging).apply(average_flux_error)
            

    # 12.07.2023 The rolling window option should be deleted because it is never used. 
    # The rolling window might be broken, but it's not ever used.
    #elif(averaging_mode == 'rolling_window'):
        # for STEP electrons, the resampling is done independently, but rolling_window is not supported!
     #   if(instrument!='step'):
      #      df_electron_fluxes = df_electron_fluxes.rolling(window=averaging, min_periods=1).mean()


    if(ion_conta_corr and (instrument == 'ept')):

        ion_cont_corr_matrix = np.loadtxt('EPT_ion_contamination_flux_paco.dat')
        Electron_Flux_cont = np.zeros(np.shape(df_electron_fluxes))
        Electron_Uncertainty_cont = np.zeros(np.shape(df_electron_uncertainties))
        
        for tt in range(len(df_electron_fluxes)):
            Electron_Flux_cont[tt,:] = np.sum(ion_cont_corr_matrix * np.ma.masked_invalid(df_proton_fluxes.values[tt, :]), axis=1)
            # the matrix multiplication np.matmul does not work if there are nan vales in the matrix because it does not have an inbuilt ignore nan variable
            # so for now we can ignore nans by using the above more 'by hand' calculation with np.ma.masked_invalid that ignore both inf and nan values
            # Electron_Flux_cont[tt, :] = np.matmul(ion_cont_corr_matrix, np.ma.masked_invalid(df_proton_fluxes.values[tt, :]))
            Electron_Uncertainty_cont[tt, :] = np.sqrt(np.matmul(ion_cont_corr_matrix**2, np.ma.masked_invalid(df_proton_uncertainties.values[tt, :]**2 )))
            
        df_electron_fluxes = df_electron_fluxes - Electron_Flux_cont
        df_electron_uncertainties = np.sqrt(df_electron_uncertainties**2 + Electron_Uncertainty_cont**2 )
    

    if(instrument=='ept'):
        ion_string = 'Ion_contamination_correction'

    elif(instrument=='step'):
        ion_string = 'Ion_masking'

    elif(instrument=='het'):
        ion_string = ''
    
    # Main information dataframe containing most of the required data.
    #df_info = pd.DataFrame({'Plot_period':[], 'Search_period':[], 'Bg_period':[], 'Averaging':[], '{}'.format(ion_string):[], 'Energy_channel':[], 'Primary_energy':[], 'Energy_error_low':[], 'Energy_error_high':[], 'Peak_timestamp':[], 'Flux_peak':[], 'Peak_significance':[], 'Peak_electron_uncertainty':[], 'Background_flux':[],'Bg_electron_uncertainty':[], 'Bg_subtracted_peak':[], 'Backsub_peak_uncertainty':[], 'rel_backsub_peak_err':[], 'frac_nonan':[]})
    #df_info = pd.DataFrame({'Plot_period':[], 'Averaging':[], '{}'.format(ion_string):[], 'Energy_channel':[], 'Primary_energy':[], 'Energy_error_low':[], 'Energy_error_high':[], 'Peak_timestamp':[], 'Flux_peak':[], 'Peak_significance':[], 'Peak_electron_uncertainty':[], 'Background_flux':[],'Bg_electron_uncertainty':[], 'Bg_subtracted_peak':[], 'Backsub_peak_uncertainty':[], 'rel_backsub_peak_err':[], 'frac_nonan':[]})
    df_info = pd.DataFrame({'Plot_period':[], 'Averaging':[], '{}'.format(ion_string):[], 'Energy_channel':[], 'Primary_energy':[]})
    
    # Adds basic metadata to main info df.
    df_info['Plot_period'] = [plotstart]+[plotend]+['']*(len(channels)-2)
    #df_info['Search_period'] = [searchstart]+[searchend]+['']*(len(channels)-2)
    #df_info['Bg_period'] = [bgstart]+[bgend]+['']*(len(channels)-2)

    if(instrument=='ept'):
        df_info['Ion_contamination_correction'] = [ion_conta_corr]+['']*(len(channels)-1)

    elif(instrument=='step'):
        df_info['Ion_masking'] = [masking]+['']*(len(channels)-1)

    if(averaging is None):
        df_info['Averaging'] = ['No averaging']+['']*(len(channels)-1)

    #elif(averaging_mode == 'rolling_window'):
     #   df_info['Averaging'] = ['Rolling window', 'Window size = ' + str(averaging)] + ['']*(len(channels)-2)

    elif(averaging is not None):
        df_info['Averaging'] = ['Mean', 'Resampled to ' + averaging] + ['']*(len(channels)-2)

 
    # Energy bin primary energies; geometric mean.
    # Are be used to calculate beta and velocity of particles.

    primary_energies = []

    for i in range(0,len(e_low)):
        primary_energies.append(np.sqrt(e_low[i]*e_high[i]))

    primary_energies_channels = []

    for energy in channels:
        primary_energies_channels.append(primary_energies[energy])

    df_info['Primary_energy'] = primary_energies_channels

    # Calculates energy errors for spectrum plot.
    energy_error_low = []
    energy_error_high = []

    for i in range(0,len(primary_energies)):

        energy_error_low.append(primary_energies[i]-e_low[i])
        energy_error_high.append(e_high[i]-primary_energies[i])


    energy_error_low_channels = []
    energy_error_high_channels = []

    for i in channels:

        energy_error_low_channels.append(energy_error_low[i])
        energy_error_high_channels.append(energy_error_high[i])

    df_info['Energy_error_low'] = energy_error_low_channels
    df_info['Energy_error_high'] = energy_error_high_channels

    # Calculating plasma beta and velocity with kinetic energy (primary energy)
    # the velocity is in km/s
    velocity = []

    for energy in primary_energies:
        velocity.append(evolt2speed(energy, 2))


    # Using calculated velocity to find the right search period
    # the travel distance from au to km
    travel_distance = travel_distance*1.496E8
    
    DV = []
    
    for v in velocity:
        DV.append(travel_distance/v)
    
    searchstart = []
    
    for i in DV:
        searchstart.append(pd.to_datetime(t_inj)+pd.Timedelta(seconds = i))
        
    searchend = []
    
    # same thing for second slope if fixed_window = None (to find searchend)
    if fixed_window is None:
        travel_distance_second_slope = travel_distance_second_slope*1.496E8

        DV2 = []
    
        for v in velocity:
            DV2.append(travel_distance_second_slope/v)

        searchend = []
   
        for i in DV2:
            searchend.append(pd.to_datetime(t_inj)+pd.Timedelta(seconds = i))
            

    if fixed_window is not None:
        for i in searchstart:
            searchend.append(i+pd.to_timedelta(fixed_window))
            
        
    if bg_distance_from_window is None:
        bg_start = bgstart
        bg_end = bgend
        bgstart = []
        bgend   = []
        for i in range(0, len(searchstart)):
            bgstart.append(bg_start)
            bgend.append(bg_end)
    
    if bg_distance_from_window is not None:
        bgstart = []
        bgend   = []
        for i in range(0,len(searchstart)):
            bgend.append(searchstart[i]-pd.to_timedelta(bg_distance_from_window))
            bgstart.append(bgend[i]-pd.to_timedelta(bg_period))
            #bgend.append(bgstart[i]+pd.to_timedelta(bg_period))

    # Next blocks of code calculate information from data and append them to main info df.
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
    #list_average_electron_uncertainties = [] change to new unc determination later

    n = 0
     
    for channel in channels:
        b_f = df_electron_fluxes['Electron_Flux_{}'.format(channel)][searchstart[n]:searchend[n]]
        
        # This is where I check if the BG is zero. Can temporarely change this. This was if len(b_f) ==0: bg_flux = np.nan list_bg_fluxes.append(bg_flux) Change back when needed
        if len(b_f) ==0:
            bg_flux = np.nan
            #bg_flux = df_electron_fluxes['Electron_Flux_{}'.format(channel)][bgstart[n]:bgend[n]].min()
            list_bg_fluxes.append(bg_flux)
        if len(b_f)!= 0:
            bg_flux = df_electron_fluxes['Electron_Flux_{}'.format(channel)][bgstart[n]:bgend[n]].mean(skipna=True)
            list_bg_fluxes.append(bg_flux)
        
        f_p = df_electron_fluxes['Electron_Flux_{}'.format(channel)][searchstart[n]:searchend[n]]
        if len(f_p) == 0 :
            flux_peak = np.nan
        if len(f_p) != 0:
            flux_peak = df_electron_fluxes['Electron_Flux_{}'.format(channel)][searchstart[n]:searchend[n]].max()
        list_flux_peaks.append(flux_peak)
        

        # check if a large enough fraction of data points are not nan. If there are too many nan's in the search time interval, frac_nonan can be used to exclude the channel from the spectrum
        frac_nonan = 1 - np.sum(np.isnan(f_p)) / len(f_p) # fraction of data in interval that is NOT nan
        list_frac_nonan.append(frac_nonan)
            
        p_t = df_electron_fluxes['Electron_Flux_{}'.format(channel)][searchstart[n]:searchend[n]]
        if len(p_t) == 0:
            peak_timestamp = np.nan
            list_peak_timestamps.append(peak_timestamp)
        if len(p_t) != 0:
            peak_timestamp = df_electron_fluxes['Electron_Flux_{}'.format(channel)][searchstart[n]:searchend[n]].idxmax(skipna = True)
            list_peak_timestamps.append(peak_timestamp)
        
        t_l = df_electron_uncertainties['Electron_Uncertainty_{}'.format(channel)]
        
        # First finding the index location of the peak timestamp in uncertainty dataframe and the getting value of that index location.
        if pd.isna(peak_timestamp):
            list_peak_electron_uncertainties.append(np.nan)
        if len(t_l) == 0:
            list_peak_electron_uncertainties.append(np.nan)
        if len(t_l)!= 0 and pd.isna(peak_timestamp)==False:
            timestamp_loc = df_electron_uncertainties['Electron_Uncertainty_{}'.format(channel)].index.get_loc(peak_timestamp, method='nearest')
            peak_electron_uncertainty = df_electron_uncertainties['Electron_Uncertainty_{}'.format(channel)].iloc[timestamp_loc]
            list_peak_electron_uncertainties.append(peak_electron_uncertainty)

        average_bg_uncertainty = np.sqrt((df_electron_uncertainties['Electron_Uncertainty_{}'.format(channel)]
                                          [bgstart[n]:bgend[n]]**2).sum(axis=0))/len(df_electron_uncertainties['Electron_Uncertainty_{}'.format(channel)][bgstart[n]:bgend[n]])
        list_average_bg_uncertainties.append(average_bg_uncertainty)

        bg_std = df_electron_fluxes['Electron_Flux_{}'.format(channel)][bgstart[n]:bgend[n]].std()
    
        list_bg_std.append(bg_std)
        
        f_a = df_electron_fluxes['Electron_Flux_{}'.format(channel)][searchstart[n]:searchend[n]]
        if len(f_a) == 0:
            flux_average = np.nan
            list_flux_average.append(flux_average)
        if len(f_a) != 0:
            flux_average = df_electron_fluxes['Electron_Flux_{}'.format(channel)][searchstart[n]:searchend[n]].mean(skipna=True)
            list_flux_average.append(flux_average)
        

        n = n+1

    
    for i in range(0,len(list_flux_peaks)):

        list_bg_subtracted_peaks.append(list_flux_peaks[i]-list_bg_fluxes[i])


        list_peak_significance.append(list_bg_subtracted_peaks[i]/list_bg_std[i])
        #sometimes the background can be higher than the peak to need to delete those values (set to nan)
        if list_bg_subtracted_peaks[i]<list_bg_fluxes[i]:
             list_peak_significance[i] = -1

        list_bg_subtracted_average.append(list_flux_average[i]-list_bg_fluxes[i])
        list_average_significance.append(list_bg_subtracted_average[i]/list_bg_std[i])
        #sometimes the background can be higher than the peak to need to delete those values (set to nan)
        if list_bg_subtracted_average[i]<list_bg_fluxes[i]:
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
    #df_info['Average_electron_uncertainty'] = list_average_electron_uncertainties  change to new unc determination later
    df_info['Bg_subtracted_average'] = list_bg_subtracted_average
    df_info['Average_significance'] = list_average_significance
    
    df_info['Backsub_peak_uncertainty'] = np.sqrt(df_info['Peak_electron_uncertainty']**2 + df_info['Bg_electron_uncertainty']**2)
    df_info['rel_backsub_peak_err'] = np.abs(df_info['Backsub_peak_uncertainty'] / df_info['Bg_subtracted_peak'])
    df_info['frac_nonan'] = list_frac_nonan

    

    return df_electron_fluxes, df_info, [searchstart, searchend], [e_low, e_high], [instrument, data_type]



def make_step_electron_flux(stepdata, mask_conta=True):
    """
    We use the calibration factors from Paco (Alcala) to calculate the electron flux 
    out of the (integral - magnet) fluxes (we now use level2 data files to get these)
    we also check if the integral counts are sufficiently higher than the magnet counts 
    so that we can really assume it's electrons (otherwise we mask the output arrays)
    As suggested by Alex Kollhoff & Berger use a 5 sigma threshold:
    C_INT >> C_MAG:
    C_INT - C_MAG > 5*sqrt(C_INT)
    
    Args:
        stepdata (pandas dataframe): STEP data
        mask_conta (bool, optional): If true, time intervals with significant 
                (5 sigma) ion contamination are masked. Defaults to True. 

    Returns:
        df_electron_fluxes (pandas dataframe): electron flux data
        df_electron_uncertainties
        paco.E_low
        paco.E_hi

    """

    # calculate electron flux from F_INT - F_MAG:
    colnames = ["ch_num", "E_low", "E_hi", "factors"]
    paco = pd.read_csv('step_electrons_calibration.csv', names=colnames, skiprows=1)
    paco.E_low = round(paco.E_low/1000, 5)
    paco.E_hi = round(paco.E_hi/1000, 5)

    F_INT = stepdata['Integral_Flux']
    F_MAG = stepdata['Magnet_Flux']
    step_flux =  (F_INT - F_MAG) * paco.factors.values
    U_INT = stepdata['Integral_Uncertainty']
    U_MAG = stepdata['Magnet_Uncertainty']
    # from Paco:
    # Ele_Uncertainty = k * sqrt(Integral_Uncertainty^2 + Magnet_Uncertainty^2)
    step_unc = np.sqrt(U_INT**2 + U_MAG**2) * paco.factors.values
    param_list = ['Electron_Flux', 'Electron_Uncertainty']

    if mask_conta:

        # C_INT = stepdata['Integral_Rate']
        # C_MAG = stepdata['Magnet_Rate']
        # clean = (C_INT - C_MAG) > 5*np.sqrt(C_INT)
        # step_flux = step_flux.mask(clean)
        # step_unc = step_unc.mask(clean)
        clean = (F_INT-F_MAG)> 2 * U_INT # call 2 conta_threshold
        step_flux = step_flux.mask(~clean)
        step_unc = step_unc.mask(~clean)    
        
    step_data = pd.concat([step_flux, step_unc], axis=1, keys=param_list)

    df_electron_fluxes = step_data['Electron_Flux']
    df_electron_uncertainties = step_data['Electron_Uncertainty']

    for channel in df_electron_fluxes:

        df_electron_fluxes = df_electron_fluxes.rename(columns={channel:'Electron_Flux_{}'.format(channel)})

    for channel in df_electron_uncertainties:

        df_electron_uncertainties = df_electron_uncertainties.rename(columns={channel:'Electron_Uncertainty_{}'.format(channel)})

    return df_electron_fluxes, df_electron_uncertainties, paco.E_low, paco.E_hi

def average_flux_error(flux_err: pd.DataFrame) -> pd.Series:

    return np.sqrt((flux_err ** 2).sum(axis=0)) / len(flux_err.values)

def plot_channels(args, bg_subtraction=False, savefig=False, sigma=3, path='', key='', frac_nan_threshold=0.4, rel_err_threshold=0.5, plot_pa=False, coverage=None, sensor = 'ept', viewing='sun', centre_pix = False, date = None):
    """Creates a timeseries plot showing the particle flux for each energy channel of
        the instrument (STEP, EPT, HET). The timeseries plot shows also the peak window and
        background window. The peak is marked with different color lines:
        green: peak is ok
        grey: too many nans in window
        blue: low sigma
        orange: high relative error

    Args:
        args : Output of the extract_data function. Incudes:
                df_electron_fluxes: pandas DataFrame
                df_info : pandas DataFrame. This data frame contains the spectrum data 
                and all its metadata (which is saved to csv in the function write_to_csv())
                [searchstart, searchend]: list of strings. The search window start and end times.
                [e_low, e_high] : list of float. The lowest and highest energy corresponding to 
                each energy channel.
                [instrument, data_type] : list of strings.
        bg_subtraction (bool, optional): Subtract bg from data. Defaults to False.
        savefig (bool, optional): saving the timeseries plot. Defaults to False.
        sigma (int, optional): sigma threshold value. Is used to check if the sigma value is 
                high enough fro the data within the search-period interval. If not, the flux and 
                uncertainty value of that energy channel are set to nan and therefore 
                excluded from the spectrum. Defaults to 3.
        path (str, optional): path to folder where the timeseries will be saved. Defaults to ''.
        key (str, optional): _description_. Defaults to ''.
        frac_nan_threshold (float, optional):  is used to to check if there is enough non-nan 
                flux data points in the search-period interval. If not, the flux and 
                uncertainty value of that energy channel are set to nan and therefore 
                excluded from the spectrum. Defaults to 0.4.
        rel_err_threshold (float, optional): is used to check that relative error is 
                low enough in the search period interval. If not, the flux and 
                uncertainty value of that energy channel are set to nan and therefore 
                excluded from the spectrum. Defaults to 0.5.
        plot_pa (bool, optional): include pitch angles in the plot. Defaults to False.
        coverage (pandas dataframe or None, optional): dataframe to be used to plot the pitch angles. Defaults to None.
        sensor (str, optional): sensor used for plotting the pitch angles. Defaults to 'ept'.
        viewing (str, optional): viewing direction of EPT or HET, used for plotting the pitch angles of these telescopes. Defaults to 'sun'. Is ignored if sensor=='step'
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
    

    if viewing is None or sensor in ['STEP', 'step']:
        viewing = 'sun'

    title_string = instrument.upper() + ', ' + data_type.upper() + ', ' + date_string
    filename = 'channels-' + file_date + '-' + instrument.upper() + '-' +viewing+ '-' + data_type.upper() 
    
    if(df_info['Averaging'][0]=='Mean'):

        title_string = title_string + ', ' + df_info['Averaging'][1].split()[2] + ' averaging'
        filename = filename + '-' + df_info['Averaging'][1].split()[2] + '_averaging'

    elif(df_info['Averaging'][0]=='No averaging'):

        title_string = title_string + ', no averaging'
        filename = filename + '-no_averaging'

    if(bg_subtraction):
        
       title_string = title_string + ', bg subtraction on'
       filename = filename + '-bg_subtr'

    else:

        title_string = title_string + ', bg subtraction off'
    
    if(instrument == 'ept'):
        
        if(df_info['Ion_contamination_correction'][0]):

            title_string = title_string + ', ion correction on'
            filename = filename + '-ion_corr'

        elif(df_info['Ion_contamination_correction'][0]==False):

            title_string = title_string + ', ion correction off'

    if instrument == 'step' and centre_pix:
        filename = filename + '-centre_pix'
        title_string = title_string + ', centre pix'


    # If background subtraction is enabled, subtracts bg_flux from all observations. If flux value is negative, changes it to NaN.
    if(bg_subtraction == False):
        pass
    elif(bg_subtraction == True):
        df_electron_fluxes = df_electron_fluxes.sub(df_info['Background_flux'].values, axis=1)
        df_electron_fluxes[df_electron_fluxes<0] = np.NaN

    # Plotting part.
    # Initialized the main figure.
    # fig = plt.figure()
    color = {'sun':'crimson','asun':'orange', 'north':'darkslateblue', 'south':'c'}
    npanels = len(df_info['Energy_channel'])
    if plot_pa: 
        npanels = npanels + 1

    if sensor == 'step':
        fsize = (20,60)
    if sensor == 'ept':
        fsize = (20,48)
    if sensor == 'het':
        fsize = (20,12)
    fig, axes = plt.subplots(npanels, sharex=True, figsize=fsize)
    # plt.xticks([])
    # plt.yticks([])
    # plt.ylabel("Flux \n [1/s cm$^2$ sr MeV]", labelpad=40)
    fig.supylabel("Flux [1/s cm$^2$ sr MeV]", size=20)
    axes[0].set_title(title_string, size=20)


    # Loop through selected energy channels and create a subplot for each.
    n=0
    for channel in df_info['Energy_channel']:
        #ax = fig.add_subplot(npanels,1,n)
        #ax = df_electron_fluxes['Electron_Flux_{}'.format(channel)].plot(logy=True, figsize=fsize, color=color[viewing], drawstyle='steps-mid')
        ax = axes[n]
        ax.plot(df_electron_fluxes.index, df_electron_fluxes['Electron_Flux_{}'.format(channel)], color=color[viewing], drawstyle='steps-mid')
        ax.set_yscale('log')
        plt.text(0.025,0.7, str(energy_bin[0][channel]) + " - " + str(energy_bin[1][channel]) + " MeV", transform=ax.transAxes, size=13)

        # Search area vertical lines.
        ax.axvline(search_area[0][n], color='black')
        ax.axvline(search_area[1][n], color='black')
        ax.set_xlim(df_electron_fluxes.index[0], df_electron_fluxes.index[-1])
        
        # Peak vertical line.
        if df_info['Peak_timestamp'][n] is not pd.NaT:
            if  (rel_err[n] > rel_err_threshold): # if the relative error too large, we exlcude the channel
                ax.axvline(df_info['Peak_timestamp'][n], linestyle=':', linewidth=4, color='orange')
            if df_info['frac_nonan'][n] < frac_nan_threshold:  # we only plot a line if the fraction of non-nan data points in the search interval is larger than frac_nan_threshold
                ax.axvline(df_info['Peak_timestamp'][n], linestyle='--', linewidth=3, color='gray')
            if (peak_sig[n] < sigma): # if the peak is not significant, we discard the energy channel
                ax.axvline(df_info['Peak_timestamp'][n], linestyle='-.', linewidth=2, color='blue')
            if (peak_sig[n] >= sigma) and (rel_err[n] <= rel_err_threshold) and (df_info['frac_nonan'][n] > frac_nan_threshold):
                ax.axvline(df_info['Peak_timestamp'][n], color='green')
            if bg_subtraction == True:
                if (np.isnan(peak_sig[n]))  and (~np.isnan(df_info['Bg_subtracted_peak'][n])): # no background
                    ax.axvline(df_info['Peak_timestamp'][n], linestyle='-', linewidth=2, color='purple')
            if bg_subtraction == False:
                if (np.isnan(peak_sig[n]))  and df_info['Flux_average'][n]!=0.: # no background
                    ax.axvline(df_info['Peak_timestamp'][n], linestyle='-', linewidth=2, color='purple')
            

        # Background measurement area.
        ax.axvspan(df_info['Bg_start'][n], df_info['Bg_end'][n], color='gray', alpha=0.25)

        ax.get_xaxis().set_visible(False)

        if(n == len(df_info['Energy_channel'])-1 and plot_pa==False):

            ax.get_xaxis().set_visible(True)
            ax.set_xlabel("Time", labelpad=45)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
            #ax.xaxis.set_minor_locator(hours)

        n+=1
    if plot_pa:  # add a panel that shows the pitch angle of the telescope
        # ax = fig.add_subplot(npanels,1,n)
        ax = axes[n]
        if sensor in ['HET', 'het', 'EPT', 'ept']: 
            #for direction in ['sun', 'asun', 'north', 'south']: 
            col = color[viewing]
            # fill the minimum-maximum range of the pitch angle coverage
            ax.fill_between(coverage.index, coverage[viewing]['min'], coverage[viewing]['max'], alpha=0.5, color=col, edgecolor=col, linewidth=0.0, step='mid')
            # plot the central pitch angle as a thin line
            ax.plot(coverage.index, coverage[viewing]['center'], linewidth=0.7, label=viewing, color=col, drawstyle='steps-mid')

        if sensor in ['STEP', 'step']:
            col_list = plt.cm.viridis(np.linspace(0.,0.95,16))
            for p in range(1, 16):  # loop over 15 sectors/pixels
                # plot the central pitch angle as a thin line
                ax.plot(coverage.index, coverage[f'Pixel_{p}']['center'], color = col_list[p-1], linewidth=1, label=f'Pixel_{p}', drawstyle='steps-mid')

        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')

       
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=instrument)
        ax.set_ylim([0, 180])
        ax.yaxis.set_ticks(np.arange(0, 180+45, 45))
        ax.set_ylabel('PA / °', size=13)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        plt.tick_params(axis='x', which='major', labelsize=16)
        plt.tick_params(axis='y', which='major', labelsize=13)
        ax.set_xlabel("Time", labelpad=45, size=16)
    
    # Saves figure, if enabled.
    if(path[len(path)-1] != '/'):

        path = path + '/'

    if(savefig):

        plt.savefig(path + filename + str(key) +'.jpg', bbox_inches='tight')

    plt.show()

# This plot_check function is not finished, but it does produce cool rainbow coloured plots.
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

def plot_spectrum_peak(args, bg_subtraction=True, savefig=False, path='', key='', sigma=3, frac_nan_threshold=0.4, rel_err_threshold=0.5, direction=None, centre_pix = False, date = None):
    """_summary_

    Args:
        args (_type_): _description_
        bg_subtraction (bool, optional): _description_. Defaults to True.
        savefig (bool, optional): _description_. Defaults to False.
        path (str, optional): _description_. Defaults to ''.
        key (str, optional): _description_. Defaults to ''.
        sigma (int, optional): _description_. Defaults to 3.
        frac_nan_threshold (float, optional): _description_. Defaults to 0.4.
        rel_err_threshold (float, optional): _description_. Defaults to 0.5.
        direction (_type_, optional): _description_. Defaults to None.
    """
    color = {'sun':'crimson','asun':'orange', 'north':'darkslateblue', 'south':'c'}
    df_info = args[1]
    instrument = args[4][0]
    data_type = args[4][1]
    if direction is None or instrument in ['STEP', 'step']:
        viewing = 'sun'
        direction = 'sun'
    else:
        viewing = f'-{direction}' 

    date_string = ''
    file_date = ''

    if date is None:
        date_string = str(df_info['Plot_period'][0][:-5])
        file_date = str(df_info['Plot_period'][0][:-5])

    else:
        date_string = str(date)[:-3]
        file_date = str(date)[:-3].replace(' ', '-').replace(':', '')
    
    title_string = instrument.upper() + ', ' + data_type.upper() + ', ' + date_string
    filename = 'spectrum-' + file_date + '-' + instrument.upper() + viewing+ '-' + data_type.upper() 
    
    if(df_info['Averaging'][0]=='Mean'):

        title_string = title_string + ', ' + df_info['Averaging'][1].split()[2] + ' averaging'
        filename = filename + '-' + df_info['Averaging'][1].split()[2] + '_averaging'

    elif(df_info['Averaging'][0]=='No averaging'):

        title_string = title_string + ', no averaging'
        filename = filename + '-no_averaging'

    if(bg_subtraction):
        
       title_string = title_string + ', bg subtraction on'
       filename = filename + '-bg_subtr'

    else:

        title_string = title_string + ', bg subtraction off'
    
    if(instrument == 'ept'):

        if(df_info['Ion_contamination_correction'][0] and instrument=='ept'):

            title_string = title_string + ', ion correction on'
            filename = filename + '-ion_corr'

        elif(df_info['Ion_contamination_correction'][0]==False):

            title_string = title_string + ', ion correction off'

    if instrument == 'step' and centre_pix:
        filename = filename + '-centre_pix'
        title_string = title_string + ', centre pix'

    # this is to plot the points that are excluded due to different reasons 
    df_nan = df_info.where((df_info['frac_nonan'] < frac_nan_threshold), np.nan)
    df_no_sig = df_info.where((df_info['Peak_significance'] < sigma), np.nan)
    df_rel_err = df_info.where((df_info['rel_backsub_peak_err'] > rel_err_threshold), np.nan)

    # Plots either the background subtracted or raw flux peaks depending on choice.
   
    if(bg_subtraction):
        f, ax = plt.subplots(figsize=(13,10)) 
        if direction == '':
            direction = 'sun'
        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Bg_subtracted_peak'], yerr=df_info['Backsub_peak_uncertainty'],
                    xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']], color=color[direction], fmt='o', ecolor=color[direction], zorder=0, label='Flux peaks')
        ax.plot(df_nan.Primary_energy, df_nan.Bg_subtracted_peak, 'o', markersize=15, c='gray', label='excluded (NaNs)')
        ax.plot(df_no_sig.Primary_energy, df_no_sig.Bg_subtracted_peak, 'o', c='blue', markersize=11, label='excluded (sigma)')
        ax.plot(df_rel_err.Primary_energy, df_rel_err.Bg_subtracted_peak, 'o', c='orange', markersize=6, label='excluded (rel error)')
    elif(bg_subtraction == False):
        f, ax = plt.subplots(figsize=(13,10))
        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Flux_peak'], yerr=df_info['Peak_electron_uncertainty'],
                    xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']], fmt='o', color=color[direction],ecolor=color[direction], zorder=0, label='Flux peaks')
        ax.plot(df_nan.Primary_energy, df_nan.Flux_peak, 'o', markersize=15, c='gray', label='excluded (NaNs)')
        ax.plot(df_no_sig.Primary_energy, df_no_sig.Flux_peak, 'o', markersize=11, c='blue', label='excluded (sigma)')
        ax.plot(df_rel_err.Primary_energy, df_rel_err.Flux_peak, 'o', markersize=6, c='orange', label='excluded (rel error)')

    # Plots background flux and background errorbars in same scatterplot.
    ax.errorbar(x=df_info['Primary_energy'], y=df_info['Background_flux'], yerr=df_info['Bg_electron_uncertainty'], xerr=[df_info['Energy_error_low'],df_info['Energy_error_high']],
                fmt='o', color=color[direction], ecolor=color[direction], alpha=0.15, label='Background flux')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Energy [MeV]', size=20)
    ax.set_ylabel('Flux \n [1/s cm$^2$ sr MeV]', size=20)
    plt.tick_params(axis='x', which='minor', labelsize=16)
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    #plt.tick_params(axis='y', which='minor')
    #ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
    plt.legend(prop={'size': 18})
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.grid()
    plt.title(title_string)

    for label in ax.xaxis.get_ticklabels(which='minor')[1::2]:

        label.set_visible(False)
    
    if(path[len(path)-1] != '/'):

        path = path + '/'

    if(savefig):

        plt.savefig(path + filename + str(key) +'.jpg', dpi=300, bbox_inches='tight')

    plt.show()

def plot_spectrum_average(args, bg_subtraction=True, savefig=False, path='', key='', sigma=3, frac_nan_threshold=0.4, rel_err_threshold=0.5, direction=None, centre_pix = False, date = None):
    """_summary_

    Args:
        args (_type_): _description_
        bg_subtraction (bool, optional): _description_. Defaults to True.
        savefig (bool, optional): _description_. Defaults to False.
        path (str, optional): _description_. Defaults to ''.
        key (str, optional): _description_. Defaults to ''.
        sigma (int, optional): _description_. Defaults to 3.
        frac_nan_threshold (float, optional): _description_. Defaults to 0.4.
        rel_err_threshold (float, optional): _description_. Defaults to 0.5.
        direction (_type_, optional): _description_. Defaults to None.
    """
    color = {'sun':'crimson','asun':'orange', 'north':'darkslateblue', 'south':'c'}
    df_info = args[1]
    instrument = args[4][0]
    data_type = args[4][1]
    if direction is None or instrument in ['STEP', 'step']:
        viewing = 'sun'
    else:
        viewing = f'-{direction}' 



    date_string = ''
    file_date = ''

    if date is None:
        date_string = str(df_info['Plot_period'][0][:-5])
        file_date = str(df_info['Plot_period'][0][:-5])

    else:
        date_string = str(date)[:-3]
        file_date = str(date)[:-3].replace(' ', '-').replace(':', '')
    

    title_string = instrument.upper() + ', ' + data_type.upper() + ', ' + date_string
    filename = 'spectrum-' + file_date + '-' + instrument.upper()  +viewing+ '-' + data_type.upper() 
    
    if(df_info['Averaging'][0]=='Mean'):

        title_string = title_string + ', ' + df_info['Averaging'][1].split()[2] + ' averaging'
        filename = filename + '-' + df_info['Averaging'][1].split()[2] + '_averaging'

    elif(df_info['Averaging'][0]=='No averaging'):

        title_string = title_string + ', no averaging'
        filename = filename + '-no_averaging'

    if(bg_subtraction):
        
       title_string = title_string + ', bg subtraction on'
       filename = filename + '-bg_subtr'

    else:

        title_string = title_string + ', bg subtraction off'
    
    if(instrument == 'ept'):

        if(df_info['Ion_contamination_correction'][0] and instrument=='ept'):

            title_string = title_string + ', ion correction on'
            filename = filename + '-ion_corr'

        elif(df_info['Ion_contamination_correction'][0]==False):

            title_string = title_string + ', ion correction off'

    if instrument == 'step' and centre_pix:
        filename = filename + '-centre_pix'
        title_string = title_string + ', centre pix'


    # this is to plot the points that are excluded due to different reasons 
    df_nan = df_info.where((df_info['frac_nonan'] < frac_nan_threshold), np.nan)
    df_no_sig = df_info.where((df_info['Average_significance'] < sigma), np.nan)
    df_rel_err = df_info.where((df_info['rel_backsub_peak_err'] > rel_err_threshold), np.nan)

    # Plots either the background subtracted or raw flux peaks average depending on choice.
    if(bg_subtraction):
        f, ax = plt.subplots(figsize=(13,10)) 
        if direction == '':
            direction = 'sun'
        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Bg_subtracted_average'], yerr=df_info['Backsub_peak_uncertainty'],
                    xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']], color=color[direction], fmt='o', ecolor=color[direction], zorder=0, label='Flux average')
        ax.plot(df_nan.Primary_energy, df_nan.Bg_subtracted_average, 'o', markersize=15, c='gray', label='excluded (NaNs)')
        ax.plot(df_no_sig.Primary_energy, df_no_sig.Bg_subtracted_average, 'o', c='blue', markersize=11, label='excluded (sigma)')
        ax.plot(df_rel_err.Primary_energy, df_rel_err.Bg_subtracted_average, 'o', c='orange', markersize=6, label='excluded (rel error)')
    
        # ax = df_info.plot.scatter(x='Primary_energy', y='Bg_subtracted_average', c='red', label='Flux average', figsize=(13,10))
        # ax.errorbar(x=df_info['Primary_energy'], y=df_info['Bg_subtracted_average'], yerr=df_info['Backsub_peak_uncertainty'],
        #             xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']], fmt='.', ecolor='red', alpha=0.5)
    elif(bg_subtraction == False):
        f, ax = plt.subplots(figsize=(13,10))
        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Flux_average'], yerr=df_info['Peak_electron_uncertainty'],
                    xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']], fmt='o', color=color[direction],ecolor=color[direction], zorder=0, label='Flux average')
        ax.plot(df_nan.Primary_energy, df_nan.Flux_average, 'o', markersize=15, c='gray', label='excluded (NaNs)')
        ax.plot(df_no_sig.Primary_energy, df_no_sig.Flux_average, 'o', markersize=11, c='blue', label='excluded (sigma)')
        ax.plot(df_rel_err.Primary_energy, df_rel_err.Flux_average, 'o', markersize=6, c='orange', label='excluded (rel error)')

        # ax = df_info.plot.scatter(x='Primary_energy', y='Flux_average', c='red', label='Flux average', figsize=(13,10))
        # ax.errorbar(x=df_info['Primary_energy'], y=df_info['Flux_average'], yerr=df_info['Peak_electron_uncertainty'],
        #             xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']], fmt='.', ecolor='red', alpha=0.5)
    
    # Plots background flux and background errorbars in same scatterplot.
    ax.errorbar(x=df_info['Primary_energy'], y=df_info['Background_flux'], yerr=df_info['Bg_electron_uncertainty'], xerr=[df_info['Energy_error_low'],df_info['Energy_error_high']],
                fmt='o', color=color[direction], ecolor=color[direction], alpha=0.15, label='Background flux')
    # df_info.plot(kind='scatter', x='Primary_energy', y='Background_flux', c='red', alpha=0.25, ax=ax, label='Background flux')
    # ax.errorbar(x=df_info['Primary_energy'], y=df_info['Background_flux'], yerr=df_info['Bg_electron_uncertainty'], xerr=[df_info['Energy_error_low'],df_info['Energy_error_high']],
    #             fmt='.', ecolor='red', alpha=0.15)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Energy [MeV]', size=20)
    ax.set_ylabel('Flux \n [1/s cm$^2$ sr MeV]', size=20)
    plt.tick_params(axis='x', which='minor', labelsize=16)
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    #plt.tick_params(axis='y', which='minor')
    #ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
    plt.legend(prop={'size': 18})
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.grid()
    plt.title(title_string)

    for label in ax.xaxis.get_ticklabels(which='minor')[1::2]:

        label.set_visible(False)
    
    if(path[len(path)-1] != '/'):

        path = path + '/'

    if(savefig):

        plt.savefig(path + filename + str(key) +'.jpg', dpi=300, bbox_inches='tight')

    plt.show()


def write_to_csv(args, date, path='', key='', direction=None,  centre_pix = False):
    """_summary_

    Args:
        args (_type_): _description_
        path (str, optional): _description_. Defaults to ''.
        key (str, optional): _description_. Defaults to ''.
        direction (_type_, optional): _description_. Defaults to None.
    """

    df_info = args[1]
    instrument = args[4][0]
    data_type = args[4][1]

    #date = str(df_info['Plot_period'][0][:-5])
    #hour = int(df_info['Plot_period'][0][-4:-2])
    #if hour >20 or hour<4:
    #    date = str(df_info['Plot_period'][1][:-5])

    

    if direction is None:
        viewing = 'sun'
    else:
        viewing = f'{direction}' 
    filename = 'electron_data-' + date + '-' + instrument.upper() +'-'+ viewing+ '-' + data_type.upper()

    if(df_info['Averaging'][0] == 'Mean'):
        
        filename = filename + '-' + df_info['Averaging'][1].split()[2] + '_averaging'

    elif(df_info['Averaging'][0] == 'No averaging'):

        filename = filename + '-no_averaging'

    if(instrument == 'ept'):

        if(df_info['Ion_contamination_correction'][0]):

            filename = filename + '-ion_corr'

    if  instrument == 'step' and centre_pix:
        
        filename =  filename + '-centre_pix'


    df_info.to_csv(path + filename + str(key) + '.csv',  sep = ';', index=False)

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



def centre_pix_average_comparison_spec(args, args_pix, bg_subtraction=True, savefig=False, path='', key='', sigma=3, frac_nan_threshold=0.4, rel_err_threshold=0.5, direction=None, date = None):
    """_summary_

    Args:
        args (_type_): _description_
        bg_subtraction (bool, optional): _description_. Defaults to True.
        savefig (bool, optional): _description_. Defaults to False.
        path (str, optional): _description_. Defaults to ''.
        key (str, optional): _description_. Defaults to ''.
        sigma (int, optional): _description_. Defaults to 3.
        frac_nan_threshold (float, optional): _description_. Defaults to 0.4.
        rel_err_threshold (float, optional): _description_. Defaults to 0.5.
        direction (_type_, optional): _description_. Defaults to None.
    """
    color = {'sun':'crimson','sun_pix':'purple'}
    df_info = args[1]
    df_info_pix = args_pix[1]

    instrument = 'STEP'
    data_type = args[4][1]
    
    viewing = 'sun'
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
    filename = 'spectrum-pix-comparison' + file_date + '-' + instrument + viewing+ '-' + data_type.upper() 
    
    
    if(df_info['Averaging'][0]=='Mean'):

        title_string = title_string + ', ' + df_info['Averaging'][1].split()[2] + ' averaging'
        filename = filename + '-' + df_info['Averaging'][1].split()[2] + '_averaging'
        

    elif(df_info['Averaging'][0]=='No averaging'):

        title_string = title_string + ', no averaging'
        filename = filename + '-no_averaging'
        

    if(bg_subtraction):
        
       title_string = title_string + ', bg subtraction on'
       filename = filename + '-bg_subtr'
       

    else:

        title_string = title_string + ', bg subtraction off'
    

    # this is to plot the points that are excluded due to different reasons 
    df_nan = df_info.where((df_info['frac_nonan'] < frac_nan_threshold), np.nan)
    df_no_sig = df_info.where((df_info['Peak_significance'] < sigma), np.nan)
    df_rel_err = df_info.where((df_info['rel_backsub_peak_err'] > rel_err_threshold), np.nan)

    df_nan_pix = df_info_pix.where((df_info_pix['frac_nonan'] < frac_nan_threshold), np.nan)
    df_no_sig_pix = df_info_pix.where((df_info_pix['Peak_significance'] < sigma), np.nan)
    df_rel_err_pix = df_info_pix.where((df_info_pix['rel_backsub_peak_err'] > rel_err_threshold), np.nan)

    

    # Plots either the background subtracted or raw flux peaks depending on choice.
   
    if(bg_subtraction):
        f, ax = plt.subplots(figsize=(13,10)) 
        if direction == '':
            direction = 'sun'
        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Bg_subtracted_peak'], yerr=df_info['Backsub_peak_uncertainty'],
                    xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']], color=color[direction], fmt='o', ecolor=color[direction], zorder=0, label='Flux peaks all pix avg')
        ax.plot(df_nan.Primary_energy, df_nan.Bg_subtracted_peak, 'o', markersize=15, c='gray', label='excluded (NaNs)')
        ax.plot(df_no_sig.Primary_energy, df_no_sig.Bg_subtracted_peak, 'o', c='blue', markersize=11, label='excluded (sigma)')
        ax.plot(df_rel_err.Primary_energy, df_rel_err.Bg_subtracted_peak, 'o', c='orange', markersize=6, label='excluded (rel error)')

        ax.errorbar(x=df_info_pix['Primary_energy'], y=df_info_pix['Bg_subtracted_peak'], yerr=df_info_pix['Backsub_peak_uncertainty'],
                    xerr=[df_info_pix['Energy_error_low'], df_info_pix['Energy_error_high']], color=color['sun_pix'], fmt='o', ecolor=color['sun_pix'], zorder=0, label='Flux peaks centre pix')
        ax.plot(df_nan_pix.Primary_energy, df_nan_pix.Bg_subtracted_peak, 'o', markersize=15, c='gray')#, label='excluded (NaNs)')
        ax.plot(df_no_sig_pix.Primary_energy, df_no_sig_pix.Bg_subtracted_peak, 'o', c='blue', markersize=11)#, label='excluded (sigma)')
        ax.plot(df_rel_err_pix.Primary_energy, df_rel_err_pix.Bg_subtracted_peak, 'o', c='orange', markersize=6)#, label='excluded (rel error)')



    elif(bg_subtraction == False):
        f, ax = plt.subplots(figsize=(13,10))
        ax.errorbar(x=df_info['Primary_energy'], y=df_info['Flux_peak'], yerr=df_info['Peak_electron_uncertainty'],
                    xerr=[df_info['Energy_error_low'], df_info['Energy_error_high']], fmt='o', color=color[direction],ecolor=color[direction], zorder=0, label='Flux peaks all pix avg')
        ax.plot(df_nan.Primary_energy, df_nan.Flux_peak, 'o', markersize=15, c='gray', label='excluded (NaNs)')
        ax.plot(df_no_sig.Primary_energy, df_no_sig.Flux_peak, 'o', markersize=11, c='blue', label='excluded (sigma)')
        ax.plot(df_rel_err.Primary_energy, df_rel_err.Flux_peak, 'o', markersize=6, c='orange', label='excluded (rel error)')

        ax.errorbar(x=df_info_pix['Primary_energy'], y=df_info_pix['Flux_peak'], yerr=df_info_pix['Peak_electron_uncertainty'],
                    xerr=[df_info_pix['Energy_error_low'], df_info_pix['Energy_error_high']], fmt='o', color=color['sun_pix'],ecolor=color['sun_pix'], zorder=0, label='Flux peaks centre pix')
        ax.plot(df_nan_pix.Primary_energy, df_nan_pix.Flux_peak, 'o', markersize=15, c='gray')#, label='excluded (NaNs)')
        ax.plot(df_no_sig_pix.Primary_energy, df_no_sig_pix.Flux_peak, 'o', markersize=11, c='blue')#, label='excluded (sigma)')
        ax.plot(df_rel_err_pix.Primary_energy, df_rel_err_pix.Flux_peak, 'o', markersize=6, c='orange')#, label='excluded (rel error)')


    # Plots background flux and background errorbars in same scatterplot.
    ax.errorbar(x=df_info['Primary_energy'], y=df_info['Background_flux'], yerr=df_info['Bg_electron_uncertainty'], xerr=[df_info['Energy_error_low'],df_info['Energy_error_high']],
                fmt='o', color=color[direction], ecolor=color[direction], alpha=0.15, label='Background flux all pix avg')

    ax.errorbar(x=df_info_pix['Primary_energy'], y=df_info_pix['Background_flux'], yerr=df_info_pix['Bg_electron_uncertainty'], xerr=[df_info_pix['Energy_error_low'],df_info_pix['Energy_error_high']],
                fmt='o', color=color['sun_pix'], ecolor=color['sun_pix'], alpha=0.15, label='Background flux centre pix')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Energy [MeV]', size=20)
    ax.set_ylabel('Flux \n [1/s cm$^2$ sr MeV]', size=20)
    plt.tick_params(axis='x', which='minor', labelsize=16)
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    #plt.tick_params(axis='y', which='minor')
    #ax.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
    plt.legend(prop={'size': 18})
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.grid()
    plt.title(title_string)

    for label in ax.xaxis.get_ticklabels(which='minor')[1::2]:

        label.set_visible(False)
    
    if(path[len(path)-1] != '/'):

        path = path + '/'

    if(savefig):

        plt.savefig(path + filename + str(key) +'.jpg', dpi=300, bbox_inches='tight')

    plt.show()
