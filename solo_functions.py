# solo_functions.py
import numpy as np
import pandas as pd
import my_power_law_fits_odr as pl_fit
from scipy.stats import t as studentt
# import cdflib
#from epd_loader import *
# from solo_epd_loader import *
from solo_epd_loader import epd_load
import datetime
from my_func_py3 import unit_vector, angle_between
from tqdm.auto import tqdm


def calc_EPT_corrected_e(df_ept_e, df_ept_p):
    ion_cont_corr_matrix = np.loadtxt('/home/annafed/Documents/GitHub/SolOEPD-Plot-n-Fit-SEE/EPT_ion_contamination_flux_paco.dat')  # using the new calibration files (using the sun_matrix because they don't differ much)
    Electron_Flux_cont = np.zeros(np.shape(df_ept_e))*np.nan
    for tt in range(len(df_ept_e)):
        Electron_Flux_cont[tt, :] = np.sum(ion_cont_corr_matrix * np.ma.masked_invalid(df_ept_p.values[tt, :]), axis=1)
    df_ept_e_corr = df_ept_e - Electron_Flux_cont
    return df_ept_e_corr

def calc_av_en_flux_EPD(df, energies, en_channel, species, instrument):  # original from Nina Slack Feb 9, 2022, rewritten Jan Apr 8, 2022
    """This function averages the flux of several energy channels of HET into a combined energy channel
    channel numbers counted from 0

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing HET data
    energies : dict
        Energy dict returned from epd_loader (from Jan)
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons', 'p', 'i', 'protons', 'ions'
    instrument : string
        'ept' or 'het'

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux

    Raises
    ------
    Exception
        [description]
    """
    if species.lower() in ['e', 'electrons']:
        en_str = energies['Electron_Bins_Text']
        bins_width = 'Electron_Bins_Width'
        flux_key = 'Electron_Flux'
    if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
        if instrument.lower() == 'het':
            en_str = energies['H_Bins_Text']
            bins_width = 'H_Bins_Width'
            flux_key = 'H_Flux'
        if instrument.lower() == 'ept':
            en_str = energies['Ion_Bins_Text']
            bins_width = 'Ion_Bins_Width'
            flux_key = 'Ion_Flux'
    if type(en_channel) == list:
        energy_low = en_str[en_channel[0]][0].split('-')[0]

        energy_up = en_str[en_channel[-1]][0].split('-')[-1]

        en_channel_string = energy_low + '-' + energy_up

        if len(en_channel) > 2:
            raise Exception('en_channel must have len 2 or less!')
        if len(en_channel) == 2:
            DE = energies[bins_width]
            try:
                df = df[flux_key]
            except (AttributeError, KeyError):
                None
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                if bins == en_channel[0]:
                    I_all = df[f'{flux_key}_{bins}'] * DE[bins]
                else:
                    I_all = I_all + df[f'{flux_key}_{bins}'] * DE[bins]
            DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
            flux_out = pd.DataFrame({'flux': I_all/DE_total}, index=df.index)
        else:
            en_channel = en_channel[0]
            # flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'{flux_key}_{en_channel}']}, index=df.index)
            flux_out = pd.DataFrame({'flux': df[f'{flux_key}_{en_channel}']}, index=df.index)
            en_channel_string = en_str[en_channel][0]
    else:
        en_channel_string = en_str[en_channel][0]
        #flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'{flux_key}_{en_channel}']}, index=df.index)
        flux_out = pd.DataFrame({'flux': df[f'{flux_key}_{en_channel}']}, index=df.index)
    return flux_out, en_channel_string


def resample_df(df, resample):
    """
    Resample Pandas Dataframe
    """
    try:
        # _ = pd.Timedelta(resample)  # test if resample is proper Pandas frequency
        df = df.resample(resample).mean()
        df.index = df.index + pd.tseries.frequencies.to_offset(pd.Timedelta(resample)/2)
    except ValueError:
        raise Warning(f"Your 'resample' option of [{resample}] doesn't seem to be a proper Pandas frequency!")
    return df



def calc_pa_coverage(instrument, mag_data):
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

# def calc_pa_coverage(instrument, mag_data):
#     mag_data = mag_data.rename(columns={'B_SRF_0':'Bx', 'B_SRF_1':'By','B_SRF_2':'Bz'})
#     print(f'Calculating PA coverage for {instrument}...')
#     if instrument not in ['ept', 'EPT', 'het', 'HET']:
#         print("instrument not known, select either 'EPT' or 'HET' ")
#         coverage = pd.DataFrame(mag_data.index)
#     else:
#         if instrument in ['ept', 'EPT']:
#             opening = 30
#         if instrument in ['het', 'HET']:
#             opening = 43
#         mag_vec = np.array([mag_data.Bx.values, mag_data.By.values, mag_data.Bz.values])
#         # pointing directions of EPD (arrows point into the sensor)
#         pointing_sun =  np.array([-0.8192, 0.5736, 0.]) 
#         pointing_asun =  np.array([ 0.8192, -0.5736, 0.])
#         pointing_north =  np.array([ 0.3030, 0.4765, -0.8253])
#         pointing_south =  np.array([-0.3030, -0.4765, 0.8253])
#         pa_sun = np.ones(len(mag_data.Bx.values)) * np.nan
#         pa_asun = np.ones(len(mag_data.Bx.values)) * np.nan
#         pa_north = np.ones(len(mag_data.Bx.values)) * np.nan
#         pa_south = np.ones(len(mag_data.Bx.values)) * np.nan

#         for i in range(len(mag_data.Bx.values)):
#             pa_sun[i] = np.rad2deg(angle_between(pointing_sun, mag_vec[:,i]))
#             pa_asun[i] = np.rad2deg(angle_between(pointing_asun, mag_vec[:,i]))
#             pa_north[i] = np.rad2deg(angle_between(pointing_north, mag_vec[:,i]))
#             pa_south[i] = np.rad2deg(angle_between(pointing_south, mag_vec[:,i]))

#     sun_min = pa_sun - opening/2
#     sun_max = pa_sun + opening/2
#     asun_min = pa_asun - opening/2
#     asun_max = pa_asun + opening/2
#     north_min = pa_north - opening/2
#     north_max = pa_north + opening/2
#     south_min = pa_south - opening/2
#     south_max = pa_south + opening/2    
#     cov_sun = pd.DataFrame({'min':sun_min, 'center':pa_sun, 'max':sun_max}, index=mag_data.index)
#     cov_asun = pd.DataFrame({'min':asun_min, 'center':pa_asun, 'max':asun_max}, index=mag_data.index)
#     cov_north = pd.DataFrame({'min':north_min, 'center':pa_north, 'max':north_max}, index=mag_data.index)
#     cov_south = pd.DataFrame({'min':south_min, 'center':pa_south, 'max':south_max}, index=mag_data.index)
#     keys = [('sun'), ('asun'), ('north'), ('south')]
#     coverage = pd.concat([cov_sun, cov_asun, cov_north, cov_south], keys=keys, axis=1)
#     coverage[coverage > 180] = 180
#     coverage[coverage < 0] = 0
#     return coverage

    

def print_EPT_energies(energies):
    species = 'Ion_Bins_Text'
    print('')
    print('Ions')
    for i in range(len(energies[species])):
        print(i, energies[species][i])
    species = 'Electron_Bins_Text'
    print('')
    print('Electrons')
    for i in range(len(energies[species])):
        print(i, energies[species][i])

def print_HET_energies(energies):
    species = 'H_Bins_Text'
    print('')
    print('Protons')
    for i in range(len(energies[species])):
        print(i, energies[species][i])
    species = 'Electron_Bins_Text'
    print('')
    print('Electrons')
    for i in range(len(energies[species])):
        print(i, energies[species][i])




def last_day_of_month(date):
    if date.month == 12:
        return date.replace(day=31)
    return date.replace(month=date.month+1, day=1) - datetime.timedelta(days=1)

def concat_epd_cdf_monthly(year, mon, averaging, instrument, sector, species='e'):
    """
    !!! EPT not implemented yet! 
    makes monthly files of EPD/EPT or HET data with certain time averaging applied
        example for making a whole year of data files: 
            for mon in np.arange(1,13,1):concat_epd_cdf_monthly(2021, mon, '1h', 'het','asun', 'e')
        to read the data use epd_load_concat_monthly() [see below]

    Parameters
    ----------
    year : int
        
    mon : int
        
    averaging : string
        '1h', '10min'
    instrument : string
       'EPT' or 'HET'
    sector : string
        
    species : str, optional
        electrons 'e' or protons 'p', by default 'e'

    Returns
    -------
    returns None
       saves the dataframe (only for selected species) to a csv file in the same folder where the original cdf files are

    """    

    path = '/Users/dresing/data/projects/solo/SOAR/'
    sdate = year*10000+mon*100+1
    date = datetime.date(year, mon, 1)
    last = last_day_of_month(date)
    edate = last.year*10000+last.month*100+last.day
    if sector == 'omni':
        df_p_sun, df_e_sun, energies = epd_load(instrument, 'l2', sdate, enddate=edate, viewing='sun', path=path, autodownload=True)
        df_p_asun, df_e_asun, energies = epd_load(instrument, 'l2', sdate, enddate=edate, viewing='asun', path=path, autodownload=True)
        df_p_north, df_e_north, energies = epd_load(instrument, 'l2', sdate, enddate=edate, viewing='north', path=path, autodownload=True)
        df_p_south, df_e_south, energies = epd_load(instrument, 'l2', sdate, enddate=edate, viewing='south', path=path, autodownload=True)
        df_p = (df_p_sun + df_p_asun + df_p_north + df_p_south) / 4
        df_e = (df_e_sun + df_e_asun + df_e_north + df_e_south) / 4
    df_p, df_e, energies = epd_load(instrument, 'l2', sdate, enddate=edate, viewing=sector, path=path, autodownload=True)
    if species in ['e', 'electron', 'electrons']:
        df = df_e
        species = 'e'
    if species in ['p', 'proton', 'protons', 'ions', 'ion']:
        df = df_p
        species = 'p'
    
    df  = df.resample(averaging,label='left').mean()
    df.to_csv(path+'l2/epd/'+instrument+'/'+'solo_L2_epd-'+instrument+'-'+sector+'-rates_'+str(year)+'_'+str(mon)+'_'+species+'_'+averaging+'.csv')
    return None
    

def concat_epd_cdf(sdate, edate, averaging, instrument, sector, species='e'):
    path = '/Users/dresing/data/projects/solo/SOAR/'

    df_p, df_e, energies = read_epd_cdf(instrument, sector, 'l2', sdate, edate, path=path, autodownload=True)
    if species in ['e', 'electron', 'electrons']:
        df = df_e
        species = 'e'
    if species in ['p', 'proton', 'protons', 'ions', 'ion']:
        df = df_p
        species = 'p'
    # df_het_e = het_e['Electron_Flux'] 
    df  = df.resample(averaging,label='left').mean()
    df.to_csv(path+'l2/epd/'+instrument+'/'+'solo_L2_epd-'+instrument+'-'+sector+'-rates_'+str(sdate)+'-'+str(edate)+'_'+species+'_'+averaging+'.csv')
    return None

def epd_load_concat_monthly(year, mon, averaging, instrument, sector, species='e'):
    path = '/Users/dresing/data/projects/solo/SOAR/l2/epd/'
    data = pd.read_csv(path+instrument+'/'+'solo_L2_epd-'+instrument+'-'+sector+'-rates_'+str(year)+'_'+str(mon)+'_'+species+'_'+averaging+'.csv', header=[0,1], parse_dates=True, index_col=0)
    return data

def epd_load_concat(sdate, edate, averaging, instrument, sector, species='e'):
    path = '/Users/dresing/data/projects/solo/SOAR/l2/epd/'
    data = pd.read_csv(path+instrument+'/'+'solo_L2_epd-'+instrument+'-'+sector+'-rates_'+str(sdate)+'-'+str(edate)+'_'+species+'_'+averaging+'.csv', header=[0,1], parse_dates=True, index_col=0)
    return data


def calc_av_en_flux_HET(df, energies, en_channel, species):
    """This function averages the flux of several energy channels of HET into a combined energy channel
    channel numbers counted from 0          

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing HET data
    energies : dict
        Energy dict returned from epd_loader (from Jan)
    en_channel : int or list
        energy channel or list with first and last channel to be used
    species : string
        'e', 'electrons', 'p', 'i', 'protons', 'ions'

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux

    Raises
    ------
    Exception
        [description]
    """    
    try:
        if species not in ['e', 'electrons', 'p', 'protons', 'H']:
            raise ValueError("species not defined. Must by one of 'e', 'electrons', 'p', 'protons', 'H'")
    except ValueError as error:
        print(repr(error))
        raise
    

    if species in ['e', 'electrons']:
        en_str = energies['Electron_Bins_Text']
        bins_width = 'Electron_Bins_Width'
        flux_key = 'Electron_Flux'
    if species in ['p', 'protons', 'H']:
        en_str = energies['H_Bins_Text']
        bins_width = 'H_Bins_Width'
        flux_key = 'H_Flux'
        if flux_key not in df.keys():
            flux_key = 'H_Flux'
    if type(en_channel) == list:
        en_channel_string = en_str[en_channel[0]][0].split()[0]+' - '+en_str[en_channel[-1]][0].split()[2]+' '+en_str[en_channel[-1]][0].split()[3]
        if len(en_channel) > 2:
            raise Exception('en_channel must have len 2 or less!')
        if len(en_channel) == 2:
            DE = energies[bins_width]
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                if bins == en_channel[0]:
                    I_all = df[flux_key].values[:,bins] * DE[bins]
                else:
                    I_all = I_all + df[flux_key].values[:,bins] * DE[bins]
            DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
            flux_av_en = pd.Series(I_all/DE_total, index=df.index)
            flux_out = pd.DataFrame({'flux':flux_av_en}, index=df.index)
        else:
            en_channel = en_channel[0]
            flux_out = pd.DataFrame({'flux':df[flux_key].values[:,en_channel]}, index=df.index)
    else:
        flux_out = pd.DataFrame({'flux':df[flux_key].values[:,en_channel]}, index=df.index)
        en_channel_string = en_str[en_channel]
    return flux_out, en_channel_string

def calc_av_en_flux_EPT(df, energies, en_channel, species):
    """
    This function averages the flux of several energy channels of EPT into a combined energy channel
    channel numbers counted from 0          

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing EPT data
        DataFrame containing EPT data
    energies : dict
        Energy dict returned from epd_loader (from Jan)
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons', 'p', 'i', 'protons', 'ions'

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux

    Raises
    ------
    Exception
        [description]
    """    
    try:
        if species not in ['e', 'electrons', 'p', 'i', 'protons', 'ions']:
            raise ValueError("species not defined. Must by one of 'e', 'electrons', 'p', 'i', 'protons', 'ions'")
    except ValueError as error:
        print(repr(error))
        raise

    
    if species in ['e', 'electrons']:
        bins_width = 'Electron_Bins_Width'
        flux_key = 'Electron_Flux'
        en_str = energies['Electron_Bins_Text']
    if species in ['p', 'i', 'protons', 'ions']:
        bins_width = 'Ion_Bins_Width'
        flux_key = 'Ion_Flux'
        en_str = energies['Ion_Bins_Text']
        if flux_key not in df.keys():
            flux_key = 'H_Flux'
    if type(en_channel) == list:
        en_channel_string = en_str[en_channel[0]][0].split()[0]+' - '+en_str[en_channel[-1]][0].split()[2]+' '+en_str[en_channel[-1]][0].split()[3]
        if len(en_channel) > 2:
            raise Exception('en_channel must have len 2 or less!')
        if len(en_channel) == 2:
            DE = energies[bins_width]
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                if bins == en_channel[0]:
                    I_all = df[flux_key].values[:,bins] * DE[bins]
                else:
                    I_all = I_all + df[flux_key].values[:,bins] * DE[bins]
            DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
            flux_av_en = pd.Series(I_all/DE_total, index=df.index)
            flux_out = pd.DataFrame({'flux':flux_av_en}, index=df.index)
        else:
            en_channel = en_channel[0]
            flux_out = pd.DataFrame({'flux':df[flux_key].values[:,en_channel]}, index=df.index)
    else:
        flux_out = pd.DataFrame({'flux':df[flux_key].values[:,en_channel]}, index=df.index)
        en_channel_string = en_str[en_channel]
    return flux_out, en_channel_string

def make_step_electron_flux(stepdata, mask_conta=False):
    '''
    March 2023: corrected the wrong masking for STEP contaminated electron periods
    here we use the calibration factors from Paco (Alcala) to calculate the electron flux out of the (integral - magnet) fluxes (we now use level2 data files to get these)
    we also check if the integral counts are sufficiently higher than the magnet counts so that we can really assume it's electrons (ohterwise we mask the output arrays)
    As suggested by Alex Kollhoff & Berger use a 5 sigma threashold:
    C_INT >> C_MAG:
    C_INT - C_MAG > 5*sqrt(C_INT)
    not yet implemented: Alex: die count rates und fuer die uebrigen Zeiten gebe ich ein oberes Limit des Elektronenflusses an, das sich nach 5*sqrt(C_INT) /(E_f - E_i) /G_e berechnet.
    '''

    # calculate electron flux from F_INT - F_MAG:
    colnames = ["ch_num", "E_low", "E_hi", "factors"]
    paco = pd.read_csv('/Users/dresing/data/projects/solo/step_electrons_calibration.csv', names=colnames, skiprows=1)
    F_INT = stepdata['Integral_Flux']
    F_MAG = stepdata['Magnet_Flux']

    step_flux = (F_INT - F_MAG) * paco.factors.values

    U_INT = stepdata['Integral_Uncertainty']
    U_MAG = stepdata['Magnet_Uncertainty']

    # from Paco:
    # Ele_Uncertainty = k * sqrt(Integral_Uncertainty^2 + Magnet_Uncertainty^2)
    step_unc = np.sqrt(U_INT**2 + U_MAG**2) * paco.factors.values


    param_list = ['Electron_Flux', 'Electron_Uncertainty']

    if mask_conta:
        # C_INT = stepdata['Integral_Rate']
        # C_MAG = stepdata['Magnet_Rate']
        # # clean = (C_INT - C_MAG) > 5*np.sqrt(C_INT) ### this was the wrong way around and also a too high value
        # step_flux = step_flux.mask(clean)
        # step_unc = step_unc.mask(clean)

        # new:
        # clean = (df[f'Integral_Avg_Flux_{i}'] - df[f'Magnet_Avg_Flux_{i}']) > contamination_threshold*df[f'Integral_Avg_Uncertainty_{i}']  ### from Jan's new loader
        clean = (F_INT - F_MAG) > 2 * U_INT
        # mask non-clean data
        step_flux = step_flux.mask(~clean)
        step_unc = step_unc.mask(~clean)

    step_data = pd.concat([step_flux, step_unc], axis=1, keys=param_list)

    return step_data, paco.E_low, paco.E_hi


def average_flux_error(flux_err: pd.DataFrame) -> pd.Series:
    return np.sqrt((flux_err ** 2).sum(axis=0)) / len(flux_err.values)



def MAKE_THE_FIT(spec_e, spec_flux, e_err, flux_err, ax, direction='sun', which_fit='broken', e_min=None, e_max=None, g1_guess=-2., g2_guess=None, alpha_guess=5., break_guess=0.120, c1_guess=1e3):
    '''
    fits a spectrum with power law function (either single or double pl)
    '''
    color = {'sun':'crimson','asun':'orange', 'north':'darkslateblue', 'south':'c'}
    spec_e = np.array(spec_e)
    spec_flux = np.array(spec_flux)
    e_err = np.array(e_err)
    flux_err = np.array(flux_err)

    if e_min is None:
        e_min = spec_e[0]
    if e_max is None:
        e_max = spec_e[-1]

    if g2_guess is None:
        g2_guess = g1_guess - 0.1

    xplot = np.logspace(np.log10(np.nanmin(spec_e)), np.log10(np.nanmax(spec_e)), num=500)
    xplot = xplot[np.where((xplot >= e_min) & (xplot <= e_max))[0]]

    fit_ind   = np.where((spec_e >= e_min) & (spec_e <= e_max))[0]
    spec_e    = spec_e[fit_ind]
    spec_flux = spec_flux[fit_ind]
    e_err     = e_err[fit_ind]
    flux_err  = flux_err[fit_ind]

    if which_fit == 'single':
        result_single_pl = pl_fit.power_law_fit(spec_e, spec_flux, e_err, flux_err, gamma1=g1_guess, c1=c1_guess)
        result = result_single_pl
        # dof_single                   = len(x) - len(result_single_pl.beta)
        redchi_single  = result_single_pl.res_var  #result_single_pl.sum_square / dof_single
        c1          = result_single_pl.beta[0]
        gamma1      = result_single_pl.beta[1]
        dof         = len(spec_e) - len(result_single_pl.beta)
        t_val       = studentt.interval(0.95, dof)[1]
        errors      = t_val * result_single_pl.sd_beta  #np.sqrt(np.diag(final_fit.cov_beta))
        gamma1_err  = errors[1]
        ax.plot(xplot, pl_fit.simple_pl([c1, gamma1], xplot), '-', color=color[direction], label=r'$\mathregular{\delta=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err))
        ax.plot(xplot, pl_fit.simple_pl([c1, gamma1], xplot), '--k')

    if which_fit == 'broken':
        result_broken = pl_fit.broken_pl_fit(spec_e, spec_flux, e_err, flux_err, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break=break_guess, maxit=10000)
        result        = result_broken
        breakp        = result_broken.beta[4]
        alpha         = result_broken.beta[3]
        dof           = len(spec_e) - len(result_broken.beta)
        redchi_broken = result_broken.res_var

        t_val      = studentt.interval(0.95, dof)[1]
        errors     = t_val * result_broken.sd_beta  #np.sqrt(np.diag(result_broken.cov_beta))
        breakp_err = errors[4]
        if alpha > 0 :
            gamma1     = result_broken.beta[1]
            gamma1_err = errors[1]
            gamma2     = result_broken.beta[2]
            gamma2_err = errors[2]
        if alpha < 0 :
            gamma1     = result_broken.beta[2]
            gamma1_err = errors[2]
            gamma2     = result_broken.beta[1]
            gamma2_err = errors[1]

        fit_plot = pl_fit.broken_pl_func(result_broken.beta, xplot)
        fit_plot[fit_plot == 0] = np.nan
        ax.plot(xplot, fit_plot, '-b', label=r'$\mathregular{\delta_1=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err)+'\n'+r'$\mathregular{\delta_2=}$%5.2f' %round(gamma2, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma2_err)+'\n'+r'$\mathregular{\alpha=}$%5.2f' %round(alpha, ndigits=2))#, lw=lwd)
        ax.axvline(x=breakp, color='blue', linestyle='--', label=r'$\mathregular{E_b=}$ '+str(round(breakp*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_err*1e3, ndigits=0))+' keV')

    return result


def DETERMINE_PEAK_SPEC(flux, flux_err):
    '''
    returns the peak flux of each energy channel in flux
    returns the assocated peak times
    flux: flux array, flux time series of all energy channels
    flux covers only the time interval of interest and is already time averaged
    '''
    # spec_energy = flux.columns
    peak_times = []
    peak_flux = []
    peak_flux_err = []
    for ch in range(flux.values.shape[1]):
        if any(np.isfinite(flux.values[:,ch])):
            max_ind = np.where(np.nanmax(flux.values[:,ch]) == flux.values[:,ch])[0][0]
            peak_times.append(flux.index[max_ind])
            peak_flux.append(flux.values[max_ind,ch])
            peak_flux_err.append(flux_err.values[max_ind,ch])
        else:
            peak_flux.append(np.nan)
            peak_times.append(pd.NaT)
            peak_flux_err.append(np.nan)
            # TO DO: ADD ALSO METADATA TO spec_data, multiply by 1e2

    spec_data = pd.DataFrame({'peak_time':peak_times, 'flux':peak_flux, 'flux_err':peak_flux_err})#, 'spec_energy':spec_energy})
    return spec_data
