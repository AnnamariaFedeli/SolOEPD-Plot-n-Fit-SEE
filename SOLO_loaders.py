# SOLO_loaders.py

import numpy as np
import pandas as pd
#import cdflib
import datetime as dt
import os
from cdflib.epochs import CDFepoch
import glob
from seppy.loader.solo import mag_load
#from solo_mag_loader import mag_load  # https://github.com/jgieseler/solo-mag-loader
from pandas.tseries.frequencies import to_offset
from sunpy.timeseries import TimeSeries
path_loc = '/Users/dresing/data/projects/solo/'

# cdf = pycdf.CDF('rbspa_rel03_ect-rept-sci-L2_20170101_v5.1.0.cdf')
# print(cdf.attrs)           #general info about the cdf file
# print(cdf['L'].attrs)     #more info about the corresponding variable, in this case L
# cdf_info(cdf)     #list all variable description of the cdf file read-in as 'cdf'

def cdf_info(cdf):
    from spacepy import pycdf
    ccdf=pycdf.CDFCopy(cdf)    #Python dictionary containing attributes copied from the CDF.
    for i in ccdf.keys():
        print('"'+i+'"')
        print(cdf[i].attrs)
        print('')
    return



def solo_soar_step_loader(startdate, enddate, data_product='l2'):
    '''
    reads L2 data files but returns only a few paramters
    unit of Fluxes and Uncertainties is particles / (s cm^2 sr MeV)
    '''
    from spacepy import pycdf
    all_cdf = []

    for date in pd.date_range(startdate.date(), enddate.date()):
        all_cdf.append(solo_soar_step_loader_sf(date, data_product))
    cdf = pycdf.concatCDF(all_cdf)



    cdffile = pycdf.CDFCopy(cdf)
    if data_product == 'l2':
        param_list = ['Integral_Flux', 'Magnet_Flux', 'Integral_Rate', 'Magnet_Rate', 'Magnet_Uncertainty', 'Integral_Uncertainty']
    if data_product == 'll':
        # not yet working!!!
        # no 'Integral_Rate' in ll data!!! this is needed to make the conta correction / masking...
        param_list = ['Integral_Flux', 'Ion_Flux', 'Integral_Rate', 'Magnet_Rate', 'Ion_Flux_Sigma', 'Integral_Flux_Sigma']
    df_list = []
    for key in param_list:
        df_list.append(pd.DataFrame(cdffile[key], index=cdffile['EPOCH']))

    data = pd.concat(df_list, axis=1, keys=param_list)

    return data

def solo_soar_step_loader_sf(indate, data_product):
    '''
    reads L2 data cdf files
    '''
    from spacepy import pycdf
    cyear = str(indate.year)
    cmon = '%02d'%(indate.month)
    cday = '%02d'%(indate.day)
    version = 1
    vers = '%02d'%(version)
    path = f'{path_loc}SOAR/{data_product}/epd/STEP/{cyear}/'
    if data_product == 'l2':
        filename = f'solo_L2_epd-step-rates_{cyear}{cmon}{cday}_V{vers}.cdf'
        if os.path.isfile(path+filename):
            with pycdf.CDF(path+filename) as cdf:
                cdffile = pycdf.CDFCopy(cdf)
        else:
            version = version+1
            vers = '%02d'%(version)
            filename = f'solo_L2_epd-step-rates_{cyear}{cmon}{cday}_V{vers}.cdf'
            with pycdf.CDF(path+filename) as cdf:
                cdffile = pycdf.CDFCopy(cdf)

    if data_product == 'll':
        filename = glob.glob(path+f'solo_LL02_epd-step-rates_{cyear}{cmon}{cday}*')[0]
        if os.path.isfile(filename):
            with pycdf.CDF(filename) as cdf:
                cdffile = pycdf.CDFCopy(cdf)
    return cdffile#, cdffile



def autodownload_mag_soar(startdate, enddate, level=2, resolution='1-MINUTE'):
    """[summary]

    Parameters
    ----------
    startdate : str
        e.g., '2021-04-17'
    enddate : str
        e.g., '2021-04-17'
    level : int
        0, 1, 2
    """    
    # works only for l2 data
    # works not if startdate and enddate are in different years
    # with the identifier one could also implement the download of burst, rtn, and 1min rtn data
    # import sunpy_soar
    from sunpy.net import Fido
    from sunpy.net.attrs import Instrument, Level, Time
    from sunpy_soar.attrs import Identifier

    if level == 2:
        # identifier = Identifier('MAG-SRF-NORMAL')  # l2 data
        # path = '/Users/dresing/data/projects/solo/mag/l2_soar/srf/'+startdate[0:4]+'/'
        if resolution == '1-MINUTE':
            identifier = Identifier('MAG-RTN-NORMAL-1-MINUTE')
            path = '/Users/dresing/data/projects/solo/mag/l2_soar/rtn_1minute/'+startdate[0:4]+'/'
        if resolution == 'BURST':
            identifier = Identifier('MAG-RTN-BURST')
            path = '/Users/dresing/data/projects/solo/mag/l2_soar/rtn_high_time_res/'+startdate[0:4]+'/'        
    # if level == 0:
   

    # Create search attributes
    instrument = Instrument('MAG')
    time = Time(startdate, enddate)
    level = Level(level)
    # Do search & fetch files
    result = Fido.search(instrument & time & level & identifier)
    files = Fido.fetch(result, path=path)
    print(files)

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
        mav = f'{av}min' 
        mag_offset = f'{av/2}min' 
        mag_data = mag_data.resample(mav,label='left').mean()
        mag_data.index = mag_data.index + to_offset(mag_offset)

    return mag_data

def solo_mag_loader_local(filelist, frame = 'srf', av=None):
    """
    loads files, that have already been downloaded to SOAR folder

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
        number of minutes to average, by default None

    Returns
    -------
    [type]
        [description]
    """    
    print('Loading MAG...')
    #mag_data = mag_load(sdate, edate, level=level, data_type=type, frame=frame, path=path)
    mag_data = TimeSeries(filelist, concatenate=True).to_dataframe()
    #mag_data = mag_load(sdate, edate, level=level, frame=frame, path=path)
    if frame == 'rtn':
        mag_data.rename(columns={'B_RTN_0':'B_r', 'B_RTN_1':'B_t', 'B_RTN_2':'B_n'}, inplace=True)
    if frame == 'srf':
        mag_data.rename(columns={'B_SRF_0':'Bx', 'B_SRF_1':'By', 'B_SRF_2':'Bz'}, inplace=True)
    if av is not None:
        mav = f'{av}min' 
        mag_offset = f'{av/2}min' 
        mag_data = mag_data.resample(mav,label='left').mean()
        mag_data.index = mag_data.index + to_offset(mag_offset)

    return mag_data


# def solo_mag_loader(sdate, edate, level='l2', type='normal', frame='rtn', av=None):
#     """
#     to do: implement higher resultion averaging ('1S' (seconds)) for burst data
#     loads SolO/MAG data from soar using function mag_load() from Jan: 
#     autodownloads if files are not there

#     Parameters
#     ----------
#     sdate : int
#         20210417
#     edate : int
#         20210418
#     level : str, optional
#         by default 'l2'
#     type : str, optional
#         'burst', 'normal-1-minute', by default 'normal'
#     frame : str, optional
#         'srf', by default 'rtn'
#     av : int or None, optional
#         number of minutes to average, by default None

#     Returns
#     -------
#     [type]
#         [description]
#     """    
#     print('Loading MAG...')
#     mag_data = mag_load(sdate, edate, level='l2', type=type, frame=frame)
#     if frame == 'rtn':
#         mag_data.rename(columns={'B_RTN_0':'B_r', 'B_RTN_1':'B_t', 'B_RTN_2':'B_n'}, inplace=True)
#     if frame == 'srf':
#         mag_data.rename(columns={'B_SRF_0':'Bx', 'B_SRF_1':'By', 'B_SRF_2':'Bz'}, inplace=True)
#     if av is not None:
#         mav = f'{av}min' 
#         mag_offset = f'{av/2}min' 
#         mag_data = mag_data.resample(mav,label='left').mean()
#         mag_data.index = mag_data.index + to_offset(mag_offset)

#     return mag_data




# def solo_mag_loader(year, mon, day, type='normal', frame='rtn'):
#     '''
#     To do: solve issue with filename versions

#     Reads Solar Orbiter MAG data with 1min res (to do: higher res)
#     Files are in cdf format and are daily files
#         PARAMETERS:
#         -----------
#         year: int
#         mon: int
#         day: int
#         frame: str, 'rtn' or 'srf'
#             which coordinate system, default: 'rtn', 'srf'=spacecraft reference frame
#         type: str, 'normal' or 'burst'
#             which time resolution, default: 'normal'

#         RETURNS:
#         --------
#         data: pandas dataframe of selected data
#         cdf: the whole cdf data structure

#         FURTHER INFO REGARDING DATA USE:
#         --------------------------------
#         QUALITY_FLAG: int
#         {'CATDESC': 'High level information about the quality of the magnetic field vector', 'DEPEND_0': 'EPOCH', 'DISPLAY_TYPE': 'time_series', 'FIELDNAM': 'Quality flag', 'FILLVAL': 254, 'FORMAT': 'I1', 'LABLAXIS': 'Quality flag', 'SCALEMAX': 4, 'SCALEMIN': 0, 'SCALETYP': 'linear', 'UNITS': 'None', 'VALIDMAX': 4, 'VALIDMIN': 0, 'VAR_NOTES': 'Flag setting: 0:Bad data; 1: Known problems use at your own risk; 2: Survey data, possibly not publication quality; 3: Good for publication subject to PI approval; 4: Excellent data which has received special treatment; refer SOL-MAG-DPDD for more information on how these flags are generated.', 'VAR_TYPE': 'metadata'}
#     '''
#     cyear = str(year)
#     cmon = '%02d'%(mon)
#     cday = '%02d'%(day)

#     path = path_loc+'mag/l2_soar/rtn_1minute/'+cyear+'/'
#     filename = 'solo_L2_mag-'+frame+'-'+type+'-1-minute_'+cyear+cmon+cday+'_V02.cdf'
#     with pycdf.CDF(path+filename) as cdf:
#         cdffile = pycdf.CDFCopy(cdf)
#     if frame=='rtn':
#         d = {'date':cdffile['EPOCH'], 'Br':cdffile['B_RTN'][:,0], 'Bt':cdffile['B_RTN'][:,1], 'Bn':cdffile['B_RTN'][:,2], 'QualityFlag':cdffile['QUALITY_FLAG']}
#     if frame=='srf':
#         d = {'date':cdffile['EPOCH'], 'Bx':cdffile['B_SRF'][:,0], 'By':cdffile['B_SRF'][:,1], 'Bz':cdffile['B_SRF'][:,2], 'QualityFlag':cdffile['QUALITY_FLAG']}
#     data = pd.DataFrame(data=d)
#     return data, cdffile

