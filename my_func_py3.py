# my_func_py3.py

#import datetime as datetime
import numpy as np
import math
import matplotlib.ticker as ticker
import pylab
import bisect
#from matplotlib.dates import DateFormatter
# from scipy import asarray as ar,exp
import scipy as sp
from lmfit import Model
#from datetime import *
import datetime
from matplotlib.dates import *
import pandas as pd
# import sys



def log_interp1d(xx, yy, kind='linear'):
    """make a logarithmic interpolation by going to lin space and taking the logarithms of the data
        check: https://stackoverflow.com/questions/29346292/logarithmic-interpolation-in-python
    Parameters
    ----------
    xx : array
        x-data
    yy : array
        y-data
    kind : str, optional
        kind of interpolation, by default 'linear' (see documentation of sp.interpolate.interp1d)

    Returns
    -------
    function 
        that makes a logarithmic interpolation based on my input values
    """    
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

    ## example how to run:
    x = [10, 100]
    y = [4e3, 2e2]
    f, ax = plt.subplots(1, figsize=(12,7))
    ax.loglog(x, y, 'ok')
    x_value = 50
    func = log_interp1d(x, y)
    interpolated_value = func(x_value)
    ax.plot(x_value, interpolated_value, 'om')



#def compare_arrays(x,y):
    #if len(x) > len(y):
        #x1 = x
        #x2 = y
    #if len(x) < len(y):
        #x1 = y
        #x2 = x
    ##if len(x) == len(y):
        ##goto
    #if len(x) != len(y):
        #out = np.full(len(x1), False, dtype=bool)
        #shift = 0
        #for i in range(len(x1)):
            #if x1[i] == x2[i+shift]:
                #out[i] = True
            #else:
                #shift = shift+1

    #else:
        #return np.ones(len(x),dtype=bool)
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
    
def cdf_info(cdf):
    from spacepy import pycdf
    if type(cdf) == str:
        cdf = pycdf.CDF(cdf)
    ccdf=pycdf.CDFCopy(cdf)    #Python dictionary containing attributes copied from the CDF.
    for i in ccdf.keys():
        print('"'+i+'"')
        print(cdf[i].attrs)
        print('')
    return cdf

def steradiant(cone_angle):
    x = cone_angle / 2.
    xx = x = 0.5*x*np.pi/180
    return(4*np.pi*np.sin(xx)*np.sin(xx))

def orthodrome(lon1,lat1, lon2, lat2, rad=False):
    '''
    calculates the othodrom (total angular separtion) between two coordinates 
    defined by their lon/lat positions
    '''
    if rad == False:
        lon1 = np.deg2rad(lon1)
        lon2 = np.deg2rad(lon2)
        lat1 = np.deg2rad(lat1)
        lat2 = np.deg2rad(lat2)
    ortho = np.arccos(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))

    return np.rad2deg(ortho)
    #
    # # approximate radius of earth in km
    # R = 6373.0
    #
    # lat1 = radians(52.2296756)
    # lon1 = radians(21.0122287)
    # lat2 = radians(52.406374)
    # lon2 = radians(16.9251681)
    #
    # dlon = lon2 - lon1
    # dlat = lat2 - lat1
    #
    # a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    # c = 2 * atan2(sqrt(a), sqrt(1 - a))
    #


def TIMESTAMP2DATE(timestamp):
    days = timestamp.days
    hours, sec = divmod(timestamp.seconds, 3600)
    minutes = divmod(sec, 60)[0]
    return days, hours, minutes

def convert_date_doy(year, month, day):
    """
    takes year, month, day; returns day of year
    """
    doy = (datetime.datetime(int(year),int(month),int(day)) - datetime.datetime(int(year),1,1)).days+1
    return doy

def doy2h(doy):
    h = np.fix(np.double(doy-np.fix(doy))*24.)
    minute = np.fix((((doy-np.fix(doy))*24.)%1)*60.)
    sec = np.fix((((((doy-np.fix(doy))*24.)%1)*60.)%1)*60.)
    return h, minute, sec

def date2doy(year, month, day):
    """
    takes year, month, day; returns day of year
    """
    if len(np.atleast_1d(year)) == 1:
        doy = (datetime.datetime(int(year),int(month),int(day))-datetime.datetime(int(year),1,1)).days+1
    else:
        doy = np.array([(datetime.datetime(int(year[i]),int(month[i]),int(day[i]))-datetime.datetime(int(year[i]),1,1)).days+1 for i in range(len(year))])
    return doy

def leap_year(year):
    """
    takes year, returns '1' if year is a leap year, '0' if not
    """
    #leap_year=(399+(int(year) % 400))/400 - (3+(int(year) % 4))/4
    #if year in [2000, 2400]:
        #leap_year = 1
    leap_year = 0
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    if leap:
        leap_year = 1
    return leap_year

def doy2date(year, doy):
    '''
    input: year,doy
    returns day, month
    '''
    day = []
    month = []
    if hasattr(year, "__len__") == False:
    #if isinstance(year, (int, np.uint)) or isinstance(year, (float) or isinstance(year, (np.int64))):
        year = [year]
    #if np.issubdtype(doy, np.integer) or np.issubdtype(doy, np.float):
    #if isinstance(doy, (int, np.uint)) or isinstance(doy, (float)):
    if hasattr(doy, "__len__") == False:
        doy = [doy]
        I = 1

    #if (type(year) == int) or (type(year) == float):
	#year = [year]
    #if (type(doy) == int) or (type(doy) == float):
        #I = 1
        #doy = [doy]
    else:
        I = len(doy)

    if (type(year) == int) or (len(year) == 1):
        year = np.zeros(len(doy))+year

    for i in range(0,I):
        leap=leap_year(year[i])
        # if type(year) == int:
        #     leap=leap_year(year)
        # else:
        #     leap=leap_year(year[i])
        # Jan
        if doy[i] < 32:
            day = np.append(day, int(np.fix(doy[i])))
            month = np.append(month, 1)
        # Feb
        if (doy[i] >= 32) and (doy[i] < 60+leap):
            day = np.append(day, int(np.fix(doy[i])-31))
            month = np.append(month, 2)
        # Mar
        if (doy[i] >= 60+leap) and (doy[i] < 91+leap):
            day = np.append(day, int(np.fix(doy[i])-(59+leap)))
            month = np.append(month, 3)
        # April
        if (doy[i] >= 91+leap) and (doy[i] < 121+leap):
            day = np.append(day, int(np.fix(doy[i])-(90+leap)))
            month = np.append(month, 4)
        # May
        if (doy[i] >= 121+leap) and (doy[i] < 152+leap):
            day = np.append(day, int(np.fix(doy[i])-(120+leap)))
            month = np.append(month, 5)
        # June
        if (doy[i] >= 152+leap) and (doy[i] < 182+leap):
            day = np.append(day, int(np.fix(doy[i])-(151+leap)))
            month = np.append(month, 6)
        # July
        if (doy[i] >= 182+leap) and (doy[i] < 213+leap):
            day = np.append(day, int(np.fix(doy[i])-(181+leap)))
            month = np.append(month, 7)
        # Aug
        if (doy[i] >= 213+leap) and (doy[i] < 244+leap):
            day = np.append(day, int(np.fix(doy[i])-(212+leap)))
            month = np.append(month, 8)
        # Sep
        if (doy[i] >= 244+leap) and (doy[i] < 274+leap):
            day = np.append(day, int(np.fix(doy[i])-(243+leap)))
            month = np.append(month, 9)
        # Oct
        if (doy[i] >= 274+leap) and (doy[i] < 305+leap):
            day = np.append(day, int(np.fix(doy[i])-(273+leap)))
            month = np.append(month, 10)
        # Nov
        if (doy[i] >= 305+leap) and (doy[i] < 335+leap):
            day = np.append(day, int(np.fix(doy[i])-(304+leap)) )
            month = np.append(month, 11)
        # Dec
        if (doy[i] >= 335+leap) and (doy[i] < 366+leap):
            day = np.append(day, int(np.fix(doy[i])-(334+leap)))
            month = np.append(month, 12)
    if len(day) == 1:
        day   = int(day[0])
        month = int(month[0])
    return day, month

def string_date_to_doy(sdate, return_date=False):
    if type(sdate) == str:
        sdate = [sdate]
    year = []
    mon  = []
    day  = []

    for date in sdate:
        sy,sm,sd = date.split('-')
        year.append(int(sy))
        mon.append(int(sm))
        day.append(int(sd))


    if len(year) == 1:
        year = year[0]
        mon  = mon[0]
        day  = day[0]

    if return_date:
        return year, mon, day
    else:
        doy = date2doy(year, mon, day)
        return doy

def doy2dt_pd(years, months=1, days=1, weeks=None, hours=None, minutes=None,
                 seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)



def doy2dt(year, doy):
    # convert decimal day of year to datetime
    if hasattr(year, "__len__") == False:
        year = [year]
    if hasattr(doy, "__len__") == False:
        doy = [doy]
    if len(doy) > len(year):
        year = np.zeros(len(doy))+year
    datearray = []
    for i in range(len(doy)):
        if np.isnan(doy[i]):
            datearray.append(pd.NaT)
        else:
            datearray.append((datetime.datetime(int(year[i]),1,1,0)+datetime.timedelta(float(doy[i])-1)))

    return np.array(datearray)


def dt_ind(datearray, X1, X2, year=2000):  # datetime index
    if (type(X1) == int) or (type(X1) == float):
        print( 'converting doy to datetime with year '+str(year))
        X1 = doy2dt(year, X1)
        X2 = doy2dt(year, X2)

    lower = bisect.bisect_right(datearray, X1[0])
    upper = bisect.bisect_left(datearray, X2[0])

    ind = np.arange(lower, upper+1)
    return np.array(ind)



def jul2date(jul):
    """
    Convert Julian Day to date.
    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
    4th ed., Duffet-Smith and Zwart, 2011.
    Parameters
    ----------
    jd : float
    Julian Day
    Returns
    -------
    year : int
    Year as integer. Years preceding 1 A.D. should be 0 or negative.
    The year before 1 A.D. is 0, 10 B.C. is year -9.
    month : int
    Month as integer, Jan = 1, Feb. = 2, etc.
    day : float
    Day, may contain fractional part.
    Examples
    --------
    Convert Julian Day 2446113.75 to year, month, and day.
    >>> jd_to_date(2446113.75)
    (1985, 2, 17.25)
    """
    yy = []
    mm = []
    dd = []
    if (type(jul) == float) or type(jul) == int:
        jul = [jul]
    for i in np.arange(len(jul)):
        jd = jul[i] + 0.5
        F, I = math.modf(jd)
        I = int(I)
        A = math.trunc((I - 1867216.25)/36524.25)
        if I > 2299160:
            B = I + 1 + A - math.trunc(A / 4.)
        else:
            B = I
        C = B + 1524
        D = math.trunc((C - 122.1) / 365.25)
        E = math.trunc(365.25 * D)
        G = math.trunc((C - E) / 30.6001)
        day = C - E + F - math.trunc(30.6001 * G)
        if G < 13.5:
            month = G - 1
        else:
            month = G - 13
        if month > 2.5:
            year = D - 4716
        else:
            year = D - 4715

        yy = np.append(yy, year)
        mm = np.append(mm, month)
        dd = np.append(dd, int(day))
    return yy, mm, dd



def jul2doy(jul):
    ddoy = []
    if (type(jul) == float) or type(jul) == int:
        jul = [jul]
    for i in np.arange(len(jul)):
        jd = jul[i] + 0.5
        F, I = math.modf(jd)
        I = int(I)
        A = math.trunc((I - 1867216.25)/36524.25)
        if I > 2299160:
            B = I + 1 + A - math.trunc(A / 4.)
        else:
            B = I
        C = B + 1524
        D = math.trunc((C - 122.1) / 365.25)
        E = math.trunc(365.25 * D)
        G = math.trunc((C - E) / 30.6001)
        day = C - E + F - math.trunc(30.6001 * G)
        if G < 13.5:
            month = G - 1
        else:
            month = G - 13
        if month > 2.5:
            year = D - 4716
        else:
            year = D - 4715

        doy = date2doy(year, month, int(day))
        ddoy = np.append(ddoy, doy+F)

    return ddoy



def progressbar(label, step, count_steps, bar_length, delay):
    """
    takes label, step, number of all steps, delay bw. steps; returns nothing
    """
    if delay > 0.0:
        # import threading
        threading._sleep(delay)
    print( "\r", label, '['+ \
        ('#'*(step*bar_length/(count_steps-1)))+ \
        (' '*(bar_length-(step*bar_length/(count_steps-1))))+']', \
        "%3d%%" %(step*100/(count_steps-1)),)
    return

def runningMean(x, N):
    # N: window size, should be uneven, if not, N will be extended so that the window length on both sides of the entry is the same

    if np.mod(N,2) != 0:
        N = N+1

    y   = np.zeros(len(x))
    sig = np.zeros(len(x))

    for i in np.arange(len(x)):
        y[0] = x[0]
        if (i > 0) and (i < N/2):
            y[i]   = np.nanmean(x[0:i])
            sig[i] = np.nanstd(x[0:i])

        if (i>= N/2) or (i <= len(x)-N/2):
            y[i]   = np.nanmean(x[i-N/2:i+N/2])
            sig[i] = np.nanstd(x[i-N/2:i+N/2])

        if i > len(x)-N/2:
            y[i]   = np.nanmean(x[i:len(x)])
            sig[i] = np.nanstd(x[i:len(x)])

        #y[-1] = x[-1]
    return y, sig

def runningVar(x, N):
    # N: window size, should be uneven, if not, N will be extended so that the window length on both sides of the entry is the same

    if np.mod(N,2) != 0:
        N = N+1

    y   = np.zeros(len(x))
    var = np.zeros(len(x))

    for i in np.arange(len(x)):
        y[0] = x[0]
        if (i > 0) and (i < N/2):
            y[i]   = np.nanmean(x[0:i])
            var[i] = np.nanvar(x[0:i])

        if (i>= N/2) or (i <= len(x)-N/2):
            y[i]   = np.nanmean(x[i-N/2:i+N/2])
            var[i] = np.nanvar(x[i-N/2:i+N/2])

        if i > len(x)-N/2:
            y[i]   = np.nanmean(x[i:len(x)])
            var[i] = np.nanvar(x[i:len(x)])

        #y[-1] = x[-1]
    return y, var

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    #TODO: the window parameter could be the window itself if an array instead of a string
    #NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if (window_len % 2 == 0) == True:
        window_len+=1

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    cut_end = int(((window_len-1)/2)*-1)
    y = y[int((window_len-1)/2):cut_end]
    return y



def polarity(Br,Bt,Bn):
    phi=[]

    for i in np.arange(len(Br)):
        if Bt[i] > 0.:
            phi = np.append(phi, np.arccos(Br[i]/(np.sqrt(Br[i]**2+Bt[i]**2))))
        else:
            phi = np.append(phi, 2*np.pi-np.arccos(Br[i]/(np.sqrt(Br[i]**2+Bt[i]**2))))

    phi = np.rad2deg(phi)

    pol = np.zeros(len(Br))+np.nan

    for j in np.arange(len(Br)):
        if phi[j] > 55 and phi[j] < 215:
            pol[j]=-1                       # -1 = -
        if phi[j] > 235:
            pol[j]=1                        #  1 = +
        if phi[j] < 35:
            pol[j]=1
        if phi[j] >= 35 and phi[j] <= 55:
            pol[j]=0                        #  0 = unclear range
        if phi[j] >= 215 and phi[j] <= 235:
            pol[j]=0

    return pol


def mag_angles(B,Br,Bt,Bn):
    theta = np.arccos(Bn/B)
    alpha = 90-(180/np.pi*theta)

    r = np.sqrt(Br**2 + Bt**2 + Bn**2)
    phi = np.arccos(Br/np.sqrt(Br**2 + Bt**2))*180/np.pi

    sel = np.where(Bt < 0)
    count = len(sel[0])
    if count > 0:
        phi[sel] = 2*np.pi - phi[sel]
    sel = np.where(r <= 0)
    count = len(sel[0])
    if count > 0:
        phi[sel] = 0

    return alpha, phi

def solo_mag_angles(Br,Bt,Bn):
    B = np.sqrt(Br**2+Bt**2+Bn**2)
    theta = np.arccos(Bn/B)
    alpha = 90-(180/np.pi*theta)

    r = np.sqrt(Br**2 + Bt**2 + Bn**2)
    phi = np.arccos(Br/np.sqrt(Br**2 + Bt**2))*180/np.pi

    sel = np.where(Bt < 0)
    count = len(sel[0])
    if count > 0:
        phi[sel] = 2*np.pi - phi[sel]
    sel = np.where(r <= 0)
    count = len(sel[0])
    if count > 0:
        phi[sel] = 0

    return alpha, phi


def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def polyn(x,c0,c1,c2):  # 2nd order
    poly = c0+c1*x+c2*x**2#+c3*x**3
    return poly

def polyn3(x,c0,c1,c2,c3):  # 3rd order poly
    poly = c0+c1*x+c2*x**2+c3*x**3
    return poly

def polyn4(x,c0,c1,c2,c3,c4):   # 4th order poly
    poly = c0+c1*x+c2*x**2+c3*x**3+c4*x**4
    return poly

def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def sym_gauss(x,a,sigma):
    return a*np.exp(-(x)**2/(2*sigma**2))

def line(x, a, b):
    return a*x+b

##this one is from the typo in Agueda & Lario 2016
#def ani_func(x, a0, a1, a2):
    ##x = np.arccos(x)
    ##F = a0 + a1*np.cos(x) + a2/2. * (3.*np.cos(2.*x)-1)

    #F = a0 + a1*x + a2 * (3*x**2 -2)
    #return F

def ani_func(x, a0, a1, a2): # typo corrected
    # own typo corrected Oct'18:
    # I had F = a0 + a1*x + a2 * (3*x**2 -2)
    # x: mu (cos(theta))

    # function for theta:
    # F = a0 + a1*np.cos(x) + a2/2. * (3.*np.cos(x)*np.cos(x)-1)
    #F = a0 + a1*x + a2 * (3*x**2 -2)

    F = a0 + a1*x + 0.5 * a2 * (3*x**2 -1)
    return F


def shadow(ax, x, y, shade_range, y1=0, color='cadetblue', alpha=0.5):
    '''
    this plots a colored (default=gray) shadow below the (time series) data
    this is done by plotting little rectangular boxes below the y-data for each time step
    x : x-array
    y : y-array (curve, below which the filling should be)
    shade_range: [x_shade_begin, x_shade_end]
    '''

    shade_all = x[np.where((x >= shade_range[0]) & (x <= shade_range[1]))[0]]
    for i in range(0,len(shade_all)-1):
        shade_int = np.where((x>=shade_all[i]) & (x <= shade_all[i+1]))[0]
        # shade_x = np.reshape(x[shade_int], (len(shade_x,)))
        shade_x = np.reshape(x[shade_int], (len(shade_int)))
        shade_y = y[shade_int]
        if y1 == 0:
            y1p=np.zeros(len(shade_x))+1e-12
        else:
            y1p = np.zeros(len(shade_x))+y1
        ax.fill_between(shade_x, y1p, shade_y, where=shade_y>y1p, facecolor=color, alpha=alpha, linewidth=0.0)

    return shade_x, shade_y

def shadow_bar(ax, x, y1, y2, shade_range, color='khaki', alpha=0.65):
    '''
    this plots a bar (rectangle)
    '''
    nonan = np.where(np.isnan(x) == False)[0]
    x = x[nonan]
    shade_range = list(shade_range)
    for i in range(0,len(shade_range)-1, 2):
        shade_int = np.where((x >= shade_range[i]) & (x <= shade_range[i+1]))[0]
        shade_x = x[shade_int]
        shade_x = np.reshape(shade_x, (len(shade_x,)))
        y1p = np.zeros(len(shade_x))+y1
        y2p = np.zeros(len(shade_x))+y2
        shadow = ax.fill_between(shade_x, y1p, y2p, where=y2p>y1p, facecolor=color, alpha=alpha, edgecolor="w")
    return shadow


def shadow_bar_datetime(ax, x, y1, y2, shade_range, color='khaki', alpha=0.65):
    '''
    this plots a bar (rectangle)
    '''
    # nonan = np.where(np.isnan(x) == False)[0]
    # x = x[nonan]
    shade_range = list(shade_range)
    for i in range(0,len(shade_range)-1, 2):
        shade_int = np.where((x >= shade_range[i]) & (x <= shade_range[i+1]))[0]
        shade_x = x[shade_int]
        shade_x = np.reshape(shade_x, (len(shade_x,)))
        y1p = np.zeros(len(shade_x))+y1
        y2p = np.zeros(len(shade_x))+y2
        shadow = ax.fill_between(shade_x, y1p, y2p, where=y2p>y1p, facecolor=color, alpha=alpha, edgecolor="w")
    return shadow


def new_shadow_bar(ax, x, shade_range, color='bisque', alpha=0.65):  #'cadetblue'
    '''
    this plots a bar (rectangle)
    it now gets the ylimits itself
    '''
    if isinstance(x, pd.core.indexes.datetimes.DatetimeIndex):
        nonan = np.where(pd.isna(x) == False)[0]
    else:
        nonan = np.where(np.isnan(x) == False)[0]

    x = x[nonan]
    shade_range = list(shade_range)
    y1, y2 = ax.get_ylim()
    for i in range(0,len(shade_range)-1, 2):
        shade_int = np.where((x >= shade_range[i]) & (x <= shade_range[i+1]))[0]
        shade_x = x[shade_int]
        y1p = np.zeros(len(shade_x))+y1
        y2p = np.zeros(len(shade_x))+y2
        shadow = ax.fill_between(shade_x, y1p, y2p, where=y2p>y1p, facecolor=color, alpha=alpha, edgecolor="w")
    return shadow


#######################################################################################################
def my_time_axis(ax, X1, X2, year, which='doy', xlabel='', labels=True, date_format='%b %d', nminor=-1):
    '''
    workaround for error message that x=0 not works:
    start plotting interval 0.0000001 later
    '''

    which_ax = {'doy':0, 'date':1, 'time':2}
    if labels:
        which_labels = ['Doy in %4d'%year, 'Date in %4d'%year, 'Time (UT)']
        if xlabel == '':
            xlabel = which_labels[which_ax[which]]

    dur = X2-X1
    one_hour = 1/24.
    one_min = 1/1440.

    if which == 'doy':
        if nminor == -1:
            if (dur<=5):
                min_ticks = 0.5
            if ((dur>5) & (dur<=15)):
                min_ticks = 1
            if ((dur>15) & (dur<=25)):
                min_ticks = 1
            if ((dur>25) & (dur<=40)):
                min_ticks = 1
            if dur>=40:
                min_ticks = 1
        else:
            min_ticks = 1./nminor
        #minor ticks:
        minorLocator = ticker.MultipleLocator(min_ticks)
        ax.xaxis.set_minor_locator(minorLocator)

        if labels:
            if dur < 2:
                def my_formatter_fun(x,y):
                    return "%.2f" %x
                ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(my_formatter_fun))
            ax.set_xlabel(xlabel)
        else:
            ax.get_xaxis().set_ticklabels('')

    if (which=='date'):
        try:
            dur < 8
        except:
            dur = dur.days
        if (dur<=8):
            h_ticks = 1
            min_ticks = 0.5
        if ((dur>8) & (dur<=15)):
            h_ticks = 2
            min_ticks = 1
        if ((dur>15) & (dur<=25)):
            h_ticks = 3
            min_ticks = 1
        if ((dur>25) & (dur<=40)):
            h_ticks = 4
            min_ticks = 1
        if dur>=40:
            h_ticks = 5
            min_ticks = 1
        if nminor > -1:
            min_ticks = 1./nminor


        majorLocator   = ticker.MultipleLocator(h_ticks)
        ax.xaxis.set_major_locator(majorLocator)

        #minor ticks:
        minorLocator = ticker.MultipleLocator(min_ticks)
        ax.xaxis.set_minor_locator(minorLocator)

        if labels:
            ax.xaxis.set_major_formatter(DateFormatter(date_format))
            ax.set_xlabel(xlabel)
        else:
            ax.get_xaxis().set_ticklabels('')



    if which == 'time':
        try:
            dur < 4*one_hour
        except:
            dur = dur.hours
        if (dur<=4*one_hour):
            h_ticks = 0.5
            min_ticks = 0.25
        if ((dur>=4*one_hour) & (dur<=7*one_hour)):
            h_ticks = 1
            min_ticks = 0.5
        if ((dur>7*one_hour) & (dur<=132*one_hour)):
            h_ticks = 2
            min_ticks = 1
        if ((dur>13*one_hour) & (dur<=24*one_hour)):
            h_ticks = 4
            min_ticks = 1
        if dur>24*one_hour:
            h_ticks = 6
            min_ticks = 1

        majorLocator = ticker.MultipleLocator(one_hour*h_ticks)
        ax.xaxis.set_major_locator(majorLocator)

        #minor ticks:
        minorLocator = ticker.MultipleLocator(min_ticks*one_hour)
        ax.xaxis.set_minor_locator(minorLocator)

        if labels:
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.set_xlabel(xlabel)
        else:
            ax.get_xaxis().set_ticklabels('')
    #ax.tick_params(labelsize=14)
    return None


###########################################################################################
def my_datetime_axis(ax, X1, X2, which_ax='Date', xlabel='', labels=True, date_format='', portrait=False):
    '''
    X1, X2: x-limits of the axis in datetime.datetime objects, (e.g., X1 = datetime.datetime(2010, 11, 12, 7, 45))

    '''

    dur = X2-X1   # duration in seconds
    #one_hour = 1/24.
    one_hour = pd.Timedelta(hours=1)
    if portrait: dur = dur*1.5


    if (dur < one_hour):
        m_ticks = 10
        min_ticks = 1
    if ((dur > one_hour) & (dur<=4*one_hour)):
        m_ticks = 30
        min_ticks = 15
    if ((dur>=4*one_hour) & (dur<=13*one_hour)):
        m_ticks = 1
        min_ticks = 30
    if ((dur>13*one_hour) & (dur<=24*one_hour)):
        m_ticks = 4
        min_ticks = 1
    if (dur>24*one_hour) & (dur <=pd.Timedelta(days=1, hours=12)):
        m_ticks = 6
        min_ticks = 1
    if (dur>pd.Timedelta(days=1, hours=12)) & (dur<=pd.Timedelta(days=8)):
        m_ticks = 1
        min_ticks = 6
    if ((dur>pd.Timedelta(days=8)) & (dur<=pd.Timedelta(days=15))):
        m_ticks = 2
        min_ticks = 1
    if ((dur>pd.Timedelta(days=15)) & (dur<=pd.Timedelta(days=25))):
        m_ticks = 3
        min_ticks = 1
    if ((dur>pd.Timedelta(days=25)) & (dur<=pd.Timedelta(days=40))):
        m_ticks = 4
        min_ticks = 1
    if dur>=pd.Timedelta(days=40):
        m_ticks = 5
        min_ticks = 1

    if (dur < one_hour):
        majorLocator = MinuteLocator(byminute=range(0,60,m_ticks))
        minorLocator = MinuteLocator(byminute=range(0,60,min_ticks))

    if ((dur > one_hour) & (dur<=4*one_hour)):
        majorLocator = MinuteLocator(byminute=range(0,60,m_ticks))
        minorLocator = MinuteLocator(byminute=range(0,60,min_ticks))

    if ((dur>=4*one_hour) & (dur <= 13.*one_hour)):
        majorLocator = HourLocator(byhour=range(0,24,m_ticks))
        minorLocator = MinuteLocator(byminute=range(0,60,min_ticks))


    if ((dur>13*one_hour) & (dur <= pd.Timedelta(days=1, hours=12))):
        majorLocator = HourLocator(byhour=range(0,24,m_ticks))
        minorLocator = HourLocator(byhour=range(0,24,min_ticks))

    if (dur>pd.Timedelta(days=1, hours=12)) & (dur <= pd.Timedelta(days=8)):
        majorLocator = DayLocator(interval=m_ticks)
        minorLocator = HourLocator(byhour=range(0,24,min_ticks))

    if (dur > pd.Timedelta(days=8)):
        majorLocator = DayLocator(interval=m_ticks)
        minorLocator = DayLocator(interval=min_ticks)

    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)

    if which_ax == 'date': which_ax = 'Date'
    if which_ax == 'time': which_ax = 'Time'
    if which_ax == 'doy': which_ax = 'Doy'

    ax_titles  = {'Date':'Date in %4d'%X1.year, 'Time':'Time (UT)', 'Doy':'Doy in %4d'%X1.year}
    ax_formats = {'Date':'%b %d', 'Time':'%H:%M', 'Doy':'%j'}
    if date_format == '':
        date_format = ax_formats[which_ax]
    if xlabel == '':
        xlabel = ax_titles[which_ax]

    if labels:
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        ax.set_xlabel(xlabel)
    else:
        ax.get_xaxis().set_ticklabels('')

    #sys.exit()

    return None

#######################################################################################################
def auto_yrange(ax, X1, X2, doy, y):

    y = y[:,(np.where(doy>=X1)) and (np.where(doy<=X2))]
    flat_y = y.flatten()
    if sum(np.isnan(flat_y)) != len(flat_y):
        low = np.min(y[y>0])
        high = np.max(y[np.where(np.isnan(y) == False)])
        ax.set_ylim(low-0.1*low, high+0.5*high)
        ax.set_yscale('log')
        Y1 = low-0.1*low
        Y2 = high+0.5*high

    return Y1, Y2


#######################################################################################################
def auto_yrange_sec(ax, X1, X2, doy, y):
	'''
	here y is a matrix containing several intensities which are all plotted in one panel
	e.g. for sectored SEPT data
	'''

	if X1 == -1:
		X1 = doy[0]
	if X2 == -1:
		X2 = doy[-1]

	n = len(y[0,:])
	low = []
	high = []
	index = np.where((doy>=X1) & (doy<=X2))[0]
	y = y[index,:]
	for i in range(n):
		inte = y[:,i]
		print( inte)
		#inte = inte[(np.where(doy>=X1)) and (np.where(doy<=X2))]
		#flat_y = inte.flatten()
		#if sum(np.isnan(flat_y)) != len(flat_y):
		low.append(np.nanmin(inte[inte>0]))
		high.append(np.nanmax(inte))
		print( 'i, low, high:')
		print( i, low, high)
	low  = np.nanmin(low)
	high = np.nanmax(high)
	if ax != None:
		ax.set_ylim(low-0.1*low, high+0.5*high)
		ax.set_yscale('log')
	Y1 = low-0.1*low
	Y2 = high+0.5*high

	return Y1, Y2


#######################################################################################################
def CME_height(Rs, hp, alpha):
    '''
    returns the height of the CME in units of solar radii Rs from image data
    Parameters
    -----------
    Rs: solar radius as measured from an image
    hp: projected height of the CME above the limb as measured from the image
    alpha: angle between limb and center of CME (footpoint of heigth measure)
    '''

    h = ((hp + Rs)/np.cos(np.deg2rad(alpha))) - Rs
    h = h/Rs

    return h


#######################################################################################################
def ICME_height(start_time, speed, tt, start_height=2., h=0, minu=0, year=0):

    '''
    returns the hight of an ICME traveled with a certain speed for a certain time
    start_time: start time of the CME in dec. doy (or int(doy) and then h and minu determine the time)
    speed in km/s
    tt: travel time in dec doy
    start_height in R_sun (default = 1 R_sun)
    '''
    AU = 149597870.700  # km
    R_sun = 695700. # in km
    distance = (tt*86400.*speed  - (start_height)*R_sun) / AU

    #distance = distance * AU - (start_height)*R_sun
    #tt = distance /speed
    #t_ddoy = tt / 86400.
    return distance

########################################################################################################
def CME_arrival(start_time, speed, distance, start_height=2., h=0, minu=0, year=0):
    '''returns the estimated arrival time of a CME_arrival at a certain distance

    start_time: start time of the CME in dec. doy (or int(doy) and then h and minu determine the time)
    speed in km/s
    distance in AU
    start_height in R_sun (default = 1 R_sun)
    '''

    AU = 149597870.700  # km
    R_sun = 695700. # in km
    distance = distance * AU - (start_height)*R_sun
    tt = distance /speed
    t_ddoy = tt / 86400.

    if (h > 0) or (minu > 0):
        start_time = start_time + h/24. + minu/1440.
    else:
        h, minu, sec = doy2h(start_time)

    t_arr = start_time + t_ddoy

    print( 'Start of CME at '+str(int(h))+':'+str(int(minu))+'UT on day '+str(int(np.fix(start_time))))


    print( 'Arrival of ICME with v='+str(speed)+'km/s :')


    print( 'doy: '+str(np.fix(int(t_arr))))
    if year > 0:
        day, mon = doy2date(year, int(np.fix(t_arr)))
        print( 'Date (day/mon/year): '+str(int(day)).zfill(2)+'/'+str(int(mon)).zfill(2)+'/'+str(year))

    hour, minu, sec = doy2h(t_arr)
    print( 'Time: '+str(int(hour)).zfill(2)+':'+str(int(minu)).zfill(2)+':'+str(int(sec)).zfill(2))


    return t_arr



########################################################################################################
def number_of_turns(L, X, R):
    # N = number of number_of_turns
    # L path length of electrons, i.e. field line length inside the MC
    # X flux rope axial length
    # R minor radius (from diameter of the MC)

    N = 1/(2*np.pi) * np.sqrt(L**2/X**2 -1) * X/R

    tau = (2*np.pi*N)/X
    print( 'twist tau:')
    print( tau)

    return N

###########################################################################################################
def cusumcalc(j, sampling, tole):  #, cusum, ha, background):
    # from Rauls function CUSUMcalc
    #;this routine calculates CUSUM values for unidimensional time series jj[*]
    #;nume is the number of points to sample for average background calculation
    #;and also estimates the cusum onset
    #;tole is the number of sigmas to consider in the algorithm
    #;e.g.cusumcalc,j[i,*],sampling,cusum,ha
    print( 'calculating cusum with')
    print( 'sampling: '+str(sampling))
    print( 'tolerance: '+str(tole))


    last       = len(j)-1
    cusum      = np.zeros(len(j))  #initialize cusum values
    backmean   = np.zeros(len(j)) #initialize cusum values
    ha         = np.zeros(len(j)) #floor(cusum);initialize decission variable
    k          = np.zeros(len(j))
    for i in np.arange(sampling, last-3): #loop over whole intensity array

        background = j[0:i]  # background is summed up from the beginning up to i


        sample = j[(i-sampling):(i-1)]
  # needed for the background period; to get its mean and std
        if (len(np.where(np.isnan(sample) == True)[0]) <= len(sample)-2):
            #result = [np.nanmean(sample), np.nanstd(sample)]
            result = [np.nanmean(background), np.nanstd(background)]
        else:
            result = [1E99,1E99]
            # warning-->CHECK FOR FALSE ONSET BECAUSE THESE FORCED NEGATIVE VALUES WHEN NO DATA AVAILABLE....
        backmean[i]   = result[0]
        sigma         = result[1]
        mua           = result[0]
        mud           = mua+tole*sigma
        #print mua, mud
        if (mua == 0):
            ka=1
        else:
            ka=(mud-mua)/(math.log(mud)-math.log(mua))
        k[i] = ka
        #print ka
        #;if ka lt 1 then ha[i]=1 else ha[i]=2
        #;2014 modification: ha values are 1 or 2 IN COUNTS, but this program uses fluxes instead.
        #;for this reason we "calibrate" ha in the following way:
        #;ha=1 ---> use the non-cero minimum flux value instead of 1
        #;ha=2 ---> use 2*the "ha=1 replacement value"
        valids = np.where((np.isnan(j) == False) & (j > 0))[0]
        if (len(valids) > 0):
            ha1 = np.nanmin(j[valids])
            ha2 = 2*ha1
            #print 'hi'
        else:
            ha1 = 1
            ha2 = 2
        if (ka < 1):
            ha[i] = ha1
        else:
            ha[i] = ha2
            #print 'hu'
        #print j[i]-ka+cusum[i-1]

        if np.isnan(j[i]) == True:
            cusum[i] = cusum[i-1]
        else:
            cusum[i] = np.nanmax([[0],[j[i]-ka+cusum[i-1]]])
        #print cusum[i]
        #print ka[i]


    return cusum, ha, backmean, k



#######################################################################################################
def cusum_onset(doy, j, sampling, tole, points_over, cusum, ha, back):
    print('cusum search with '+str(points_over)+' points over ha')


    cusum_over = np.where(cusum >= ha)[0]
    onset = -1
    for i in np.arange(sampling, len(cusum_over)):
        if (cusum_over[(i+points_over)] == cusum_over[i]+points_over):
            onset = doy[cusum_over[i]]
            break
        onset = -999
    return onset






# This function gave much too small values..
# However, beta is available in the pressure data set on the STEREO mag data page
########################################################################################################
#def stereo_plasma_beta(mag_doy, mag_B, v_doy, v_n, v_T, reso):
    #'''
    #input:
    #mag_doy, mag_B: doy and |B| from mag dataset
    #v_doy, v_n, v_T: solar wind plasma data: doy, density, temperature
    #returns:
    #plasma-beta
    #'''
    #kB   = 1.3806488e-23
    #mu0 = 4*np.pi*1e-7
    #t_steps = {'1min': (1/1440.), '10min': (15/1440.), '1h': (1/24.)}
    #tstep = t_steps[reso]

    ## here we produce doy-arrays with exactly same sized steps (there were longer gaps in mag)
    #sort_mag_doy = np.arange(mag_doy[0], mag_doy[-1], tstep)
    #sort_ind = np.searchsorted(sort_mag_doy, mag_doy, side='right')-1
    #sort_mag_B = np.zeros(len(sort_mag_doy))*np.nan
    #sort_mag_B[sort_ind] = mag_B

    #sort_v_doy = np.arange(v_doy[0], v_doy[-1], tstep)
    #sort_ind = np.searchsorted(sort_v_doy, v_doy, side='right')-1
    #sort_n = np.zeros(len(sort_v_doy))*np.nan
    #sort_n[sort_ind] = v_n
    #sort_T = np.zeros(len(sort_v_doy))*np.nan
    #sort_T[sort_ind] = v_T

    ## here we cut if the mag or the vsw data are over a longer period than the other data set
    #if len(sort_mag_doy) < len(sort_v_doy):
	#ind        = np.where((sort_v_doy >= sort_mag_doy[0]) & (sort_v_doy <= sort_mag_doy[-1]))[0]
	#sort_v_doy = sort_v_doy[ind]
	#sort_n     = sort_n[ind]
	#sort_T     = sort_T[ind]
    #if len(mag_doy) > len(v_doy):
	#ind          = np.where((sort_mag_doy >= sort_v_doy[0]) & (sort_mag_doy <= sort_v_doy[-1]))[0]
	#sort_mag_doy = sort_mag_doy[ind]
	#sort_mag_B   = sort_mag_B[ind]

    #beta = ((2*mu0)*sort_n*(1e-2**3)*kB*sort_T)/((sort_mag_B*1e-9)**2)
    #return sort_mag_doy, beta



#def length_of_trace(x, y):
    length = 0
    for i in np.arange(0,(len(x)-1),1):
        r = np.sqrt(abs(x[i]-x[i+1])**2 + abs(y[i]-y[i+1])**2)
        length = length+r

    return length

def E2rigidity(ekin, which='electron'):
    electron_rest_energy = 0.51 # in MeV
    proton_rest_energy = 938.28   # MeV

    if (which == 'electron'):
        rest_energy = electron_rest_energy
    if (which == 'proton'):
        rest_energy = proton_rest_energy

    ekin_MeV = ekin/1e3
    rigidity = np.sqrt(ekin_MeV*(ekin_MeV + 2.*rest_energy))
    return rigidity

#def gauss(x,a,x0,sigma):
    #return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gaussian_lmfit(x,y, a=10, x0=0, sigma=40, print_report=False):
	'''
    fits a gaussian function to data using lmfit

	x,y: data to fit
	gamma1 gamma2, c1, c2, E_break: guess-values for the fit
	returns
	'''

	plmodel = Model(gauss)
	params  = plmodel.make_params(a=a, x0=x0, sigma=sigma)

	#params['gamma1'].min = -7
	#params['gamma1'].max = -1
	#params['gamma2'].min = -7
	#params['gamma2'].max = -1
	#params['E_break'].vary = False


	result = plmodel.fit(y, params, x=x)#, weights=1/unc)
	if print_report:
		print(result.fit_report())

	return result


def fit_line(x, y, a=1, b=0, print_report=False):
    '''
    fits a line y = a*x + b
    '''
    plmodel = Model(line)
    params  = plmodel.make_params(a=a,b=b)

    result = plmodel.fit(y, params, x=x)#, weights=1/unc)
    if print_report:
        print(result.fit_report())

    return result
