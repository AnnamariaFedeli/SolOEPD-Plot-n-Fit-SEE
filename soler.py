# NEW VERSION OF THE GENERAL VERSION. iN THE MAKING.

from turtle import title
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as pltt
from sunpy.coordinates import get_horizons_coord
import make_the_fit_tripl as fitting
#from make_the_fit_tripl import  MAKE_THE_FIT
#from make_the_fit import closest_values
#from make_the_fit import find_c1
import combining_files as comb
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import soler_functions as sf


def run_the_fit(path, data, save, channels_to_exclude = None, plot_title = '', x_label = 'Intensity [/]', y_label = 'Energy [MeV]', legend_title = '', data_label_for_legend = 'data', which_fit = 'best', e_min = None, e_max = None, g1_guess = -1.9, g2_guess = -2.5, g3_guess = -4, c1_guess = 1000, alpha_guess = 10, beta_guess = 10, break_guess_low = 0.6, break_guess_high = 1.2, cut_guess = 1.2, exponent_guess = 2, use_random = True, iterations = 20 , legend_details = False):
    """This function calls the make_the_fit functoin that creates the fit. It plots and saves the results of the fit.

    Args:
        path (string): The path to the folder where the fit results and plots will be saved.
        data (dataframe): The data that will be fit (energy, energy uncertainty, intensity and intensity uncertainty)
        save (bool): if True the plots and fit results will be saved. Note: the title of the plot will be used as the file name
        channels_to_exclude (list): Defaults to 'None'
        plot_title (str, optional): The title of the plot, will also be used when saving the results. Defaults to ''.
        x_label (str, optional): label for the x axis. Defaults to 'Intensity [/]'.
        y_label (str, optional): label for the y axis. Defaults to 'Energy [MeV]'.
        legend_title (str, optional): title for the legend. Defaults to ''.
        which_fit (str, optional): _description_. Defaults to 'best'.
        e_min (_type_, optional): _description_. Defaults to None.
        e_max (_type_, optional): _description_. Defaults to None.
        g1_guess (float, optional): _description_. Defaults to -1.9.
        g2_guess (float, optional): _description_. Defaults to -2.5.
        g3_guess (int, optional): _description_. Defaults to -4.
        c1_guess (int, optional): _description_. Defaults to 1000.
        alpha_guess (int, optional): _description_. Defaults to 10.
        beta_guess (int, optional): _description_. Defaults to 10.
        break_guess_low (float, optional): _description_. Defaults to 0.6.
        break_guess_high (float, optional): _description_. Defaults to 1.2.
        cut_guess (float, optional): _description_. Defaults to 1.2.
        use_random (bool, optional): _description_. Defaults to True.
        iterations (int, optional): _description_. Defaults to 20.
    """
    
    
    # in make the fit we have two paths. one for pickle files (deleted from here) and path2 to save the fit variables.
    title_from_path = path
    # TO DO: CHANGE IN CASE THERE ARE BLANK SPACES IN THE TITLE THE NAME WILL NOT WORK. REPLACE ' ' WITH '-' DONE BUT TEST
    name_string = plot_title.replace(" ", "_")

    fit_var_path = title_from_path+'-'+name_string+'-fit-result-variables_'+which_fit+'.csv'

    all_data = data

    dataframe_to_fit = data
    dataframe_to_exclude = pd.DataFrame()

    if channels_to_exclude != None:
        args = sf.exclude_channels(data, channels_to_exclude)
        dataframe_to_fit = args[0]
        dataframe_to_exclude = args[1]


    x_data = dataframe_to_fit['Energy'] # energy for spectra
    x_err  = dataframe_to_fit['E_err']
    y_data   = dataframe_to_fit['Intensity']
    y_err    = dataframe_to_fit['I_err']   
 

    f, ax = plt.subplots(1, figsize=(6, 5), dpi = 300)
    
    fitting.MAKE_THE_FIT(x_data, y_data, x_err, None, ax, direction='sun', e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, g3_guess = g3_guess, alpha_guess=alpha_guess, beta_guess = beta_guess, break_low_guess=break_guess_low, break_high_guess = break_guess_high, cut_guess = cut_guess, c1_guess = c1_guess, exponent_guess = exponent_guess, use_random = use_random, iterations = iterations, path = None, path2 = fit_var_path, detailed_legend = legend_details)
	                    

    #colors = ['red', 'darkorange', 'marroon', 'blue']
    #print(data)
    #print(data[0])
    #print(data[0]['x'])
    #print(all_data)


    #for i in range(len(data)):
    #    x_data = data[i]['x'] # energy for spectra
    #    x_data_err  = data[i]['x error']
    #   y_data   = data[i]['y']
    #   y_data_err    = data[i]['y error']
        #print(x_data)    
    ax.errorbar(x_data, y_data, xerr = x_err, yerr=None, marker='o', markersize= 3 , linestyle='', color='red', alpha = 0.5, label=data_label_for_legend, zorder = -1)
    if channels_to_exclude != None:
        ax.errorbar(dataframe_to_exclude['Energy'], dataframe_to_exclude['Intensity'], xerr = dataframe_to_exclude['E_err'], yerr=dataframe_to_exclude['I_err'], marker='o', markersize= 3 , linestyle='', color='gray', alpha = 0.5, label='excluded channels', zorder = -1)
   


    x_range_min = min(all_data['Energy'])
    x_range_max = max(all_data['Energy'])
    

    ax.set_xscale('log')
    ax.set_yscale('log')
        
    ax.set_xlim(x_range_min-(x_range_min/2), x_range_max+(x_range_max/2))
        #ax.set_ylim(y_range_min-(y_range_min/2), y_range_max+(y_range_max/2))
    
    locmin = pltt.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
        
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(pltt.NullFormatter())

    plt.legend(title=''+legend_title+'',  prop={'size': 7})
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(plot_title)
    
    if save:
        plt.savefig(title_from_path+'-'+name_string+'-fit-plot_'+which_fit+'.png', dpi=300)

    plt.show()



