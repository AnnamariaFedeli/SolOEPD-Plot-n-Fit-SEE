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

def subtract_pl_fit_intensities(x):
    return x


def run_all(path, data, savefig, plot_title = '', x_label = 'Intensity [/]', y_label = 'Energy [MeV]', legend_title = '', data_label_for_legend = ['data'], which_fit = 'best', e_min = None, e_max = None, g1_guess = -1.9, g2_guess = -2.5, g3_guess = -4, c1_guess = 1000, alpha_guess = 10, beta_guess = 10, break_guess_low = 0.6, break_guess_high = 1.2, cut_guess = 1.2, exponent_guess = 2, use_random = True, iterations = 20 , legend_details = False):
    """_summary_

    Args:
        path (_type_): _description_
        data (_type_): _description_
        savefig (_type_): _description_
        plot_title (str, optional): _description_. Defaults to ''.
        x_label (str, optional): _description_. Defaults to 'Intensity [/]'.
        y_label (str, optional): _description_. Defaults to 'Energy [MeV]'.
        legend_title (str, optional): _description_. Defaults to ''.
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
    title_from_path = path[:-4]
    fit_var_path = title_from_path+'-'+plot_title+'-fit-result-variables_'+which_fit+'.csv'

    all_data = data
    


    if len(data)>1:
        all_data = comb.combine_data_general(data, path+'-combined-data-'+plot_title+'-'+which_fit+'.csv')
        all_data.columns = ['x', 'y', 'x error', 'y error']

    if len(data)==1:
        all_data = data[0]
        all_data.columns = ['x', 'y', 'x error', 'y error']
            
    #print(all_data)
    bad_data = all_data.index[all_data['y']<=0].tolist()
    all_data = all_data.drop(bad_data, axis = 0)
    all_data.reset_index(drop=True, inplace=True)   

    x_data = all_data['x'] # energy for spectra
    x_data_err  = all_data['x error']
    y_data   = all_data['y']
    y_data_err    = all_data['y error']   
    

    #print(all_data)


    f, ax = plt.subplots(1, figsize=(6, 5), dpi = 200)
    
    fitting.MAKE_THE_FIT(x_data, y_data, x_data_err, y_data_err, ax, direction='sun', e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, g3_guess = g3_guess, alpha_guess=alpha_guess, beta_guess = beta_guess, break_low_guess=break_guess_low, break_high_guess = break_guess_high, cut_guess = cut_guess, c1_guess = c1_guess, exponent_guess = exponent_guess, use_random = use_random, iterations = iterations, path = None, path2 = fit_var_path, detailed_legend = legend_details)
	                    #spec_energy_step_ept, spec_flux_step_ept, energy_err_step_ept[1], flux_err_step_ept, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, g3_guess = g3_guess, alpha_guess=alpha_guess, beta_guess = beta_guess, break_low_guess=break_guess_low, break_high_guess = break_guess_high, cut_guess = cut_guess, c1_guess=c1_guess,use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path, detailed_legend = legend_details)


    colors = ['red', 'darkorange', 'marroon', 'blue']
    #print(data)
    #print(data[0])
    #print(data[0]['x'])
    print(all_data)

    for i in range(len(data)):
        x_data = data[i]['x'] # energy for spectra
        x_data_err  = data[i]['x error']
        y_data   = data[i]['y']
        y_data_err    = data[i]['y error']
        #print(x_data)    
        ax.errorbar(x_data, y_data, yerr=y_data_err, xerr = x_data_err, marker='o', markersize= 3 , linestyle='', color=colors[i], alpha = 0.5, label=data_label_for_legend[i], zorder = -1)

    x_range_min = min(all_data['x'])
    x_range_max = max(all_data['x'])
    

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
    
    if savefig:
        plt.savefig(title_from_path+'-'+plot_title+'-fit-plot_'+which_fit+'.png', dpi=300)

    plt.show()



