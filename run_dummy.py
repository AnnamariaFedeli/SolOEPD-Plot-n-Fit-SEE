from turtle import title
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as pltt
from sunpy.coordinates import get_horizons_coord
from make_the_fit import  MAKE_THE_FIT
from make_the_fit import closest_values
from make_the_fit import find_c1
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def run_all(path_to_file, path_to_savefig,savefig, save_pickle, save_fit_variables, uncertainty_data_included, plot_title, x_label, y_label, legend_title, which_fit, e_min, e_max, g1_guess, g2_guess, c1_guess, alpha_guess, break_guess, cut_guess, use_random, iterations,  ):
    make_fit = True
    data = pd.read_csv(path_to_file, skiprows = 0, sep = ';')
    if uncertainty_data_included == True:
        data.columns = ['x', 'y', 'x error', 'y error']

    if uncertainty_data_included == False:
        data.columns = ['x',  'y']
        data['x error'] = data['x']*0.1
        data['y error'] = data['y']*0.1
        
    pickle_path = None
    if save_pickle:
    	pickle_path = path_to_file+title+'-pickle_'+'-'+which_fit+'.p'
    
    fit_var_path = None
    if save_fit_variables:
    	fit_var_path = path_to_file+title+'-fit-result-variables_'+which_fit+'.csv'

    spec_energy = data['x']
    energy_err_low  = data['x error']
    energy_err_high = data['x error']
    spec_flux   = data['y']
    flux_err    = data['y error']
    
    f, ax = plt.subplots(1, figsize=(6, 5), dpi = 200)


    if make_fit:
        fit_result = MAKE_THE_FIT(spec_energy, spec_flux, energy_err_low, flux_err, ax, direction='sun', e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, alpha_guess=alpha_guess, break_guess=break_guess, cut_guess = cut_guess, c1_guess=c1_guess,use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path)
	
    ax.errorbar(spec_energy, spec_flux, yerr=flux_err, xerr = energy_err_low, marker='o', markersize= 3 , linestyle='', color='darkorange', alpha = 0.5, label='Data', zorder = -1)

    e_range_min = data['x'][0]
    e_range_max = data['x'][len(data['x'])-1]

    ax.set_xscale('log')
    ax.set_yscale('log')
    #locmin = pltt.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
    ax.set_xlim(e_range_min-(e_range_min/2), e_range_max+(e_range_max/2))
    ax.set_ylim(1e5, 1.1e8)
    #ax.yaxis.set_minor_locator(locmin)
    #ax.yaxis.set_minor_formatter(pltt.NullFormatter())


    plt.legend(title='"'+legend_title+'"',  prop={'size': 7})
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(plot_title)
    #+'  '+peak_info+'\n'+date_str+'  '+averaging+'  averaging')

    if savefig:
	    plt.savefig(path_to_savefig+title, dpi=300)

    plt.show()


