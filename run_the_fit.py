import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as pltt
from sunpy.coordinates import get_horizons_coord
from make_the_fit import  *
#from combining_files import *
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# Import mymodule
from savecsv import *
from combining_files import *


# <--------------------------------------------------------------- ALL NECESSARY INPUTS HERE ----------------------------------------------------------------->

# INITIAL INPUTS TO LOAD FILES
# The path to the folder where the data is stored
# this path is also used to create new files for all and contaminated data.
#                C:\Users\Omistaja\Desktop\SRL\2021SRL\epd_plot-main\solo_loader-main-shift\csv\18-Nov-20 1420-two-slopes
def FIT_DATA(path, date, averaging, fit_type, step = True, ept = True, het = True, which_fit = 'best', sigma = 3, rel_err = 0.5, frac_nan_threshold = 0.9, fit_to = 'peak', slope = None, e_min = None, e_max = None, g1_guess = -1.9, g2_guess = -2.5, g3_guess = -4, c1_guess = 1000, alpha_guess = 10, beta_guess = 10, break_guess_low = 0.6, break_guess_high = 1.2, cut_guess = 1.2, use_random = True, iterations = 20, leave_out_1st_het_chan = True, shift_step_data = False, shift_factor = None, save_fig = True, save_pickle = False, save_fit_variables = True, save_fitrun = True):
	"""_summary_

	Args:
		path (_type_): _description_
		date (_type_): _description_
		averaging (_type_): _description_
		fit_type (_type_): _description_
		step (bool, optional): _description_. Defaults to True.
		ept (bool, optional): _description_. Defaults to True.
		het (bool, optional): _description_. Defaults to True.
		which_fit (str, optional): _description_. Defaults to 'best'.
		sigma (int, optional): _description_. Defaults to 3.
		rel_err (float, optional): _description_. Defaults to 0.5.
		frac_nan_threshold (float, optional): _description_. Defaults to 0.9.
		fit_to (str, optional): _description_. Defaults to 'peak'.
		slope (_type_, optional): _description_. Defaults to None.
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
		leave_out_1st_het_chan (bool, optional): _description_. Defaults to True.
		shift_step_data (bool, optional): _description_. Defaults to False.
		shift_factor (_type_, optional): _description_. Defaults to None.
		save_fig (bool, optional): _description_. Defaults to True.
		save_pickle (bool, optional): _description_. Defaults to False.
		save_fit_variables (bool, optional): _description_. Defaults to True.
		save_fitrun (bool, optional): _description_. Defaults to True.
	"""
		
	#folder = '20210507-1852/2min/sun/'

	date_string = str(date) #'2021-05-07'
	averaging = str(averaging)+'min'

	#path = path+folder
	

	#step = True
	#ept  = True
	#het  = True

	separator = ';'

	step_file_name = 'electron_data-'+date_string+'-STEP-L2-'+averaging+'_averaging.csv'
	ept_file_name = 'electron_data-'+date_string+'-EPT-L2-'+averaging+'_averaging-ion_corr.csv'
	het_file_name = 'electron_data-'+date_string+'-HET-L2-'+averaging+'_averaging.csv'


	# You can choose the sigma threshold value below which the data points will not be included in the fit
	# You can also choose to leave the first HET channel out from the fit by setting  leave_out_1st_het_chan to True
	# You can choose to shift step data by a certain factor (I have used 0.8)

	#sigma = sigma
	#rel_err = 0.5
	#frac_nan_threshold = 0.9
	#leave_out_1st_het_chan = True
	#shift_step_data = False
	#shift_factor = None #0.8

	# !!! INPUTS FOR THE FIT !!!

	#fit_type = 'step_ept_het' # fit_type options: step, ept, het, step_ept, step_ept_het, ept_het
	#fit_to = 'peak'   # 'peak' or 'average'for window peak or average
	window_type = 'One slope D = 1.6 AU' #'two slopes D = 1.191 AU & 1.7 AU'
	slope = 'slope16'

	# which_fit options: 
	# 'single' will force a single pl fit to the data
	# 'broken' will force a broken pl fit to the data but ONLY if the break point is within the energy range otherwise a sigle pl fit will be produced instead
	# 'best_sb' will choose automatically the best fit type between single and broken by comparing the redchis of the fits
	# 'best' will choose automatically the best fit type by comparing the redchis of the fits
	# 'cut' will produce a single pl fit with an exponential cutoff point. If the cutoff point is outside of the energy range a broken or single pl will be fit instead
	# 'broken_cut' will produce a broken pl fit with an exponential cutoff point. If the cutoff point is outside of the energy range a broken or single pl will be fit instead
	# maybe option to just check between broken and cut 
	# 'best_cb' 


	#which_fit ='broken_cut' 

	#!!! e_min, e_max, break_guess and cut_guess SHOULD BE IN MeV !!!
	# step energy range: 0.004323343613-0.07803193193
	# ept energy range: 0.03295087252-0.452730295
	# het energy range: 0.6859485403-10.62300288
	# second het channel: 1.590048112
	# e_min and e_max can also be None. In this case the MAKE_THE_FIT function will automaically choose the values

	#e_min = None # in MeV
	#e_max =	None # in MeV
	#g1_guess = -1.9
	#g2_guess = -2.5
	#c1_guess = 1e3
	#alpha_guess = 7.16
	#break_guess_low = 0.1#in MeV
	#cut_guess = 0.12#in MeV

	# if use_random = False the fit will only be made once with the guess values
	# if use_random = True 
	# the fitting function will first create a list of reasonable values for each of the fitting parameters 
	# then randomly select values from the lists and compare the redchis of each fit to find the best one
	#use_random = True 
	#iterations = 20


	#save_fig = True # save the fit
	#save_pickle = True # save a pickle file of the odr run
	#save_fit_variables = True # save the variables from the fit
	#save_fitrun = True # save all variables used for the fit
	# <-------------------------------------------------------------- END OF NECESSARY INPUTS ---------------------------------------------------------------->


	make_fit = True
	peak_spec = True
	backsub = True

	fit_to_comb = fit_to[0].upper()+fit_to[1:]

	direction = 'sun'
	intensity_label = 'Flux\n/(s cm² sr MeV)'
	energy_label = 'primary energy (MeV)'
	peak_info = fit_to+' spec'+'\n'+window_type
	legend_title = 'Electrons'  # 'mag' or 'foil' or 'Electrons' if there is more than just ept data
	data_product = 'l2'

	date_str = date_string[8:]+'-'+date_string[5:7]+'-'+date_string[0:4] #DO NOT CHANGE. This is used later for the plot title etc.
	pos = get_horizons_coord('Solar Orbiter', date_string, 'id')
	dist = np.round(pos.radius.value, 2)
	#dist = ''

	# <---------------------------------------------------------------LOADING AND SAVING FILES------------------------------------------------------------------->

	#print(path_to_file+step_file_name)
	data_list = []
	if step :
		step_data = pd.read_csv(path+step_file_name, sep = separator)
		if shift_step_data:
			step_data['Bg_subtracted_'+fit_to] = shift_factor*step_data['Bg_subtracted_'+fit_to]
		data_list.append(step_data)
		
	if ept :
		ept_data = pd.read_csv(path+ept_file_name, sep = separator)
		data_list.append(ept_data)
	if het :
		het_data = pd.read_csv(path+het_file_name, sep = separator)
		data_list.append(het_data)

	data = combine_data(data_list, path+date_string+'-all-l2-'+averaging+'.csv', sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = leave_out_1st_het_chan, fit_to = fit_to_comb)
	data = pd.read_csv(path+date_string+'-all-l2-'+averaging+'.csv', sep = separator)

	if step and ept:
		step_ept_data = combine_data([step_data, ept_data], path+date_string+'-step_ept-l2-'+averaging+'.csv', sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = leave_out_1st_het_chan, fit_to = fit_to_comb)
		step_ept_data = pd.read_csv(path+date_string+'-step_ept-l2-'+averaging+'.csv', sep = separator)

	if ept and het:
		ept_het_data = combine_data([ept_data, het_data], path+date_string+'-ept_het-l2-'+averaging+'.csv', sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = leave_out_1st_het_chan, fit_to = fit_to_comb)
		ept_het_data = pd.read_csv(path+date_string+'-ept_het-l2-'+averaging+'.csv', sep = separator)

	# saving the contaminated data so it can be plotted separately
	# then deleting it from the data so it doesn't overlap
	contaminated_data_sigma = low_sigma_threshold(data_list, sigma = sigma, leave_out_1st_het_chan = leave_out_1st_het_chan, fit_to = fit_to_comb)
	contaminated_data_nan   = too_many_nans(data_list, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = leave_out_1st_het_chan)
	contaminated_data_rel_err = high_rel_err(data_list, rel_err = rel_err, leave_out_1st_het_chan = leave_out_1st_het_chan)
	contaminated_data = pd.concat([contaminated_data_sigma, contaminated_data_nan, contaminated_data_rel_err ])
	contaminated_data.reset_index(drop=True, inplace=True)

	#deleting low sigma data so it doesn't overplot 

	if step:
		step_data = delete_bad_data(step_data, sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, fit_to = fit_to_comb)
	if ept:
		ept_data = delete_bad_data(ept_data, sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, fit_to = fit_to_comb)
	if het:
		first_het_data = first_het_chan(het_data)
		het_data = delete_bad_data(het_data, sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = leave_out_1st_het_chan, fit_to = fit_to_comb)
		

	#-------------------------------------------------------------------------------------------------------------------------------------------------
	color = {'sun':'crimson','asun':'orange', 'north':'darkslateblue', 'south':'c'}
	prim_e = data['Primary_energy']


	pickle_path = None
	if save_pickle:
		pickle_path = path+date_string+'-pickle_'+fit_type+'-'+fit_to+'-'+which_fit+'-l2-'+averaging+'-'+direction+'.p'

	fit_var_path = None
	if save_fit_variables:
		fit_var_path = path+date_string+'-fit-result-variables_'+fit_type+'-'+fit_to+'-'+which_fit+'-l2-'+averaging+'-'+direction+'.csv'

	fitrun_path = None
	if save_fitrun:
		fitrun_path = path+date_string+'-all-fit-variables_'+fit_type+'-'+fit_to+'-'+which_fit+'-l2-'+averaging+'-'+direction+'.csv'
		
		save_info_fit(fitrun_path, date_string, averaging, direction, data_product, dist, step, ept, het,
		sigma, rel_err, frac_nan_threshold, leave_out_1st_het_chan, shift_factor, fit_type, fit_to,
		window_type, which_fit, e_min, e_max, g1_guess, g2_guess, c1_guess, alpha_guess, break_guess_low, cut_guess,
		use_random, iterations)



	# <---------------------------------------------------------------------DATA--------------------------------------------------------------------->

	# all data
	spec_energy = data['Primary_energy']
	energy_err_low  = data['Energy_error_low']
	energy_err_high = data['Energy_error_high']
	spec_flux   = data['Bg_subtracted_'+fit_to]
	flux_err    = data['Backsub_peak_uncertainty']

	energy_err = [energy_err_low, energy_err_high]

	# step ept
	if step and ept:
		spec_energy_step_ept = step_ept_data['Primary_energy']
		energy_err_low_step_ept  = step_ept_data['Energy_error_low']
		energy_err_high_step_ept = step_ept_data['Energy_error_high']
		spec_flux_step_ept   = step_ept_data['Bg_subtracted_'+fit_to]
		flux_err_step_ept    = step_ept_data['Backsub_peak_uncertainty']

		energy_err_step_ept = [energy_err_low_step_ept, energy_err_high_step_ept]

	if ept and het:
		spec_energy_ept_het = ept_het_data['Primary_energy']
		energy_err_low_ept_het  = ept_het_data['Energy_error_low']
		energy_err_high_ept_het = ept_het_data['Energy_error_high']
		spec_flux_ept_het   = ept_het_data['Bg_subtracted_'+fit_to]
		flux_err_ept_het    = ept_het_data['Backsub_peak_uncertainty']

		energy_err_ept_het = [energy_err_low_ept_het, energy_err_high_ept_het]

	# step data
	if step:
		spec_energy_step = step_data['Primary_energy']
		energy_err_low_step  = step_data['Energy_error_low']
		energy_err_high_step = step_data['Energy_error_high']
		spec_flux_step   = step_data['Bg_subtracted_'+fit_to]
		flux_err_step    = step_data['Backsub_peak_uncertainty']

		energy_err_step = [energy_err_low_step, energy_err_high_step]

	# ept data
	if ept:
		spec_energy_ept = ept_data ['Primary_energy']
		energy_err_low_ept   = ept_data ['Energy_error_low']
		energy_err_high_ept  = ept_data ['Energy_error_high']
		spec_flux_ept    = ept_data ['Bg_subtracted_'+fit_to]
		flux_err_ept     = ept_data ['Backsub_peak_uncertainty']

		energy_err_ept  = [energy_err_low_ept, energy_err_high_ept]

	# het data
	if het:
		spec_energy_het = het_data['Primary_energy']
		energy_err_low_het  = het_data['Energy_error_low']
		energy_err_high_het = het_data['Energy_error_high']
		spec_flux_het  = het_data['Bg_subtracted_'+fit_to]
		flux_err_het    = het_data['Backsub_peak_uncertainty']

		energy_err_het = [energy_err_low_het, energy_err_high_het]
		

	# contaminated data 
	spec_energy_c = contaminated_data['Primary_energy']
	energy_err_low_c  = contaminated_data['Energy_error_low']
	energy_err_high_c = contaminated_data['Energy_error_high']
	spec_flux_c  = contaminated_data['Bg_subtracted_'+fit_to]
	flux_err_c   = contaminated_data['Backsub_peak_uncertainty']


	# contaminated data sigma
	spec_energy_c_sigma = contaminated_data_sigma['Primary_energy']
	energy_err_low_c_sigma  = contaminated_data_sigma['Energy_error_low']
	energy_err_high_c_sigma = contaminated_data_sigma['Energy_error_high']
	spec_flux_c_sigma  = contaminated_data_sigma['Bg_subtracted_'+fit_to]
	flux_err_c_sigma   = contaminated_data_sigma['Backsub_peak_uncertainty']

	# contaminated data nan
	spec_energy_c_nan = contaminated_data_nan['Primary_energy']
	energy_err_low_c_nan  = contaminated_data_nan['Energy_error_low']
	energy_err_high_c_nan = contaminated_data_nan['Energy_error_high']
	spec_flux_c_nan  = contaminated_data_nan['Bg_subtracted_'+fit_to]
	flux_err_c_nan   = contaminated_data_nan['Backsub_peak_uncertainty']

	# contaminated data rel err
	spec_energy_c_rel_err = contaminated_data_rel_err['Primary_energy']
	energy_err_low_c_rel_err  = contaminated_data_rel_err['Energy_error_low']
	energy_err_high_c_rel_err = contaminated_data_rel_err['Energy_error_high']
	spec_flux_c_rel_err  = contaminated_data_rel_err['Bg_subtracted_'+fit_to]
	flux_err_c_rel_err   = contaminated_data_rel_err['Backsub_peak_uncertainty']

	energy_err_c = [energy_err_low_c, energy_err_high_c]
	energy_err_c_sigma = [energy_err_low_c_sigma, energy_err_high_c_sigma]
	energy_err_c_nan = [energy_err_low_c_nan, energy_err_high_c_nan]
	energy_err_c_rel_err = [energy_err_low_c_rel_err, energy_err_high_c_rel_err]



	if het and leave_out_1st_het_chan:
		# first het
		spec_energy_first_het = first_het_data['Primary_energy']
		energy_err_low_first_het  = first_het_data['Energy_error_low']
		energy_err_high_first_het = first_het_data['Energy_error_high']
		spec_flux_first_het  = first_het_data['Bg_subtracted_'+fit_to]
		flux_err_first_het   = first_het_data['Backsub_peak_uncertainty']

		energy_err_first_het = [energy_err_low_first_het, energy_err_high_first_het]


	# <----------------------------------------------------------------------FIT AND PLOT------------------------------------------------------------------->

	f, ax = plt.subplots(1, figsize=(8, 6), dpi = 200)

	#distance  = ''
	distance = f' (R={dist} au)'
	#ax.errorbar(spec_energy_first_het, spec_flux_first_het, yerr=flux_err_first_het, xerr = energy_err_first_het, marker='o', linestyle='', markersize= 3, color='blue', label='First HET channel', zorder = -1)


	if make_fit:
		if fit_type == 'step':
			plot_title = 'SolO '+ distance+' STEP'
			fit_result = MAKE_THE_FIT(spec_energy_step, spec_flux_step, energy_err_step[1], flux_err_step, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, alpha_guess=alpha_guess, break_guess=break_guess_low, cut_guess = cut_guess, c1_guess=c1_guess,use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path)
		if fit_type == 'ept':
			plot_title = 'SolO '+ distance+' EPT'
			fit_result = MAKE_THE_FIT(spec_energy_ept, spec_flux_ept, energy_err_ept[1], flux_err_ept, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, alpha_guess=alpha_guess, break_guess=break_guess_low, cut_guess = cut_guess, c1_guess=c1_guess,use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path)
		if fit_type == 'het':
			plot_title = 'SolO '+ distance+' HET'
			fit_result = MAKE_THE_FIT(spec_energy_het, spec_flux_het, energy_err_het[1], flux_err_het, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit='single', g1_guess=g1_guess, g2_guess=g2_guess, alpha_guess=alpha_guess, break_guess=break_guess_low, cut_guess = cut_guess,  c1_guess=c1_guess,use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path)
		if fit_type == 'step_ept':
			plot_title = 'SolO '+ distance+' STEP and EPT'
			fit_result = MAKE_THE_FIT(spec_energy_step_ept, spec_flux_step_ept, energy_err_step_ept[1], flux_err_step_ept, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, alpha_guess=alpha_guess, break_guess=break_guess_low, cut_guess = cut_guess, c1_guess=c1_guess,use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path)
		if fit_type == 'ept_het':
			plot_title = 'SolO '+ distance+' EPT and HET'
			fit_result = MAKE_THE_FIT(spec_energy_ept_het, spec_flux_ept_het, energy_err_ept_het[1], flux_err_ept_het, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, alpha_guess=alpha_guess, break_guess=break_guess_low, cut_guess = cut_guess, c1_guess=c1_guess,use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path)
		if fit_type == 'step_ept_het':
			plot_title = 'SolO '+ distance+' STEP, EPT and HET'
			fit_result = MAKE_THE_FIT(spec_energy, spec_flux, energy_err[1], flux_err, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, alpha_guess=alpha_guess, break_guess=break_guess_low, cut_guess = cut_guess, c1_guess=c1_guess,use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path)
		if step:
			ax.errorbar(spec_energy_step, spec_flux_step, yerr=flux_err_step, xerr = energy_err_step, marker='o', linestyle='',markersize= 3 ,  color='darkorange', label='STEP', zorder = -1)
		if ept:
			ax.errorbar(spec_energy_ept, spec_flux_ept, yerr=flux_err_ept, xerr = energy_err_ept, marker='o', linestyle='', markersize= 3, color=color[direction], label='EPT '+direction, zorder = -1)
		if het:
			ax.errorbar(spec_energy_het, spec_flux_het, yerr=flux_err_het, xerr = energy_err_het, marker='o', linestyle='', markersize= 3, color='maroon', label='HET '+direction, zorder = -1)
			if leave_out_1st_het_chan:
				ax.errorbar(spec_energy_first_het, spec_flux_first_het, yerr=flux_err_first_het, xerr = energy_err_first_het, marker='o', linestyle='', markersize= 3, color='black', label='First HET channel', zorder = -1)
		ax.errorbar(spec_energy_c, spec_flux_c, yerr=flux_err_c, xerr = energy_err_c, marker='o', linestyle='', markersize= 3, color='gray', label='excluded from fit', zorder = -1)
		

	if make_fit is False:
		ax.errorbar(spec_energy_c, spec_flux_c, yerr=flux_err_c, xerr = energy_err_c, marker='o', linestyle='', markersize= 3, color='gray', label = 'cont. data', zorder = -1)
		if step:
			ax.errorbar(spec_energy_step, spec_flux_step, yerr=flux_err_step, xerr = energy_err_step, marker='o', markersize= 3 , linestyle='', color='darkorange', label='STEP', zorder = -1)
			if ept and het:
				plot_title = 'SolO STEP, EPT and HET'
			if ept and het is False:
				plot_title = 'SolO STEP and EPT'
			if het and ept is False:
				plot_title = 'SolO STEP and HET'
			if ept is False and het is False:
				plot_title = 'SolO STEP'
		if ept:
			ax.errorbar(spec_energy_ept, spec_flux_ept, yerr=flux_err_ept, xerr = energy_err_ept, marker='o', linestyle='', markersize= 3, color=color[direction], label='EPT '+direction, zorder = -1)
			if het and step is False:
				plot_title = 'SolO EPT and HET'
			if step is False and het is False:
				plot_title = 'SolO EPT'
		if het:
			ax.errorbar(spec_energy_het, spec_flux_het, yerr=flux_err_het, xerr = energy_err_het, marker='o', linestyle='', markersize= 3, color='maroon', label='HET '+direction, zorder = -1)
			if ept is False and step is False:
				plot_title = 'SolO HET'
			if leave_out_1st_het_chan:
				ax.errorbar(spec_energy_first_het, spec_flux_first_het, yerr=flux_err_first_het, xerr = energy_err_first_het, marker='o', linestyle='', markersize= 3, color='black', label='First HET channel', zorder = -1)
				

		
	# for a more deailed version uncomment
	#ax.errorbar(spec_energy_c_sigma, spec_flux_c_sigma, yerr=flux_err_c_sigma, xerr = energy_err_c_sigma, marker='o', linestyle='', markersize= 3, color='blue', label='Sigma below '+str(sigma), zorder = -1)
	#ax.errorbar(spec_energy_c_nan, spec_flux_c_nan, yerr=flux_err_c_nan, xerr = energy_err_c_nan, marker='o', linestyle='', markersize= 3, color='gray', label='excluded (NaNs)', zorder = -1)
	#ax.errorbar(spec_energy_c_rel_err, spec_flux_c_rel_err, yerr=flux_err_c_rel_err, xerr = energy_err_c_rel_err, marker='o', linestyle='', markersize= 3, color='purple', label='excluded (rel err)', zorder = -1)


	if backsub:
		if step:
			ax.errorbar(spec_energy_step, step_data['Background_flux'], yerr=step_data['Bg_electron_uncertainty'], xerr = energy_err_step, marker='o', markersize= 3, linestyle='', color='darkorange', alpha=0.3)
		if ept:
			ax.errorbar(spec_energy_ept, ept_data['Background_flux'], yerr=ept_data['Bg_electron_uncertainty'], xerr = energy_err_ept, marker='o', markersize= 3, linestyle='', color=color[direction], alpha=0.3)
		if het:
			ax.errorbar(spec_energy_het, het_data['Background_flux'], yerr=het_data['Bg_electron_uncertainty'], xerr = energy_err_het, marker='o', markersize= 3, linestyle='', color='maroon', alpha=0.3)
		#ax.errorbar(spec_energy_c_sigma, contaminated_data_sigma['Background_flux'], yerr=contaminated_data_sigma['Bg_electron_uncertainty'], xerr = energy_err_c_sigma, marker='o', markersize= 3, linestyle='', color='blue', alpha=0.3)
		#ax.errorbar(spec_energy_c_nan, contaminated_data_nan['Background_flux'], yerr=contaminated_data_nan['Bg_electron_uncertainty'], xerr = energy_err_c_nan, marker='o', markersize= 3, linestyle='', color='gray', alpha=0.3)
		#ax.errorbar(spec_energy_c_rel_err, contaminated_data_rel_err['Background_flux'], yerr=contaminated_data_rel_err['Bg_electron_uncertainty'], xerr = energy_err_c_rel_err, marker='o', markersize= 3, linestyle='', color='purple', alpha=0.3)
		if make_fit:
			ax.errorbar(spec_energy_c, contaminated_data['Background_flux'], yerr=contaminated_data['Bg_electron_uncertainty'], xerr = energy_err_c, marker='o', markersize= 3, linestyle='', color='gray', alpha=0.3)
			if het and leave_out_1st_het_chan:
				ax.errorbar(spec_energy_first_het, first_het_data['Background_flux'], yerr=first_het_data['Bg_electron_uncertainty'], xerr = energy_err_first_het, marker='o', markersize= 3, linestyle='', color='black', alpha=0.3)
		
		if make_fit is False:
			ax.errorbar(spec_energy_c, contaminated_data['Background_flux'], yerr=contaminated_data['Bg_electron_uncertainty'], xerr = energy_err_c, marker='o', markersize= 3, linestyle='', color='maroon', alpha=0.3)
		
	e_range_min = None
	e_range_max = None	
	step_energy_range = [0.004323343613,0.07803193193]
	ept_energy_range =  [0.03295087252,0.452730295]
	het_energy_range =  [0.6859485403,10.62300288]
	#if fit_type == 'step':
	#	e_range_min = step_energy_range[0]
	#	e_range_max = step_energy_range[1]
	#if fit_type == 'step_ept':
	#	e_range_min = step_energy_range[0]
	#	e_range_max = ept_energy_range[1]
	#if fit_type == 'step_ept_het':
	#	e_range_min = step_energy_range[0]
	#	e_range_max = het_energy_range[1]
	#if fit_type == 'ept':
	#	e_range_min = ept_energy_range[0]
	#	e_range_max = ept_energy_range[1]
	#if fit_type == 'ept_het':
	#	e_range_min = ept_energy_range[0]
	#	e_range_max = het_energy_range[1]
	#if fit_type =='het':
	#	e_range_min = het_energy_range[0]
	#	e_range_max = het_energy_range[1]

	e_range_min = step_energy_range[0]
	e_range_max = het_energy_range[1]

	ax.set_xscale('log')
	ax.set_yscale('log')
	locmin = pltt.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
	ax.set_xlim(e_range_min-(e_range_min/2), e_range_max+(e_range_max/2))
	ax.yaxis.set_minor_locator(locmin)
	ax.yaxis.set_minor_formatter(pltt.NullFormatter())


	plt.legend(title='"'+legend_title+'"',  prop={'size': 7})
	plt.ylabel(intensity_label)
	plt.xlabel(energy_label)
	plt.title(plot_title+'  '+peak_info+'\n'+date_str+'  '+averaging+'  averaging')

	if save_fig:
		if make_fit:
			plt.savefig(path+date_string+'-'+direction+'-'+averaging+'-'+which_fit+'-'+fit_type+'-'+slope+'-'+fit_to, dpi=300)
		if make_fit is False:
			if step and ept and het:
				plt.savefig(path+date_string+'-'+averaging+'-no_fit-'+slope+'-step_ept_het', dpi=300)
			if step and ept and het is False:
				plt.savefig(path+date_string+'-'+averaging+'-no_fit-'+slope+'-step_ept', dpi=300)
			if step and het and ept is False:
				plt.savefig(path+date_string+'-'+averaging+'-no_fit-'+slope+'-step_het', dpi=300)
			if step and ept is False and het is False:
				plt.savefig(path+date_string+'-'+averaging+'-no_fit-'+slope+'-step', dpi=300)
			if ept and het and step is False:
				plt.savefig(path+date_string+'-'+averaging+'-no_fit-'+slope+'-ept_het', dpi=300)
			if ept and step is False and het is False:
				plt.savefig(path+date_string+'-'+averaging+'-no_fit-'+slope+'-ept', dpi=300)
			if het and ept is False and step is False:
				plt.savefig(path+date_string+'-'+averaging+'-no_fit-'+slope+'-het', dpi=300)
			
			


	plt.show()
