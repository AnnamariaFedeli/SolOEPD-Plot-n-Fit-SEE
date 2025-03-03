import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as pltt
from sunpy.coordinates import get_horizons_coord
#from make_the_fit import  *
#from combining_files import *
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# Import mymodule
#from savecsv import *
#from combining_files import *
import make_the_fit_tripl as fitting
import savecsv as save
import combining_files as comb
from datetime import *
import os
import shutil
# <--------------------------------------------------------------- ALL NECESSARY INPUTS HERE ----------------------------------------------------------------->

#source_folder = r"E:\demos\files\reports\\"
#destination_folder = r"E:\demos\files\account\\"

# fetch all files
#for file_name in os.listdir(source_folder):
    # construct full file path
#    source = source_folder + file_name
#    destination = destination_folder + file_name
    # copy only files
#    if os.path.isfile(source):
#        shutil.copy(source, destination)
#        print('copied', file_name)
def quality_factor_PA_coverage(data_ept, coverage, direction = 'sun', angle = 180):
	qf = 0
	for j in range(0, len(data_ept)-1):
		df = coverage[direction]
		df = df.reset_index()
		df = df.drop(np.where(df['EPOCH'] < data_ept[2][0][j])[0])
		df.reset_index(drop = True, inplace = True)
		df = df.drop(np.where(df['EPOCH'] > data_ept[2][1][j])[0])
		df.reset_index(drop = True, inplace = True)
		factors = []
		for i in range(len(df)):
			r = df.center[i]
			if angle == 180:
				if r >=165.:
					factors.append(100)
				if r<165. and r >=155:
					factors.append(90)
				if r<155. and r >=145:
					factors.append(80)
				if r<145. and r >=135:
					factors.append(70)
				if r<135. and r >=125:
					factors.append(60)
				if r<125. and r>=115.:
					factors.append(50)
				if r<115. and r>=105.:
					factors.append(40)
				if r<105. and r>=90.:
					factors.append(30)
				if r<90. and r>=70.:
					factors.append(20)
				if r<70. and r>=45.:
					factors.append(10)
				if r<45.:
					factors.append(1)

			if angle == 0:
				if r <=15.:
					factors.append(100)
				if r<25. and r >=15:
					factors.append(90)
				if r<35. and r >=25:
					factors.append(80)
				if r<45. and r >=35:
					factors.append(70)
				if r<55. and r >=45:
					factors.append(60)
				if r<65. and r>=55.:
					factors.append(50)
				if r<75. and r>=65.:
					factors.append(40)
				if r<90. and r>=75.:
					factors.append(30)
				if r<110. and r>=90.:
					factors.append(20)
				if r<135. and r>=110.:
					factors.append(10)
				if r>135.:
					factors.append(1)
				
		qf = sum(factors)/len(factors)

	return qf

def calculate_shift_factor(step_data, ept_data, sigma, rel_err, frac_nan_threshold, fit_to):
	"""_summary_

	Args:
		step_data (_type_): _description_
		ept_data (_type_): _description_
		sigma (_type_): _description_
		rel_err (_type_): _description_
		frac_nan_threshold (_type_): _description_
		fit_to (_type_): _description_
	"""

	#print(step_data['Primary_energy'])
	fit = fit_to[0].upper()+fit_to[1:]
	bad_step_data = step_data.index[step_data['Primary_energy'] <0.05 ].tolist()
	data_step = step_data.drop(bad_step_data, axis = 0)
	data_step.reset_index(drop=True, inplace=True)
	#print(bad_step_data)

	bad_step_data = data_step.index[data_step['Primary_energy']>0.07].tolist()
	data_step = data_step.drop(bad_step_data, axis = 0)
	data_step.reset_index(drop=True, inplace=True)
	#print(bad_step_data)

	data_step = comb.combine_data([data_step], path = None, sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = False, fit_to = fit)
	#print(data_step)

	n_step_chans = 0

	if len(step_data['Primary_energy'])>8:
		n_step_chans = 4


	else:
		n_step_chans = 1

	bad_ept_data = ept_data.index[ept_data['Primary_energy'] <0.05].tolist()
	data_ept = ept_data.drop(bad_ept_data, axis = 0)
	data_ept.reset_index(drop=True, inplace=True)
	#print(bad_ept_data)

	bad_ept_data = data_ept.index[data_ept['Primary_energy']>0.07].tolist()
	data_ept = data_ept.drop(bad_ept_data, axis = 0)
	data_ept.reset_index(drop=True, inplace=True)
	#print(bad_ept_data)

	data_ept = comb.combine_data([data_ept], path = None, sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = False, fit_to = fit)

	if len(data_step) < n_step_chans or len(data_ept) <4 or data_step['Primary_energy'][len(data_step['Primary_energy'])-1]<data_ept['Primary_energy'][0]:
		print('There are too few energy channels to do a comparison and find a shift factor. If you still want to shift STEP data, please set automatic_shift to False and provide a shift_factor.')
		return(1)
	else:
		step_intensity_average = data_step['Flux_'+fit_to].mean()
		ept_intensity_average = data_ept['Flux_'+fit_to].mean()
		shift_factor = step_intensity_average/ept_intensity_average
		print('STEP INTENSITY AVG '+ str(step_intensity_average))
		print('EPT INTENSITY AVG '+ str(ept_intensity_average))
		print(shift_factor)
		return(shift_factor)

def save_fit_and_run_variables_to_separate_folders(path, date, fit_var_file, run_var_file):
	"""_summary_

	Args:
		path (_type_): _description_
		date (_type_): _description_
		fit_var_file (_type_): _description_
		run_var_file (_type_): _description_
	"""
	fitvariables = path+'fit_variables/'
	runvariables = path+'run_variables/'
	newpath = path+date+'/'
	if not os.path.exists(fitvariables):
		os.makedirs(fitvariables)
	if not os.path.exists(runvariables):
		os.makedirs(runvariables)

	shutil.copy(newpath+fit_var_file, fitvariables+fit_var_file)
	shutil.copy(newpath+run_var_file, runvariables+run_var_file)

	
def FIT_DATA(path, date, averaging, fit_type, step = True, ept = True, het = True, direction='sun', which_fit = 'best', sigma = 3, rel_err = 0.5, frac_nan_threshold = 0.9, fit_to = 'peak', e_min = None, e_max = None, g1_guess = -1.9, g2_guess = -2.5, g3_guess = -4, c1_guess = 1000, alpha_guess = 10, beta_guess = 10, break_guess_low = 0.6, break_guess_high = 1.2, cut_guess = 1.2, exponent_guess = 2, use_random = True, iterations = 20, leave_out_1st_het_chan = True, shift_step_data = False, auto_shift = False,  shift_factor = None, save_fig = True, save_pickle = False, save_fit_variables = True, save_fitrun = True, legend_details = False, bg_subtraction = True, fit_to_separate_folder = False, centre_pix = False, fsize = 12):

	     # slope (float, optional): The type of slope used to find the peak (for the title). Defaults to None.
		 # #slope = None, e_min = None, e_max = None, g1_guess = -1.9, g2_guess = -2.5, g3_guess = -4, c1_guess = 1000, alpha_guess = 10, beta_guess = 10, break_guess_low = 0.6, break_guess_high = 1.2, cut_guess = 1.2, use_random = True, iterations = 20, leave_out_1st_het_chan = True, shift_step_data = False, shift_factor = None, save_fig = True, save_pickle = False, save_fit_variables = True, save_fitrun = True, legend_details = False, ion_correction = True, bg_subtraction = True):

	"""_summary_

	Args:
		path (string): The path to the folder where the data is stored and where the fit and the variable files will be saved.
		date (datetime or string): dt.datetime(yyyy, mm, dd, HH, MM) or 'yyyy-mm-dd-HHMM'
		averaging (int): the averaging of the data
		fit_type (string): fit_type options: 'step', 'ept', 'het', 'step_ept', 'step_ept_het', 'ept_het'
		step (bool, optional): True if you wish to include STEP data in the plot. Not the fit, just the plot. Defaults to True.
		ept (bool, optional): True if you wish to include EPT data in the plot. Not the fit, just the plot. . Defaults to True.
		het (bool, optional): True if you wish to include HET data in the plot. Not the fit, just the plot. . Defaults to True.
		which_fit (str, optional): which_fit options: 'single' will force a single pl fit to the data
		  			'broken' will force a broken pl fit to the data but ONLY if the break point is within the energy range otherwise a sigle pl fit will be produced instead
		  			'best_sb' will choose automatically the best fit type between single and broken by comparing the redchis of the fits
		    		'cut' will produce a single pl fit with an exponential cutoff point. If the cutoff point is outside of the energy range a broken or single pl will be fit instead
			  		'broken_cut' will produce a broken pl fit with an exponential cutoff point. If the cutoff point is outside of the energy range a broken or single pl will be fit instead
				  	'best_cb'. Defaults to 'best'.
					'triple' will force a triple pl fit. If this is not possible, the function will check which is the next best option.
					'best' will choose automatically the best fit type by comparing the redchis of the fits.
		sigma (int, optional): Standard deviation from the background. Defaults to 3.
		rel_err (float, optional): The absolute value of the uncertainty of the bg subtracted flux peak divided by the bg subtracted peak. Defaults to 0.5.
		frac_nan_threshold (float, optional): The pecentage of data withing the search window to condśider the peak reliable (data gap). Defaults to 0.9.
		fit_to (str, optional): You can fit the data either to the peak value within the search window or the avarage flux calculated within the search window. Defaults to 'peak'.
		e_min (float, optional): The lower energy limit for the fit. Defaults to None.
		e_max (float, optional): The upper energy limit for the fit. Defaults to None.
		g1_guess (float, optional): The slope of the single pl fit or the first part of a broken/triple pl fit. Defaults to -1.9.
		g2_guess (float, optional): The slope of the second part of a broken/triple pl fit. gamma2 < gamma1. Defaults to -2.5. 
		g3_guess (int, optional): The slope of the third part of a broken/triple pl fit. gamma3 < gamma2 < gamma1. Defaults to -4.
		c1_guess (int, optional): The intensity/flux value at 0.1 MeV. Defaults to 1000.
		alpha_guess (int, optional): The smoothness of the transition between gamma1 and gamma2. Defaults to 10.
		beta_guess (int, optional): The smoothness of the transition between gamma3 and gamma2. Defaults to 10.
		break_guess_low (float, optional): Guess value for the energy correponding to the break in the broken pl and first break for the triple pl. Defaults to 0.6.
		break_guess_high (float, optional): Guess value for the energy correponding to the second break for the triple pl.. Defaults to 1.2.
		cut_guess (float, optional): Guess value for the energy corresponding to the exponential cutoff.. Defaults to 1.2.
		use_random (bool, optional): If True the fitting function will, in addition to the guess values, choose random values from a predifined list of values for each variable. 
					These values are chosen close to the guess values. Defaults to True.
		iterations (int, optional): The number of times the function will choose random values to use in the fit to the data. Defaults to 20.
		leave_out_1st_het_chan (bool, optional): If True, the first HET channel will be left out from the fit. Defaults to True.
		shift_step_data (bool, optional): If True, STEP data will be shifted (up or down, intensity wise) by a factor equal to shift_factor. Defaults to False.
		shift_factor (float, optional): Factor to shift STEP data (e.g. 0.8). Defaults to None.
		save_fig (bool, optional): If True, the image of the fit will be saved to the specified path. Defaults to True.
		save_pickle (bool, optional): If True, a pickle file from the fit will be saved to the specified path. Defaults to False.
		save_fit_variables (bool, optional): If True, the variables resulting from the final fit will be saved in a csv file to the specified path. Defaults to True.
		save_fitrun (bool, optional): If True, the guess variables and other parameters from running the fit fit will be saved in a csv file to the specified path.. Defaults to True.
		legend_details (bool, optional): If True, the final fit type and the reduced chi square will be dislayed in the legend.
		ion_correction (bool, optional):
		bg_subtraction (bool, optional):
		fit_to_separate_folder (bool, optional): saves the fit to a plots folder
	"""
		
	date_string =''
	folder_time = date

	if type(date) is str:
		date_string = date[:-5]
	else:
		date_string = str(date.date())
		folder_time = str(date)[:-3].replace(' ', '-').replace(':', '')

	# #quick change if submin av
	# av = averaging
	# if averaging < 1.:
	# 	av_string = str(int(averaging*60))+'s'

	# averaging = str(averaging)+'min'
 

	separator = ';'

	step_file_name = ''

	if centre_pix:
		step_file_name = 'proton_data-'+date_string+'-STEP-'+direction+'-L2-'+averaging+'_averaging-centre_pix.csv'
	else:
		step_file_name = 'proton_data-'+date_string+'-STEP-'+direction+'-L2-'+averaging+'_averaging.csv'

	#ept_file_name = ''
	#ept_file_name = 'proton_data-'+date_string+'-EPT-' + direction+ '-L2-'+averaging+'_averaging.csv'
	ept_file_name = 'proton_data-'+date_string+'-EPT-' + direction+ '-L2-'+averaging+'_averaging.csv'

	het_file_name = 'proton_data-'+date_string+'-HET-'+ direction+'-L2-'+averaging+'_averaging.csv'

	# <-------------------------------------------------------------- END OF NECESSARY INPUTS ---------------------------------------------------------------->


	make_fit = True
	#peak_spec = True
	#backsub = True

	fit_to_comb = fit_to[0].upper()+fit_to[1:]

	# These should be changed if using ions
	direction = direction #'sun'

	intensity_label = 'Intensity\n/(s cm² sr MeV)'
	energy_label = 'Energy (MeV)'
	peak_info = fit_to+' spectrum'   #+'\n'+window_type
	legend_title = 'Protons'  # 'mag' or 'foil' or 'Electrons' if there is more than just ept data
	data_product = 'l2'

	#date_str = date_string[8:]+'-'+date_string[5:7]+'-'+date_string[0:4] #DO NOT CHANGE. This is used later for the plot title etc.
	date_str = str(date)[:-3]
	pos = get_horizons_coord('Solar Orbiter', date_string, 'id')
	dist = np.round(pos.radius.value, 2)
	

	# <---------------------------------------------------------------LOADING AND SAVING FILES------------------------------------------------------------------->

	#print(e_min, e_max)
	data_list = []
	step_shift_factor = 1
	#SHIFTING DATA 
	if step:
		step_data = pd.read_csv(path+step_file_name, sep = separator)
		if ept:
			ept_data = pd.read_csv(path+ept_file_name, sep = separator)

			if shift_step_data:
				step_shift_factor = 1
				if auto_shift:
					step_shift_factor = calculate_shift_factor(step_data, ept_data, sigma, rel_err, frac_nan_threshold, fit_to)
				else:
					step_shift_factor = shift_factor
				print('SHIFT FACTOR:  '+str(step_shift_factor))
				step_data['Bg_subtracted_'+fit_to] = step_data['Bg_subtracted_'+fit_to]/step_shift_factor
				step_data['Flux_'+fit_to] = step_data['Flux_'+fit_to]/step_shift_factor
				step_data['Background_flux'] = step_data['Background_flux']/step_shift_factor

		data_list.append(step_data)
		
	if ept :
		ept_data = pd.read_csv(path+ept_file_name, sep = separator)
		data_list.append(ept_data)
	if het :
		het_data = pd.read_csv(path+het_file_name, sep = separator)
		data_list.append(het_data)

	data = comb.combine_data(data_list, path+date_string+'-all-l2-'+direction+'-'+averaging+'.csv', sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = leave_out_1st_het_chan, fit_to = fit_to_comb)
	data = pd.read_csv(path+date_string+'-all-l2-'+direction+'-'+averaging+'.csv', sep = separator)

	if step and ept:
		step_ept_data = comb.combine_data([step_data, ept_data], path+date_string+'-step_ept-l2-'+averaging+'.csv', sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = leave_out_1st_het_chan, fit_to = fit_to_comb)
		step_ept_data = pd.read_csv(path+date_string+'-step_ept-l2-'+averaging+'.csv', sep = separator)

	if ept and het:
		ept_het_data = comb.combine_data([ept_data, het_data], path+date_string+'-ept_het-'+direction+'-l2-'+averaging+'.csv', sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = leave_out_1st_het_chan, fit_to = fit_to_comb)
		ept_het_data = pd.read_csv(path+date_string+'-ept_het-'+direction+'-l2-'+averaging+'.csv', sep = separator)

	# saving the contaminated data so it can be plotted separately
	# then deleting it from the data so it doesn't overlap
	contaminated_data_sigma = comb.low_sigma_threshold(data_list, sigma = sigma, leave_out_1st_het_chan = leave_out_1st_het_chan, fit_to = fit_to_comb)
	contaminated_data_nan   = comb.too_many_nans(data_list, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = leave_out_1st_het_chan)
	contaminated_data_rel_err = comb.high_rel_err(data_list, rel_err = rel_err, leave_out_1st_het_chan = leave_out_1st_het_chan)
	contaminated_data = pd.concat([contaminated_data_sigma, contaminated_data_nan, contaminated_data_rel_err ])
	contaminated_data.reset_index(drop=True, inplace=True)

	#deleting low sigma data so it doesn't overplot 

	if step:
		step_data = comb.delete_bad_data(step_data, sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, fit_to = fit_to_comb)
		#print('STEP   ',step_data)
	if ept:
		ept_data = comb.delete_bad_data(ept_data, sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, fit_to = fit_to_comb)
		#print('EPT  ',ept_data)
	if het:
		first_het_data = comb.first_het_chan(het_data)
		het_data = comb.delete_bad_data(het_data, sigma = sigma, rel_err = rel_err, frac_nan_threshold = frac_nan_threshold, leave_out_1st_het_chan = leave_out_1st_het_chan, fit_to = fit_to_comb)
	
	
	#if e_min is None:
#		e_min = min(data['Primary_energy'])

#	if e_max is None:
#		e_max = max(data['Primary_energy'])
	



	# <---------------------------------------------------------------------DATA--------------------------------------------------------------------->
	#print(data)
	

	# all data
	spec_energy = data['Primary_energy']
	energy_err_low  = data['Energy_error_low']
	energy_err_high = data['Energy_error_high']

	energy_err = [energy_err_low, energy_err_high]

	# step ept
	if step and ept:
		spec_energy_step_ept = step_ept_data['Primary_energy']
		energy_err_low_step_ept  = step_ept_data['Energy_error_low']
		energy_err_high_step_ept = step_ept_data['Energy_error_high']
		
		energy_err_step_ept = [energy_err_low_step_ept, energy_err_high_step_ept]

	if ept and het:
		spec_energy_ept_het = ept_het_data['Primary_energy']
		energy_err_low_ept_het  = ept_het_data['Energy_error_low']
		energy_err_high_ept_het = ept_het_data['Energy_error_high']
		
		energy_err_ept_het = [energy_err_low_ept_het, energy_err_high_ept_het]

	# step data
	if step:
		spec_energy_step = step_data['Primary_energy']
		energy_err_low_step  = step_data['Energy_error_low']
		energy_err_high_step = step_data['Energy_error_high']
		
		energy_err_step = [energy_err_low_step, energy_err_high_step]

	# ept data
	if ept:
		spec_energy_ept = ept_data ['Primary_energy']
		energy_err_low_ept   = ept_data ['Energy_error_low']
		energy_err_high_ept  = ept_data ['Energy_error_high']
		
		energy_err_ept  = [energy_err_low_ept, energy_err_high_ept]

	# het data
	if het:
		spec_energy_het = het_data['Primary_energy']
		energy_err_low_het  = het_data['Energy_error_low']
		energy_err_high_het = het_data['Energy_error_high']
		
		energy_err_het = [energy_err_low_het, energy_err_high_het]
		

	# contaminated data 
	spec_energy_c = contaminated_data['Primary_energy']
	energy_err_low_c  = contaminated_data['Energy_error_low']
	energy_err_high_c = contaminated_data['Energy_error_high']
	
	# visual studio marks the following as not used but they are used if the specific version is uncommented do not delete in case of future use
	# contaminated data sigma
	spec_energy_c_sigma = contaminated_data_sigma['Primary_energy']
	energy_err_low_c_sigma  = contaminated_data_sigma['Energy_error_low']
	energy_err_high_c_sigma = contaminated_data_sigma['Energy_error_high']
	
	# contaminated data nan
	spec_energy_c_nan = contaminated_data_nan['Primary_energy']
	energy_err_low_c_nan  = contaminated_data_nan['Energy_error_low']
	energy_err_high_c_nan = contaminated_data_nan['Energy_error_high']
	
	# contaminated data rel err
	spec_energy_c_rel_err = contaminated_data_rel_err['Primary_energy']
	energy_err_low_c_rel_err  = contaminated_data_rel_err['Energy_error_low']
	energy_err_high_c_rel_err = contaminated_data_rel_err['Energy_error_high']
	
	energy_err_c = [energy_err_low_c, energy_err_high_c]
	energy_err_c_sigma = [energy_err_low_c_sigma, energy_err_high_c_sigma]
	energy_err_c_nan = [energy_err_low_c_nan, energy_err_high_c_nan]
	energy_err_c_rel_err = [energy_err_low_c_rel_err, energy_err_high_c_rel_err]



	if het and leave_out_1st_het_chan:
		# first het
		spec_energy_first_het = first_het_data['Primary_energy']
		energy_err_low_first_het  = first_het_data['Energy_error_low']
		energy_err_high_first_het = first_het_data['Energy_error_high']
		
		energy_err_first_het = [energy_err_low_first_het, energy_err_high_first_het]


# there is no difference between average uncertainty and peak uncertainty so for both peak and avg fit use Backsub_peak_uncertainty

	if bg_subtraction:
		spec_flux   = data['Bg_subtracted_'+fit_to]
		flux_err    = data['Backsub_peak_uncertainty']

		# step ept
		if step and ept:
			spec_flux_step_ept   = step_ept_data['Bg_subtracted_'+fit_to]
			flux_err_step_ept    = step_ept_data['Backsub_peak_uncertainty']

		if ept and het:
			spec_flux_ept_het   = ept_het_data['Bg_subtracted_'+fit_to]
			flux_err_ept_het    = ept_het_data['Backsub_peak_uncertainty']

		# step data
		if step:
			spec_flux_step   = step_data['Bg_subtracted_'+fit_to]
			flux_err_step    = step_data['Backsub_peak_uncertainty']

		# ept data
		if ept:
			spec_flux_ept    = ept_data['Bg_subtracted_'+fit_to]
			flux_err_ept     = ept_data['Backsub_peak_uncertainty']

		# het data
		if het:
			spec_flux_het  = het_data['Bg_subtracted_'+fit_to]
			flux_err_het    = het_data['Backsub_peak_uncertainty']

		# contaminated data 
		spec_flux_c  = contaminated_data['Bg_subtracted_'+fit_to]
		flux_err_c   = contaminated_data['Backsub_peak_uncertainty']

		# visual studio marks the following as not used but they are used if the specific version is uncommented
		# contaminated data sigma
		spec_flux_c_sigma  = contaminated_data_sigma['Bg_subtracted_'+fit_to]
		flux_err_c_sigma   = contaminated_data_sigma['Backsub_peak_uncertainty']

		# contaminated data nan
		spec_flux_c_nan  = contaminated_data_nan['Bg_subtracted_'+fit_to]
		flux_err_c_nan   = contaminated_data_nan['Backsub_peak_uncertainty']

		# contaminated data rel err
		spec_flux_c_rel_err  = contaminated_data_rel_err['Bg_subtracted_'+fit_to]
		flux_err_c_rel_err   = contaminated_data_rel_err['Backsub_peak_uncertainty']


		if het and leave_out_1st_het_chan:
			# first het
			spec_flux_first_het  = first_het_data['Bg_subtracted_'+fit_to]
			flux_err_first_het   = first_het_data['Backsub_peak_uncertainty']

	else:
		spec_flux   = data['Flux_'+fit_to]
		flux_err    = data['Peak_proton_uncertainty']

		# step ept
		if step and ept:
			spec_flux_step_ept   = step_ept_data['Flux_'+fit_to]
			flux_err_step_ept    = step_ept_data['Peak_proton_uncertainty']

		if ept and het:
			spec_flux_ept_het   = ept_het_data['Flux_'+fit_to]
			flux_err_ept_het    = ept_het_data['Peak_proton_uncertainty']

		# step data
		if step:
			spec_flux_step   = step_data['Flux_'+fit_to]
			flux_err_step    = step_data['Peak_proton_uncertainty']

		# ept data
		if ept:
			spec_flux_ept    = ept_data['Flux_'+fit_to]
			flux_err_ept     = ept_data['Peak_proton_uncertainty']

		# het data
		if het:
			spec_flux_het  = het_data['Flux_'+fit_to]
			flux_err_het    = het_data['Peak_proton_uncertainty']

		# contaminated data 
		spec_flux_c  = contaminated_data['Flux_'+fit_to]
		flux_err_c   = contaminated_data['Peak_proton_uncertainty']

		# visual studio marks the following as not used but they are used if the specific version is uncommented
		# contaminated data sigma
		spec_flux_c_sigma  = contaminated_data_sigma['Flux_'+fit_to]
		flux_err_c_sigma   = contaminated_data_sigma['Peak_proton_uncertainty']

		# contaminated data nan
		spec_flux_c_nan  = contaminated_data_nan['Flux_'+fit_to]
		flux_err_c_nan   = contaminated_data_nan['Peak_proton_uncertainty']

		# contaminated data rel err
		spec_flux_c_rel_err  = contaminated_data_rel_err['Flux_'+fit_to]
		flux_err_c_rel_err   = contaminated_data_rel_err['Peak_proton_uncertainty']


		if het and leave_out_1st_het_chan:
			# first het
			spec_flux_first_het  = first_het_data['Flux_'+fit_to]
			flux_err_first_het   = first_het_data['Peak_proton_uncertainty']

	# energy fixed
	min_energy = ''
	max_energy = ''

	if fit_type == 'step':
		if e_min is None:
			min_energy = min(spec_energy_step)

		if e_max is None:
			max_energy = max(spec_energy_step)

	if fit_type == 'ept':
		if e_min is None:
			min_energy = min(spec_energy_ept)

		if e_max is None:
			max_energy = max(spec_energy_ept)

	if fit_type == 'het':
		if e_min is None:
			min_energy = min(spec_energy_het)

		if e_max is None:
			max_energy = max(spec_energy_het)

	if fit_type == 'step_ept':
		if e_min is None:
			min_energy = min(spec_energy_step_ept)

		if e_max is None:
			max_energy = max(spec_energy_step_ept)

	if fit_type == 'ept_het':
		if e_min is None:
			min_energy = min(spec_energy_ept_het)

		if e_max is None:
			max_energy = max(spec_energy_ept_het)

	if fit_type == 'step_ept_het':
		if e_min is None:
			min_energy = min(spec_energy)

		if e_max is None:
			max_energy = max(spec_energy)





#-------------------------------------------------------------------------------------------------------------------------------------------------
	color = {'sun':'crimson','asun':'orange', 'north':'darkslateblue', 'south':'c'}
	
	# quick change for sec resolution, change later
	# if av < 1.:
	# 	averaging = av_string


	pickle_path = None
	if save_pickle:
		pickle_path = path+folder_time+'-pickle_'+fit_type+'-'+fit_to+'-'+which_fit+'-l2-'+averaging+'-'+direction+'.p'

	fit_var_path = None
	if save_fit_variables:
		fit_var_path = path+folder_time+'-fit-result-variables_'+fit_type+'-'+fit_to+'-'+which_fit+'-l2-'+averaging+'-'+direction+'.csv'

	fitrun_path = None
	if save_fitrun:
		fitrun_path = path+folder_time+'-all-fit-variables_'+fit_type+'-'+fit_to+'-'+which_fit+'-l2-'+averaging+'-'+direction+'.csv'
		
		save.save_info_fit(fitrun_path, date_string, averaging, direction, data_product, dist, step, ept, het,
		sigma, rel_err, frac_nan_threshold, leave_out_1st_het_chan, step_shift_factor, fit_type, fit_to,
		which_fit, min_energy, max_energy, g1_guess, g2_guess, c1_guess, alpha_guess, break_guess_low, cut_guess,
		use_random, iterations)
	



	# <----------------------------------------------------------------------FIT AND PLOT------------------------------------------------------------------->

	f, ax = plt.subplots(1, figsize=(8, 6), dpi = 300)
	

	#distance  = ''
	distance = f' (R={dist} au)'
	#ax.errorbar(spec_energy_first_het, spec_flux_first_het, yerr=flux_err_first_het, xerr = energy_err_first_het, marker='o', linestyle='', markersize= 3, color='blue', label='First HET channel', zorder = -1)
	
	if legend_details:
		#if ion_correction:
			#ax.plot([], [], ' ', label="ion corr on")
		#else:
			#ax.plot([], [], ' ', label="ion corr off")

		if bg_subtraction:
			ax.plot([], [], ' ', label="bg subtraction on")
		else:
			ax.plot([], [], ' ', label="bg subtraction off")
		if shift_step_data:
			ax.plot([], [], ' ', label="Shift factor (STEP) "+ str(np.round(step_shift_factor,2)))


	if make_fit:
		if fit_type == 'step':
			plot_title = 'Solar Orbiter '+ distance+' STEP'
			#fit_result = 
			fitting.MAKE_THE_FIT(spec_energy_step, spec_flux_step, energy_err_step[1], flux_err_step, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, g3_guess = g3_guess, alpha_guess=alpha_guess, beta_guess = beta_guess, break_low_guess=break_guess_low, break_high_guess = break_guess_high, cut_guess = cut_guess, c1_guess = c1_guess, exponent_guess = exponent_guess, use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path, detailed_legend = legend_details)
		if fit_type == 'ept':
			plot_title = 'Solar Orbiter '+ distance+' EPT'
			fitting.MAKE_THE_FIT(spec_energy_ept, spec_flux_ept, energy_err_ept[1], flux_err_ept, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, g3_guess = g3_guess, alpha_guess=alpha_guess, beta_guess = beta_guess, break_low_guess=break_guess_low, break_high_guess = break_guess_high, cut_guess = cut_guess, c1_guess = c1_guess, exponent_guess = exponent_guess, use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path, detailed_legend = legend_details)
		if fit_type == 'het':
			plot_title = 'Solar Orbiter '+ distance+' HET'
			fitting.MAKE_THE_FIT(spec_energy_het, spec_flux_het, energy_err_het[1], flux_err_het, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit='single', g1_guess=g1_guess, g2_guess=g2_guess, g3_guess = g3_guess, alpha_guess=alpha_guess, beta_guess = beta_guess, break_low_guess=break_guess_low, break_high_guess = break_guess_high, cut_guess = cut_guess,  c1_guess = c1_guess, exponent_guess = exponent_guess, use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path, detailed_legend = legend_details)
		if fit_type == 'step_ept':
			plot_title = 'Solar Orbiter '+ distance+' STEP and EPT'
			fitting.MAKE_THE_FIT(spec_energy_step_ept, spec_flux_step_ept, energy_err_step_ept[1], flux_err_step_ept, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, g3_guess = g3_guess, alpha_guess=alpha_guess, beta_guess = beta_guess, break_low_guess=break_guess_low, break_high_guess = break_guess_high, cut_guess = cut_guess, c1_guess = c1_guess, exponent_guess = exponent_guess, use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path, detailed_legend = legend_details)
		if fit_type == 'ept_het':
			plot_title = 'Solar Orbiter '+ distance+' EPT and HET'
			fitting.MAKE_THE_FIT(spec_energy_ept_het, spec_flux_ept_het, energy_err_ept_het[1], flux_err_ept_het, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, g3_guess = g3_guess, alpha_guess=alpha_guess, beta_guess = beta_guess, break_low_guess=break_guess_low, break_high_guess = break_guess_high, cut_guess = cut_guess, c1_guess = c1_guess, exponent_guess = exponent_guess, use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path, detailed_legend = legend_details)
		if fit_type == 'step_ept_het':
			plot_title = 'Solar Orbiter '+ distance+' STEP, EPT and HET'
			fitting.MAKE_THE_FIT(spec_energy, spec_flux, energy_err[1], flux_err, ax, direction=direction, e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, g3_guess = g3_guess, alpha_guess=alpha_guess, beta_guess = beta_guess, break_low_guess=break_guess_low, break_high_guess = break_guess_high, cut_guess = cut_guess, c1_guess = c1_guess, exponent_guess = exponent_guess, use_random = use_random, iterations = iterations, path = pickle_path, path2 = fit_var_path, detailed_legend = legend_details)
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
				plot_title = 'Solar Orbiter STEP, EPT and HET'
			if ept and het is False:
				plot_title = 'Solar Orbiter STEP and EPT'
			if het and ept is False:
				plot_title = 'Solar Orbiter STEP and HET'
			if ept is False and het is False:
				plot_title = 'Solar Orbiter STEP'
		if ept:
			ax.errorbar(spec_energy_ept, spec_flux_ept, yerr=flux_err_ept, xerr = energy_err_ept, marker='o', linestyle='', markersize= 3, color=color[direction], label='EPT '+direction, zorder = -1)
			if het and step is False:
				plot_title = 'Solar Orbiter EPT and HET'
			if step is False and het is False:
				plot_title = 'Solar Orbiter EPT'
		if het:
			ax.errorbar(spec_energy_het, spec_flux_het, yerr=flux_err_het, xerr = energy_err_het, marker='o', linestyle='', markersize= 3, color='maroon', label='HET '+direction, zorder = -1)
			if ept is False and step is False:
				plot_title = 'Solar Orbiter HET'
			if leave_out_1st_het_chan:
				ax.errorbar(spec_energy_first_het, spec_flux_first_het, yerr=flux_err_first_het, xerr = energy_err_first_het, marker='o', linestyle='', markersize= 3, color='black', label='First HET channel', zorder = -1)
				

		
	# for a more deailed version uncomment
	#ax.errorbar(spec_energy_c_sigma, spec_flux_c_sigma, yerr=flux_err_c_sigma, xerr = energy_err_c_sigma, marker='o', linestyle='', markersize= 3, color='blue', label='Sigma below '+str(sigma), zorder = -1)
	#ax.errorbar(spec_energy_c_nan, spec_flux_c_nan, yerr=flux_err_c_nan, xerr = energy_err_c_nan, marker='o', linestyle='', markersize= 3, color='gray', label='excluded (NaNs)', zorder = -1)
	#ax.errorbar(spec_energy_c_rel_err, spec_flux_c_rel_err, yerr=flux_err_c_rel_err, xerr = energy_err_c_rel_err, marker='o', linestyle='', markersize= 3, color='purple', label='excluded (rel err)', zorder = -1)


	
	if step:
		ax.errorbar(spec_energy_step, step_data['Background_flux'], yerr=step_data['Bg_proton_uncertainty'], xerr = energy_err_step, marker='o', markersize= 3, linestyle='', color='darkorange', alpha=0.3)
	if ept:
		ax.errorbar(spec_energy_ept, ept_data['Background_flux'], yerr=ept_data['Bg_proton_uncertainty'], xerr = energy_err_ept, marker='o', markersize= 3, linestyle='', color=color[direction], alpha=0.3)
	if het:
		ax.errorbar(spec_energy_het, het_data['Background_flux'], yerr=het_data['Bg_proton_uncertainty'], xerr = energy_err_het, marker='o', markersize= 3, linestyle='', color='maroon', alpha=0.3)
	#ax.errorbar(spec_energy_c_sigma, contaminated_data_sigma['Background_flux'], yerr=contaminated_data_sigma['Bg_electron_uncertainty'], xerr = energy_err_c_sigma, marker='o', markersize= 3, linestyle='', color='blue', alpha=0.3)
	#ax.errorbar(spec_energy_c_nan, contaminated_data_nan['Background_flux'], yerr=contaminated_data_nan['Bg_electron_uncertainty'], xerr = energy_err_c_nan, marker='o', markersize= 3, linestyle='', color='gray', alpha=0.3)
	#ax.errorbar(spec_energy_c_rel_err, contaminated_data_rel_err['Background_flux'], yerr=contaminated_data_rel_err['Bg_electron_uncertainty'], xerr = energy_err_c_rel_err, marker='o', markersize= 3, linestyle='', color='purple', alpha=0.3)
	if make_fit:
		ax.errorbar(spec_energy_c, contaminated_data['Background_flux'], yerr=contaminated_data['Bg_proton_uncertainty'], xerr = energy_err_c, marker='o', markersize= 3, linestyle='', color='gray', alpha=0.3)
		if het and leave_out_1st_het_chan:
			ax.errorbar(spec_energy_first_het, first_het_data['Background_flux'], yerr=first_het_data['Bg_proton_uncertainty'], xerr = energy_err_first_het, marker='o', markersize= 3, linestyle='', color='black', alpha=0.3)
	
	if make_fit is False:
		ax.errorbar(spec_energy_c, contaminated_data['Background_flux'], yerr=contaminated_data['Bg_proton_uncertainty'], xerr = energy_err_c, marker='o', markersize= 3, linestyle='', color='maroon', alpha=0.3)
	
	e_range_min = None
	e_range_max = None	
	step_energy_range = [0.004323343613,0.07803193193]  # needs to be changed to protons!
	ept_energy_range =  [2.,5.]
	het_energy_range =  [10.,100.]
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


	plt.xticks(fontsize = fsize)
	plt.yticks(fontsize = fsize)
	plt.legend(title=legend_title,  prop={'size': 7}, fontsize = fsize-2, title_fontsize = fsize)
	plt.ylabel(intensity_label, fontsize = fsize)
	plt.xlabel(energy_label, fontsize = fsize)
	if centre_pix:
		plt.title(plot_title+'  '+peak_info+'\n'+date_str+'  '+averaging+'  averaging, centre pixels', fontsize = fsize+2)
	else:
		plt.title(plot_title+'  '+peak_info+'\n'+date_str+'  '+averaging+'  averaging', fontsize = fsize+2)
		

	plot_path = path
	if fit_to_separate_folder:
		plot_path = path+'plots/'
		if not os.path.exists(plot_path):
			os.makedirs(plot_path)

	pix = ''
	if centre_pix:
		pix = '-centre_pix'

	shift_string = ''
	if shift_step_data:
		if auto_shift:
			shift_string = '-auto-step-shift_'+str(np.round(step_shift_factor,2)).replace('.', '_')
		else:
			shift_string = '-step-shift-hift_'+str(np.round(step_shift_factor,2)).replace('.', '_')

	#shift_step_data = False, auto_shift = False, shift_factor = None

	if save_fig:
		if make_fit:
			#if not bg_subtraction:
			#	plt.savefig(plot_path+'protons-'+date_string+'-'+direction+'-'+averaging+'-'+which_fit+'-'+fit_type+'-'+fit_to+'-'+pix, dpi=300)
			if  bg_subtraction:
				plt.savefig(plot_path+'protons-'+date_string+'-'+direction+'-'+averaging+'-'+which_fit+'-'+fit_type+'-'+fit_to+'-bg_sub'+pix, dpi=300)
			#if ion_correction and bg_subtraction:
				#plt.savefig(plot_path+'protons-'+date_string+'-'+direction+'-'+averaging+'-'+which_fit+'-'+fit_type+'-'+fit_to+'-ion_corr-bg_sub'+pix, dpi=300)
			else:
				plt.savefig(plot_path+'protons-'+date_string+'-'+direction+'-'+averaging+'-'+which_fit+'-'+fit_type+'-'+fit_to+pix, dpi=300)
			
		if make_fit is False:
			if step and ept and het:
				plt.savefig(plot_path+'protons-'+date_string+'-'+averaging+'-no_fit-step_ept_het'+pix, dpi=300)
			if step and ept and het is False:
				plt.savefig(plot_path+'protons-'+date_string+'-'+averaging+'-no_fit-step_ept'+pix, dpi=300)
			if step and het and ept is False:
				plt.savefig(plot_path+'protons-'+date_string+'-'+averaging+'-no_fit-step_het'+pix, dpi=300)
			if step and ept is False and het is False:
				plt.savefig(plot_path+'protons-'+date_string+'-'+averaging+'-no_fit-step'+pix, dpi=300)
			if ept and het and step is False:
				plt.savefig(plot_path+'protons-'+date_string+'-'+averaging+'-no_fit-ept_het'+pix, dpi=300)
			if ept and step is False and het is False:
				plt.savefig(plot_path+'protons-'+date_string+'-'+averaging+'-no_fit-ept'+pix, dpi=300)
			if het and ept is False and step is False:
				plt.savefig(plot_path+'protons-'+date_string+'-'+averaging+'-no_fit-het'+pix, dpi=300)
			
			


	plt.show()
