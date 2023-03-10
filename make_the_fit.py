# solo_functions.py
import numpy as np
import pandas as pd
import my_power_law_fits_odr_20210819 as pl_fit
from scipy.stats import t as studentt
#from lmfit.models import GaussianModel
import pickle
#from scipy.odr import *


def make_step_electron_flux(stepdata, mask_conta=False):
	'''
	here we use the calibration factors from Paco (Alcala) to calculate the electron flux out of the (integral - magnet) fluxes
	(we now use level2 data files to get these)
	we also check if the integral counts are sufficiently higher than the magnet counts so that we can really assume it's electrons
	(ohterwise we mask the output arrays)
	As suggested by Alex Kollhoff & Berger use a 5 sigma threashold:
	C_INT >> C_MAG:
	C_INT - C_MAG > 5*sqrt(C_INT)
	Alex: die count rates und fuer die uebrigen Zeiten gebe ich ein oberes Limit des Elektronenflusses an,
	das sich nach 5*sqrt(C_INT) /(E_f - E_i) /G_e berechnet.
	'''
	# calculate electron flux from F_INT - F_MAG:
	colnames = ["ch_num", "E_low", "E_hi", "factors"]
	paco = pd.read_csv(r'C:\Users\Omistaja\Desktop\SRL\2021SRL\spectra_fitting\step_electrons_calibration.csv', names=colnames, skiprows=1)
	F_INT = stepdata['Integral_Flux']
	F_MAG = stepdata['Magnet_Flux']
	step_flux =  (F_INT - F_MAG) * paco.factors.values
	U_INT = stepdata['Integral_Uncertainty']
	U_MAG = stepdata['Magnet_Uncertainty']
	# from Paco:
    # Ele_Uncertainty = k * sqrt(Integral_Uncertainty^2 + Magnet_Uncertainty^2)
	step_unc =  np.sqrt(U_INT**2 + U_MAG**2) * paco.factors.values
	
	param_list = ['Electron_Flux', 'Electron_Uncertainty']
	
	if mask_conta:
		C_INT = stepdata['Integral_Rate']
		C_MAG = stepdata['Magnet_Rate']
		clean = (C_INT - C_MAG) > 5*np.sqrt(C_INT)
		step_flux = step_flux.mask(clean)
		step_unc = step_unc.mask(clean)
		step_data = pd.concat([step_flux, step_unc], axis=1, keys=param_list)
		
	return step_data, paco.E_low, paco.E_hi



def average_flux_error(flux_err: pd.DataFrame) -> pd.Series:

    return np.sqrt((flux_err ** 2).sum(axis=0)) / len(flux_err.values)


def closest_values(array, value):

	#this function finds n closest values to the guess value
	#and returns an array with the closest values
	#the number of values in the closest_values_array depends on the length of the initial array

	# the value is going to be g1_guess, g2_guess, alpha_guess, break_guess, c1_guess, cut_guess
	
	
	if len(array)<=10:
		array_size = 5
		
	if len(array)>10 and len(array)<= 20:
		array_size = 7
	
	if len(array) >20:
		array_size = 10
		
		
	array = np.delete(array, np.where(array ==value))
		
	closest_values_array = np.array(())
	
	for i in range(array_size):
		absolute_val_array = np.abs(array - value)
		#print(absolute_val_array)
		smallest_difference_index = absolute_val_array.argmin()
		closest_element = array[smallest_difference_index]
		closest_values_array = np.append(closest_values_array, closest_element)
		array = np.delete(array, np.where(array ==closest_element))
	
	closest_values_array = sorted(closest_values_array)
	
	return(closest_values_array)
	
	
def check_redchi(spec_e, spec_flux, e_err, flux_err, gamma1, gamma2, c1, alpha, E_break,  E_cut= None, fit = 'best',  maxit=10000, e_min=None, e_max=None):
	#the function also checks if the break point is outside of the energy array (also the cutoff point)
	#the min and max energies cannot be last and/or first points because it wouldn't be a physical result
	if e_min is None:
		#e_min = min(spec_e)
		e_min = spec_e[2]
	if e_max is None:
		#e_max = max(spec_e)
		e_max = spec_e[len(spec_e)-3]
		
	result_single_pl = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, c1 = c1)
	redchi_single  = result_single_pl.res_var  
										
	result_broken = pl_fit.broken_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, gamma2 = gamma2, c1 = c1, alpha = alpha, E_break = E_break, maxit=10000)
	redchi_broken = result_broken.res_var
	breakp        = result_broken.beta[4]	

	#if E_cut != None:
	#	result_cut = pl_fit.cut_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, c1 = c1, E_cut = E_cut, maxit=10000)
	#	redchi_cut= result_broken.res_var
	#	cut        = result_cut.beta[2]	#shoud maybe make distinction between cut from cut pl and cut from cut broken pl

	
	#which_fit = 'single'
	#redchi = 0
	#result = ''
	#print(type(result))
	# cut break check	
	if fit == 'single':
		which_fit = 'single'
		redchi = redchi_single
		result = result_single_pl
		return([which_fit, redchi, result])

	if fit == 'broken':
		if breakp < e_min or breakp > e_max:
			which_fit = 'single'
			redchi = redchi_single
			result = result_single_pl
			return([which_fit, redchi, result])
		if breakp >= e_min and breakp <=e_max:	
			which_fit = 'broken'
			redchi = redchi_broken
			result = result_broken
			return([which_fit, redchi, result])
	

	if fit == 'best' or fit == 'broken_cut':
		result_cut_break = pl_fit.cut_break_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=gamma1, gamma2=gamma2, c1=c1, alpha=alpha, E_break=E_break, E_cut = E_cut, print_report=False, maxit=10000)
		redchi_cut_break = result_cut_break.res_var
		breakp_cut = result_cut_break.beta[4]
		#The cut of te break + cutoff
		cut_b = result_cut_break.beta[5]

		result_cut = pl_fit.cut_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, c1 = c1, E_cut = E_cut, maxit=10000)
		redchi_cut= result_cut.res_var
		cut        = result_cut.beta[2]	#shoud maybe make distinction between cut from cut pl and cut from cut broken pl
		
		difference = cut_b-breakp_cut
		
		if difference>0.01 and redchi_cut_break<redchi_broken and redchi_cut_break <redchi_single and redchi_cut_break< redchi_cut:
			if cut_b < e_min or cut_b > e_max:
				if breakp_cut < e_min or breakp_cut > e_max:
					which_fit = 'single'
					redchi = redchi_single
					result = result_single_pl
					return([which_fit, redchi, result])
				# Need to compare break and cut to see which actually fits better
				if breakp >= e_min and breakp <=e_max:
					if redchi_broken<=redchi_cut:
						which_fit = 'broken'
						redchi = redchi_broken
						result = result_broken
						return([which_fit, redchi, result])
					if redchi_cut < redchi_broken:
						which_fit = 'cut'
						redchi = redchi_cut
						result = result_cut
						return([which_fit, redchi, result])
				else:
					which_fit = 'single'
					redchi = redchi_single
					result = result_single_pl
					return([which_fit, redchi, result])


			if cut_b >= e_min and cut_b<= e_max:
				if cut_b> breakp_cut:
					if breakp_cut >= e_min and breakp_cut <= e_max:
						which_fit = 'broken_cut'
						redchi = redchi_cut_break
						result = result_cut_break
						return([which_fit, redchi, result])

					else:
						if redchi_single<redchi_broken and redchi_single<redchi_cut:
							which_fit = 'single'
							redchi = redchi_single
							result = result_single_pl
							return([which_fit, redchi, result])
						if redchi_broken<=redchi_single and redchi_broken<=redchi_cut:
							which_fit = 'broken'
							redchi = redchi_broken
							result = result_broken
							return([which_fit, redchi, result])
						if redchi_cut<redchi_broken and redchi_cut<redchi_single:
							which_fit = 'cut'
							redchi = redchi_cut							
							result = result_cut
							return([which_fit, redchi, result])

				if cut_b<= breakp_cut:	
					if redchi_single<redchi_broken and redchi_single<redchi_cut:
						which_fit = 'single'
						redchi = redchi_single
						result = result_single_pl
						return([which_fit, redchi, result])
					if redchi_broken<=redchi_single and redchi_broken<=redchi_cut:
						which_fit = 'broken'
						redchi = redchi_broken
						result = result_broken
						return([which_fit, redchi, result])
					if redchi_cut<redchi_broken and redchi_cut<redchi_single:
						which_fit = 'cut'
						redchi = redchi_cut							
						result = result_cut
						return([which_fit, redchi, result])
							
				
		if difference<=0.01:
			if redchi_broken<=redchi_single and redchi_broken<= redchi_cut:
				if breakp <= e_min or breakp >= e_max:
					which_fit = 'single'
					redchi = redchi_single
					result = result_single_pl
					return([which_fit, redchi, result])
				if breakp > e_min and breakp <e_max:
					which_fit = 'broken'
					redchi = redchi_broken
					result = result_broken
					return([which_fit, redchi, result])

			if redchi_cut<redchi_single and  redchi_cut < redchi_broken:
				if cut <= e_min or cut >= e_max:
					which_fit = 'single'
					redchi = redchi_single
					result = result_single_pl
					return([which_fit, redchi, result])
				if cut > e_min and cut <e_max:
					which_fit = 'cut'
					redchi = redchi_cut
					result = result_cut
					return([which_fit, redchi, result])

			if redchi_broken>redchi_single and redchi_cut>redchi_single:
				which_fit = 'single'
				redchi = redchi_single
				result = result_single_pl
				return([which_fit, redchi, result])


		if redchi_broken<=redchi_single and redchi_broken <=redchi_cut_break and redchi_broken<= redchi_cut:
			if breakp <= e_min or breakp >= e_max:
				which_fit = 'single'
				redchi = redchi_single
				result = result_single_pl
				return([which_fit, redchi, result])
			if breakp > e_min and breakp <e_max:
				which_fit = 'broken'
				redchi = redchi_broken
				result = result_broken
				return([which_fit, redchi, result])

		if redchi_cut<redchi_single and redchi_cut <redchi_cut_break and redchi_cut < redchi_broken:
			if cut <= e_min or cut >= e_max:
				which_fit = 'single'
				redchi = redchi_single
				result = result_single_pl
				return([which_fit, redchi, result])
			if cut > e_min and cut <e_max:
				which_fit = 'cut'
				redchi = redchi_cut
				result = result_cut
				return([which_fit, redchi, result])

		if redchi_broken>redchi_single and redchi_cut_break>redchi_single and redchi_cut>redchi_single:
			which_fit = 'single'
			redchi = redchi_single
			result = result_single_pl
			return([which_fit, redchi, result])


	
	if fit == 'best_sb':
		if redchi_broken<=redchi_single:
			if breakp < e_min or breakp > e_max:
				which_fit = 'single'
				redchi = redchi_single
				result = result_single_pl
				return([which_fit, redchi, result])
			if breakp >= e_min and breakp <=e_max:	
				which_fit = 'broken'
				redchi = redchi_broken
				result = result_broken
				return([which_fit, redchi, result])
		if redchi_broken>redchi_single:
			which_fit = 'single'
			redchi = redchi_single
			result = result_single_pl
			return([which_fit, redchi, result])

	if fit == 'best_cb':
		result_cut = pl_fit.cut_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, c1 = c1, E_cut = E_cut, maxit=10000)
		redchi_cut= result_broken.res_var
		cut        = result_cut.beta[2]	#shoud maybe make distinction between cut from cut pl and cut from cut broken pl

		if redchi_broken<=redchi_cut:
			if breakp < e_min or breakp > e_max:
				which_fit = 'single'
				redchi = redchi_single
				result = result_single_pl
				return([which_fit, redchi, result])
			if breakp >= e_min and breakp <=e_max:	
				which_fit = 'broken'
				redchi = redchi_broken
				result = result_broken
				return([which_fit, redchi, result])
		if redchi_broken>redchi_cut:
			if cut < e_min or cut > e_max:
				which_fit = 'single'
				redchi = redchi_single
				result = result_single_pl
				return([which_fit, redchi, result])
			if cut >= e_min and cut <=e_max:	
				which_fit = 'cut'
				redchi = redchi_cut
				result = result_cut
				return([which_fit, redchi, result])


	

	
			
	# print(result)
	
	#return([which_fit, redchi, result])
	
	
def find_c1(spec_e, spec_flux, e_min, e_max):
	absolute_val_array = np.abs(spec_e - e_max)
	smallest_difference_index = absolute_val_array.argmin()
	closest_element = spec_e[smallest_difference_index]
	
	x1 = np.log10(spec_e[smallest_difference_index-5])
	y1 = np.log10(spec_flux[smallest_difference_index-5])
	
	x2 = np.log(spec_e[smallest_difference_index])
	y2 = np.log10(spec_flux[smallest_difference_index])
	
	m = (y1-y2)/(x1-x2)
	q = (x1*y2-x2*y1)/(x1-x2)
	
	c1 = m*1.0+q
	
	print('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2)
	return(c1)
	
		
	
	

def MAKE_THE_FIT(spec_e, spec_flux, e_err, flux_err, ax, direction='sun', which_fit='best', e_min=None, e_max=None, g1_guess=-2., g2_guess=None, alpha_guess=5., break_guess=0.065, cut_guess = 0.12, c1_guess=None, use_random = False, iterations = 10, path = None, path2 = None):
	'''This function fit the data to a single, double or break+cut power law. 
	The fit type can be chosen between: single,double, cut or best. 
	The best option checks between all the options and chooses between the three by checking the reduced chisqr.
	Also when the broken or cut options are chosen, the function checks if the break or cutoff points are outside of the energy range.
	In such case, a sigle pl will be fit to the data and the function will output that the breakpoint is outside of the energy range.''' 
	
	
	if g2_guess is None:
		g2_guess = g1_guess - 0.1
		
	if e_min is None:
		#e_min = min(spec_e)
		e_min = spec_e[0]
	if e_max is None:
		#e_max = max(spec_e)
		e_max = spec_e[len(spec_e)-1]
	
	if c1_guess is None:
		absolute_val_array = np.abs(spec_e - 1)
		smallest_difference_index = absolute_val_array.argmin()
		c1_guess = spec_flux[smallest_difference_index]
		
		
	# the break guess should be between min and max energy
	
	# have to construct the guesses logarithmically
	
	if use_random :	
		gamma1_array = closest_values(np.arange(-5.0,0,0.5), g1_guess)
		gamma2_array = closest_values(np.arange(-5.0,0,0.5), g2_guess)
		
	# c1_array...  we want to get a good approximation of the flux at 1, whatever 1 is in your plot. 
		c1_array = np.arange(c1_guess/10,c1_guess*10, c1_guess/10)
		
	# alpha array
		a1_array = np.arange(0.01,0.1,0.01)
		a2_array = np.arange(0.1,1.0,0.05)
		a3_array = np.arange(1,10,0.5)
		a4_array = np.arange(10,100,10)
		a5_array = np.arange(100,220,20)
		alpha_array = np.hstack((a1_array,a2_array,a3_array,a4_array,a5_array))
		alpha_array = closest_values(alpha_array, alpha_guess)
	# break array
	# cut array = break_array *1.8
		
		if e_max<0.1:
			break_array = np.arange(e_min, e_max, 0.01)
			cut_array = break_array*1.8
		if e_max>=0.1 and e_max<1.0:
			b1_array = np.arange(e_min, 0.1, 0.01)
			b2_array = np.arange(0.1, e_max, 0.05)
			break_array = np.hstack((b1_array, b2_array))
			cut_array = break_array*1.8
		if e_max >=1 and e_max < 10:
			b1_array = np.arange(e_min, 0.1, 0.01)
			b2_array = np.arange(0.1, 1, 0.05)
			b3_array = np.arange(1, e_max, 0.5)
			break_array = np.hstack((b1_array, b2_array, b3_array))
			cut_array = break_array*1.8
		if e_max>=10:
			b1_array = np.arange(e_min, 0.1, 0.01)
			b2_array = np.arange(0.1, 1, 0.05)
			b3_array = np.arange(1, 10, 0.5)
			b4_array = np.arange(10, e_max, 1)
			break_array = np.hstack((b1_array, b2_array, b3_array, b4_array))
			cut_array = break_array*1.8
	
		break_array = closest_values(break_array, break_guess)
		cut_array = closest_values(cut_array, cut_guess)

	#print(c1_guess)
	
		
		
	
	color = {'sun':'crimson', 'asun':'orange','north':'darkslateblue','south':'c'}
	spec_e = np.array(spec_e)
	spec_flux = np.array(spec_flux)
	e_err = np.array(e_err)
	flux_err = np.array(flux_err)
	
	xplot = np.logspace(np.log10(np.nanmin(spec_e)), np.log10(np.nanmax(spec_e)), num=500)
	xplot = xplot[np.where((xplot >= e_min) & (xplot <= e_max))[0]]
	
	fit_ind   = np.where((spec_e >= e_min) & (spec_e <= e_max) & (np.isfinite(spec_flux) == True) & (np.isfinite(flux_err) == True))[0]
	spec_e    = spec_e[fit_ind]
	spec_flux = spec_flux[fit_ind]
	e_err     = e_err[fit_ind]
	flux_err  = flux_err[fit_ind]

	# everything is in a for loop that chooses random values between the 
	# closest_values (n times) and checks the redchis and chooses the best one
	# try separately the input guesses and then the random ones
	
	# everything is done first with input guess values and then with randoms
	
	# parameters used as final inputs !!!(not the fit result but input)!!!
	
	which_fit_final = ''
	
	gamma1_final = 0
	gamma2_final = 0
	alpha_final = 0
	break_final = 0
	cut_final = 0
	c1_final = 0
	redchi_final = 0
	
	result_final = None
	

	if which_fit == 'best':
	#first check the redchi and if the break is outside of the energy range using the guess values then compare the random values to these 
	#if redchi is better, substitute values
		which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break=break_guess, E_cut = cut_guess, fit = 'best', maxit=10000, e_min = e_min, e_max = e_max)
		redchi_guess = which_fit_guess[1]
		print(redchi_guess)
		redchi_final = redchi_guess
		which_fit_final = which_fit_guess[0]
		result_final = which_fit_guess[2]
		if which_fit_guess[0] == 'single': 
			gamma1_final = g1_guess
			gamma2_final = np.nan
			alpha_final = np.nan
			break_final = np.nan
			cut_final = np.nan
			c1_final = c1_guess
		if which_fit_guess[0] == 'broken': 
			gamma1_final = g1_guess
			gamma2_final = g2_guess
			alpha_final = alpha_guess
			break_final = break_guess
			cut_final = np.nan
			c1_final = c1_guess
		if which_fit_guess[0] == 'cut':
			gamma1_final = g1_guess
			gamma2_final = np.nan
			alpha_final = np.nan
			break_final = np.nan
			cut_final = cut_guess
			c1_final = c1_guess
		if which_fit_guess[0] == 'broken_cut':
			gamma1_final = g1_guess
			gamma2_final = g2_guess
			alpha_final = alpha_guess
			break_final = break_guess
			cut_final = cut_guess
			c1_final = c1_guess
		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0] 
				#gamma2 should always be more negative (smaller) than gamma1
				if g1_random<g2_random:
					gamma = g1_random
					g1_random = g2_random
					g2_random = gamma
				alpha_random = np.random.choice(alpha_array, 1)[0]
				break_random = np.random.choice(break_array,1)[0]
				cut_random = np.random.choice(cut_array,1)[0]
				c1_random = np.random.choice(c1_array,1)[0]
				which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, gamma1=g1_random, gamma2=g2_random, E_break=break_random, E_cut = cut_random, maxit=10000, e_min = e_min, e_max = e_max)
				redchi_random = which_fit_random[1]
				if redchi_random < redchi_final:
					result_final = which_fit_random[2]
					if which_fit_random[0] == 'single':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = np.nan
						alpha_final = np.nan
						break_final = np.nan
						cut_final = np.nan
						c1_final = c1_random
					if which_fit_random[0] == 'broken':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = g2_random
						alpha_final = alpha_random
						break_final = break_random
						cut_final = np.nan
						c1_final = c1_random
					if which_fit_random[0] == 'cut':
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = np.nan
						alpha_final = np.nan
						break_final = np.nan
						cut_final = cut_random
						c1_final = c1_random	
					if which_fit_random[0] == 'broken_cut':
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = g2_random
						alpha_final = alpha_random
						break_final = break_random
						cut_final = cut_random
						c1_final = c1_random
						





	if which_fit == 'best_cb':
	#first check the redchi and if the break is outside of the energy range using the guess values then compare the random values to these 
	#if redchi is better, substitute values
		which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break=break_guess, E_cut = cut_guess, fit = 'best_cb', maxit=10000, e_min = e_min, e_max = e_max)
		redchi_guess = which_fit_guess[1]
		redchi_final = redchi_guess
		which_fit_final = which_fit_guess[0]
		result_final = which_fit_guess[2]
		if which_fit_guess[0] == 'single': 
			gamma1_final = g1_guess
			gamma2_final = np.nan
			alpha_final = np.nan
			break_final = np.nan
			cut_final = np.nan
			c1_final = c1_guess
		if which_fit_guess[0] == 'broken': 
			gamma1_final = g1_guess
			gamma2_final = g2_guess
			alpha_final = alpha_guess
			break_final = break_guess
			cut_final = np.nan
			c1_final = c1_guess
		if which_fit_guess[0] == 'cut':
			gamma1_final = g1_guess
			gamma2_final = np.nan
			alpha_final = np.nan
			break_final = np.nan
			cut_final = cut_guess
			c1_final = c1_guess
		
		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0] 
				#gamma2 should always be more negative (smaller) than gamma1
				if g1_random<g2_random:
					gamma = g1_random
					g1_random = g2_random
					g2_random = gamma
				alpha_random = np.random.choice(alpha_array, 1)[0]
				break_random = np.random.choice(break_array,1)[0]
				cut_random = np.random.choice(cut_array,1)[0]
				c1_random = np.random.choice(c1_array,1)[0]
				which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, gamma1=g1_random, gamma2=g2_random, E_break=break_random, E_cut = cut_random, fit = 'best_cb', maxit=10000, e_min = e_min, e_max = e_max)
				redchi_random = which_fit_random[1]
				if redchi_random < redchi_final:
					result_final = which_fit_random[2]
					if which_fit_random[0] == 'single':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = np.nan
						alpha_final = np.nan
						break_final = np.nan
						cut_final = np.nan
						c1_final = c1_random
					if which_fit_random[0] == 'broken':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = g2_random
						alpha_final = alpha_random
						break_final = break_random
						cut_final = np.nan
						c1_final = c1_random
					if which_fit_random[0] == 'cut':
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = np.nan
						alpha_final = np.nan
						break_final = np.nan
						cut_final = cut_random
						c1_final = c1_random	
					

	if which_fit == 'best_sb':
	#first check the redchi and if the break is outside of the energy range using the guess values then compare the random values to these 
	#if redchi is better, substitute values
		which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break=break_guess, E_cut = None, fit = 'best_sb', maxit=10000, e_min = e_min, e_max = e_max)
		redchi_guess = which_fit_guess[1]
		redchi_final = redchi_guess
		which_fit_final = which_fit_guess[0]
		result_final = which_fit_guess[2]
		if which_fit_guess[0] == 'single': 
			gamma1_final = g1_guess
			gamma2_final = np.nan
			alpha_final = np.nan
			break_final = np.nan
			cut_final = np.nan
			c1_final = c1_guess
		if which_fit_guess[0] == 'broken': 
			gamma1_final = g1_guess
			gamma2_final = g2_guess
			alpha_final = alpha_guess
			break_final = break_guess
			cut_final = np.nan
			c1_final = c1_guess
		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0]
				#gamma2 should always be more negative (smaller) than gamma1
				if g1_random<g2_random:
					gamma = g1_random
					g1_random = g2_random
					g2_random = gamma
				alpha_random = np.random.choice(alpha_array, 1)[0]
				break_random = np.random.choice(break_array,1)[0]
				c1_random = np.random.choice(c1_array,1)[0]
				which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, gamma1=g1_random, gamma2=g2_random, E_break=break_random, E_cut = None, fit = 'best_sb', maxit=10000, e_min = e_min, e_max = e_max)
				redchi_random = which_fit_random[1]
				if redchi_random < redchi_final:
					result_final = which_fit_random[2]
					if which_fit_random[0] == 'single':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = np.nan
						alpha_final = np.nan
						break_final = np.nan
						cut_final = np.nan
						c1_final = c1_random
					if which_fit_random[0] == 'broken':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = g2_random
						alpha_final = alpha_random
						break_final = break_random
						cut_final = np.nan
						c1_final = c1_random

	
	if which_fit == 'broken_cut':
		result_cut_guess = pl_fit.cut_break_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess, gamma2=g2_guess, c1=c1_guess, alpha=alpha_guess, E_break=break_guess, E_cut = cut_guess, print_report=False, maxit=10000)
		breakp_cut = result_cut_guess.beta[4]
		cut_b = result_cut_guess.beta[5]
	
		if breakp_cut < e_min or breakp_cut > e_max:
			print('The break point is outside of the energy range')
			which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break=break_guess, E_cut = None, fit = 'best_cb', maxit=10000, e_min = e_min, e_max = e_max)
			redchi_guess = which_fit_guess[1]
			redchi_final = redchi_guess
			which_fit_final = which_fit_guess[0]
			result_final = which_fit_guess[2]
			if which_fit_guess[0] == 'single': 
				gamma1_final = g1_guess
				gamma2_final = np.nan
				alpha_final = np.nan
				break_final = np.nan
				cut_final = np.nan
				c1_final = c1_guess
			if which_fit_guess[0] == 'broken': 
				gamma1_final = g1_guess
				gamma2_final = g2_guess
				alpha_final = alpha_guess
				break_final = break_guess
				cut_final = np.nan
				c1_final = c1_guess
			if which_fit_guess[0] == 'cut': 
				gamma1_final = g1_guess
				gamma2_final = g2_guess
				alpha_final = alpha_guess
				break_final = break_guess
				cut_final = cut_guess
				c1_final = c1_guess



		if breakp_cut >= e_min and breakp_cut <=e_max:
			if cut_b <=e_min or cut_b >=e_max:
				# The breaks are checked by redchi
				which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break=break_guess, E_cut = None, fit = 'best_cb', maxit=10000, e_min = e_min, e_max = e_max)
				redchi_guess = which_fit_guess[1]
				redchi_final = redchi_guess
				which_fit_final = which_fit_guess[0]
				result_final = which_fit_guess[2]
				if which_fit_guess[0] == 'single': 
					gamma1_final = g1_guess
					gamma2_final = np.nan
					alpha_final = np.nan
					break_final = np.nan
					cut_final = np.nan
					c1_final = c1_guess
				if which_fit_guess[0] == 'broken': 
					gamma1_final = g1_guess
					gamma2_final = g2_guess
					alpha_final = alpha_guess
					break_final = break_guess
					cut_final = np.nan
					c1_final = c1_guess
				if which_fit_guess[0] == 'cut': 
					gamma1_final = g1_guess
					gamma2_final = g2_guess
					alpha_final = alpha_guess
					break_final = break_guess
					cut_final = cut_guess
					c1_final = c1_guess

			if cut_b>e_min and cut_b< e_max:
				which_fit_final = 'broken_cut'
				result_final = result_cut_guess
				redchi_guess  = result_cut_guess.res_var
				redchi_final = redchi_guess
				gamma1_final = g1_guess
				gamma2_final = g2_guess
				alpha_final = alpha_guess
				break_final = break_guess
				cut_final = cut_guess
				c1_final = c1_guess
			
		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0]
				if g1_random<g2_random:
					gamma = g1_random
					g1_random = g2_random
					g2_random = gamma
				alpha_random = np.random.choice(alpha_array, 1)[0]
				break_random = np.random.choice(break_array,1)[0]
				cut_random = np.random.choice(cut_array, 1)[0]
				c1_random = np.random.choice(c1_array, 1)[0]

				which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, gamma1=g1_random, gamma2=g2_random, E_break=break_random, E_cut = cut_random, fit = 'broken_cut', maxit=10000, e_min = e_min, e_max = e_max)
				redchi_random = which_fit_random[1]
				if redchi_random < redchi_final:
					result_final = which_fit_random[2]
					if which_fit_random[0] == 'single':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = np.nan
						alpha_final = np.nan
						break_final = np.nan
						cut_final = np.nan
						c1_final = c1_random
					if which_fit_random[0] == 'broken':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = g2_random
						alpha_final = alpha_random
						break_final = break_random
						cut_final = np.nan
						c1_final = c1_random
					if which_fit_random[0] == 'cut':
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = np.nan
						alpha_final = np.nan
						break_final = np.nan
						cut_final = cut_random
						c1_final = c1_random	
					if which_fit_random[0] == 'broken_cut':
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						gamma1_final = g1_random
						gamma2_final = g2_random
						alpha_final = alpha_random
						break_final = break_random
						cut_final = cut_random
						c1_final = c1_random



	
	if which_fit == 'broken':
		# even if the which_fit is broken we need to check first if the break point is outside of the energy range. In that case we have to change it to single.
		result_broken_guess = pl_fit.broken_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess, gamma2=g2_guess, c1 = c1_guess, alpha = alpha_guess, E_break = break_guess, maxit=10000)
		breakp = result_broken_guess.beta[4]
		
		if breakp < e_min or breakp > e_max:
			print('The break point is outside of the energy range')
			which_fit_final = 'single'
			result_single_pl_guess = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess, c1=c1_guess)
			result_final = result_single_pl_guess
			redchi_guess  = result_single_pl_guess.res_var  
			redchi_final = redchi_guess
			gamma1_final = g1_guess
			gamma2_final = np.nan
			alpha_final = np.nan
			break_final = np.nan
			cut_final = np.nan
			c1_final = c1_guess
		if breakp >= e_min and breakp <=e_max:
			which_fit_final = 'broken'
			result_final = result_broken_guess
			redchi_guess  = result_broken_guess.res_var
			redchi_final = redchi_guess
			gamma1_final = g1_guess
			gamma2_final = g2_guess
			alpha_final = alpha_guess
			break_final = break_guess
			cut_final = np.nan
			c1_final = c1_guess
		
		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0]
				if g1_random<g2_random:
					gamma = g1_random
					g1_random = g2_random
					g2_random = gamma
				alpha_random = np.random.choice(alpha_array, 1)[0]
				break_random = np.random.choice(break_array,1)[0]
				c1_random = np.random.choice(c1_array, 1)[0]
				result_broken_random = pl_fit.broken_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = g1_random, gamma2 = g2_random, c1 = c1_random, alpha = alpha_random, E_break = break_random, maxit=10000)
				breakp = result_broken_random.beta[4]
				if breakp < e_min or breakp > e_max:
					result_single_pl_random = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_random, c1=c1_random)
					redchi_random  = result_single_pl_random.res_var  
					if redchi_random < redchi_final:
						which_fit_final = 'single'
						redchi_final = redchi_random
						result_final = result_single_pl_random
						gamma1_final = g1_random
						gamma2_final = np.nan
						alpha_final = np.nan
						break_final = np.nan
						cut_final = np.nan
						c1_final = c1_random
				if breakp >= e_min and breakp <=e_max:
					redchi_random = result_broken_random.res_var
					if redchi_random < redchi_final:
						which_fit_final = 'broken'
						redchi_final = redchi_random
						result_final =result_broken_random
						gamma1_final = g1_random
						gamma2_final = g2_random
						alpha_final = alpha_random
						break_final = break_random
						cut_final = np.nan
						c1_final = c1_random

	if which_fit == 'cut':
		# even if the which_fit is broken we need to check first if the break point is outside of the energy range. In that case we have to change it to single.
		result_cut_guess = pl_fit.cut_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess,  c1 = c1_guess,  E_cut = cut_guess, maxit=10000)
		cut = result_cut_guess.beta[2]
		
		if cut < e_min or cut > e_max:
			print('The cutoff point is outside of the energy range')
			which_fit_final = 'single'
			result_single_pl_guess = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess, c1=c1_guess)
			result_final = result_single_pl_guess
			redchi_guess  = result_single_pl_guess.res_var  
			redchi_final = redchi_guess
			gamma1_final = g1_guess
			gamma2_final = np.nan
			alpha_final = np.nan
			break_final = np.nan
			cut_final = np.nan
			c1_final = c1_guess
		if cut >= e_min and cut <=e_max:
			which_fit_final = 'cut'
			result_final = result_cut_guess
			redchi_guess  = result_cut_guess.res_var
			redchi_final = redchi_guess
			gamma1_final = g1_guess
			gamma2_final = np.nan
			alpha_final = np.nan
			break_final = np.nan
			cut_final = cut_guess
			c1_final = c1_guess
		
		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				cut_random = np.random.choice(cut_array,1)[0]
				c1_random = np.random.choice(c1_array, 1)[0]
				result_cut_random = pl_fit.cut_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = g1_random,  c1 = c1_random,  E_cut = cut_random, maxit=10000)
				cut = result_cut_random.beta[2]
				if cut < e_min or cut > e_max:
					result_single_pl_random = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_random, c1=c1_random)
					redchi_random  = result_single_pl_random.res_var  
					if redchi_random < redchi_final:
						which_fit_final = 'single'
						redchi_final = redchi_random
						result_final = result_single_pl_random
						gamma1_final = g1_random
						gamma2_final = np.nan
						alpha_final = np.nan
						break_final = np.nan
						cut_final = np.nan
						c1_final = c1_random
				if cut >= e_min and cut <=e_max:
					redchi_random = result_cut_random.res_var
					if redchi_random < redchi_final:
						which_fit_final = 'cut'
						redchi_final = redchi_random
						result_final =result_cut_random
						gamma1_final = g1_random
						gamma2_final = np.nan
						alpha_final = np.nan
						break_final = np.nan
						cut_final = cut_random
						c1_final = c1_random

	
	if which_fit == 'single':
		which_fit_final = 'single'
		result_single_pl_guess = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess, c1=c1_guess)
		result_final = result_single_pl_guess
		redchi_guess  = result_single_pl_guess.res_var  
		redchi_final = redchi_guess
		gamma1_final = g1_guess
		gamma2_final = np.nan
		alpha_final = np.nan
		break_final = np.nan
		cut_final = np.nan
		c1_final = c1_guess
		if use_random:
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				c1_random = np.random.choice(c1_array, 1)[0]
				result_single_pl_random = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_random, c1=c1_random)
				redchi_random  = result_single_pl_random.res_var  
				if redchi_random < redchi_final:
					redchi_final = redchi_random
					result_final = result_single_pl_random
					gamma1_final = g1_random
					gamma2_final = np.nan
					alpha_final = np.nan
					break_final = np.nan
					cut_final = np.nan
					c1_final = c1_random
		
	
	result_dataframe = pd.DataFrame({"FInal fit type":which_fit_final}, index = [0])
	result = result_final
	print(which_fit_final)
	if which_fit_final == 'single':
		#result_single_pl = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=gamma1_final, c1=c1_final)
		result_single_pl = result_final
		result        = result_final
		# dof_single                   = len(x) - len(result_single_pl.beta)
		redchi_single  = result_single_pl.res_var  #result_single_pl.sum_square / dof_single
		c1          = result_single_pl.beta[0]
		gamma1      = result_single_pl.beta[1]
		dof         = len(spec_e) - len(result_single_pl.beta)
		t_val       = studentt.interval(0.95, dof)[1]
		errors      = t_val * result_single_pl.sd_beta  #np.sqrt(np.diag(final_fit.cov_beta))
		gamma1_err  = errors[1]
		ax.plot(xplot, pl_fit.simple_pl([c1, gamma1], xplot), '-', color=color[direction], label=r'$\mathregular{\delta=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err))
		ax.plot(xplot, pl_fit.simple_pl([c1, gamma1], xplot), '--k', zorder=10)


		result_dataframe["Reduced chi sq"] = redchi_single
		result_dataframe["c1"] = c1
		result_dataframe["c1 err"] = errors[0]
		result_dataframe["Gamma1"] = gamma1
		result_dataframe["Gamma1 err"] = gamma1_err
		result_dataframe["Gamma2"] = None
		result_dataframe["Gamma2 err"] = None
		result_dataframe["Break point [MeV]"] = None
		result_dataframe["Break point err [MeV]"] = None 
		result_dataframe["Exponential cutoff point [MeV]"] = None
		result_dataframe["Cutoff err [MeV]"] = None
		result_dataframe["Alpha"] = None

	
	if which_fit_final == 'broken':
		#result_broken = pl_fit.broken_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err,  gamma1=gamma1_final, gamma2=gamma2_final, c1 = c1_final, alpha=alpha_final,E_break=break_final, maxit=10000)
		result_broken = result_final
		result        = result_final
		breakp        = result_broken.beta[4]
		alpha         = result_broken.beta[3]
		dof           = len(spec_e) - len(result_broken.beta)
		redchi_broken = result_broken.res_var
		t_val      = studentt.interval(0.95, dof)[1]
		errors     = t_val * result_broken.sd_beta  #np.sqrt(np.diag(result_broken.cov_beta))
		breakp_err = errors[4]
		c1         = result_broken.beta[0]

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
			
		##if gamma1<gamma2:
		#	gamma_temp = gamma1
		#	gamma_temp_err = gamma1_err
		#	gamma1 = gamma2
		#	gamma1_err = gamma2_err
		#	gamma2 = gamma_temp
		#	gamma2_err = gamma_temp_err
			
			
		fit_plot = pl_fit.broken_pl_func(result_broken.beta, xplot)
		fit_plot[fit_plot == 0] = np.nan
		ax.plot(xplot, fit_plot, '-b', label=r'$\mathregular{\delta_1=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err)+'\n'+r'$\mathregular{\delta_2=}$%5.2f' %round(gamma2, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma2_err)+'\n'+r'$\mathregular{\alpha=}$%5.2f' %round(alpha, ndigits=2))#, lw=lwd)
		ax.axvline(x=breakp, color='blue', linestyle='--', label=r'$\mathregular{E_b=}$ '+str(round(breakp*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_err*1e3, ndigits=0))+' keV')
	

		result_dataframe["Reduced chi sq"] = redchi_broken
		result_dataframe["c1"] = c1
		result_dataframe["c1 err"] = errors[0]
		result_dataframe["Gamma1"] = gamma1
		result_dataframe["Gamma1 err"] = gamma1_err
		result_dataframe["Gamma2"] = gamma2
		result_dataframe["Gamma2 err"] = gamma2_err
		result_dataframe["Break point [MeV]"] = breakp
		result_dataframe["Break point err [MeV]"] = breakp_err
		result_dataframe["Exponential cutoff point [MeV]"] = None
		result_dataframe["Cutoff err [MeV]"] = None
		result_dataframe["Alpha"] = alpha


	if which_fit_final == 'cut':
		#result_broken = pl_fit.broken_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err,  gamma1=gamma1_final, gamma2=gamma2_final, c1 = c1_final, alpha=alpha_final,E_break=break_final, maxit=10000)
		result_cut = result_final
		result        = result_final
		cut        = result_cut.beta[2]
			#shoud maybe make distinction between cut from cut pl and cut from cut broken pl
		dof           = len(spec_e) - len(result_cut.beta)
		redchi_cut = result_cut.res_var
		t_val      = studentt.interval(0.95, dof)[1]
		errors     = t_val * result_cut.sd_beta  #np.sqrt(np.diag(result_broken.cov_beta))
		c1         = result_cut.beta[0]
		gamma1     = result_cut.beta[1]
		gamma1_err = errors[1]
		cut_err = errors[2]
			
		fit_plot = pl_fit.cut_pl_func(result_cut.beta, xplot)
		fit_plot[fit_plot == 0] = np.nan
		ax.plot(xplot, fit_plot, '-b', label=r'$\mathregular{\delta_1=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err)+'\n'+r'$\mathregular{E_c=}$%5.2f' %round(cut, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(cut_err))#, lw=lwd)
		ax.axvline(x=cut, color='purple', linestyle='--', label=r'$\mathregular{E_c=}$ '+str(round(cut*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(cut_err*1e3, ndigits=0))+' keV')
		

		result_dataframe["Reduced chi sq"] = redchi_cut
		result_dataframe["c1"] = c1
		result_dataframe["c1 err"] = errors[0]
		result_dataframe["Gamma1"] = gamma1
		result_dataframe["Gamma1 err"] = gamma1_err
		result_dataframe["Gamma2"] = None
		result_dataframe["Gamma2 err"] = None
		result_dataframe["Break point [MeV]"] = None
		result_dataframe["Break point err [MeV]"] = None
		result_dataframe["Exponential cutoff point [MeV]"] = cut
		result_dataframe["Cutoff err [MeV]"] = cut_err
		result_dataframe["Alpha"] = None
		
	if which_fit_final == 'broken_cut':
		#result_broken = pl_fit.broken_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err,  gamma1=gamma1_final, gamma2=gamma2_final, c1 = c1_final, alpha=alpha_final,E_break=break_final, maxit=10000)
		result_cut = result_final
		result        = result_final
		cut			  = result_cut.beta[5]
		breakp        = result_cut.beta[4]
		alpha         = result_cut.beta[3]
		dof           = len(spec_e) - len(result_cut.beta)
		redchi_cut = result_cut.res_var
		t_val      = studentt.interval(0.95, dof)[1]
		errors     = t_val * result_cut.sd_beta  #np.sqrt(np.diag(result_cut.cov_beta))
		breakp_err = errors[4]
		cut_err = errors[5]
		c1         = result_cut.beta[0]
	
		if alpha > 0 :
			gamma1     = result_cut.beta[1]
			gamma1_err = errors[1]
			gamma2     = result_cut.beta[2]
			gamma2_err = errors[2]
			
		if alpha < 0 :
			gamma1     = result_cut.beta[2]
			gamma1_err = errors[2]
			gamma2     = result_cut.beta[1]
			gamma2_err = errors[1]
			
		##if gamma1<gamma2:
		#	gamma_temp = gamma1
		#	gamma_temp_err = gamma1_err
		#	gamma1 = gamma2
		#	gamma1_err = gamma2_err
		#	gamma2 = gamma_temp
		#	gamma2_err = gamma_temp_err
		
			
		fit_plot = pl_fit.cut_break_pl_func(result_cut.beta, xplot)
		fit_plot[fit_plot == 0] = np.nan
		ax.plot(xplot, fit_plot, '-b', label=r'$\mathregular{\delta_1=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err)+'\n'+r'$\mathregular{\delta_2=}$%5.2f' %round(gamma2, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma2_err)+'\n'+r'$\mathregular{\alpha=}$%5.2f' %round(alpha, ndigits=2))#, lw=lwd)
		ax.axvline(x=breakp, color='blue', linestyle='--', label=r'$\mathregular{E_b=}$ '+str(round(breakp*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_err*1e3, ndigits=0))+' keV')
		ax.axvline(x=cut, color='purple', linestyle='--', label=r'$\mathregular{E_c=}$ '+str(round(cut*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(cut_err*1e3, ndigits=0))+' keV')

		result_dataframe["Reduced chi sq"] = redchi_cut
		result_dataframe["c1"] = c1
		result_dataframe["c1 err"] = errors[0]
		result_dataframe["Gamma1"] = gamma1
		result_dataframe["Gamma1 err"] = gamma1_err
		result_dataframe["Gamma2"] = gamma2
		result_dataframe["Gamma2 err"] = gamma2_err
		result_dataframe["Break point [MeV]"] = breakp
		result_dataframe["Break point err [MeV]"] = breakp_err
		result_dataframe["Exponential cutoff point [MeV]"] = cut
		result_dataframe["Cutoff err [MeV]"] = cut_err
		result_dataframe["Alpha"] = alpha
	
	#print(result.beta, 'beta')
	#print(result.sd_beta, 'sd_beta')
	#print(result.cov_beta, 'cov_beta')
	#print(result.delta, 'delta')
	#print(result.eps, 'eps')
	#print(result.xplus, 'xplus')
	#print(result.y, 'y')
	#print(result.res_var, 'res_var')
	#print(result.sum_square, 'sum_square')
	#print(result.sum_square_delta, 'sum_square_delta')
	#print(result.sum_square_eps, 'sum_square_eps')
	#print(result.inv_condnum, 'inv_condnum')
	#print(result.rel_error, 'rel_error')
	#print(result.work, 'work')
	#print(result.work_ind, 'work_ind')
	#print(result.info, 'info')
	#print(result.stopreason, 'stopreason')
	
	# save result to pickle file 
	if path != None:
		#pfname =  '.p'
		with open(path, 'wb') as f:
			pickle.dump(result, f)

	# save the fitting variables
	if path2 != None:
		result_dataframe.to_csv(path2, sep = ";")

	#c1 = result.beta[0]
	#print('The fitting variable c1 is ' ,c1)
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
