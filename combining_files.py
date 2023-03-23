import os
import glob
import pandas as pd



def combine_data(data_name_list, path, sigma = 3, rel_err = 0.5, frac_nan_threshold = 0.9, leave_out_1st_het_chan = False, fit_to = 'Peak'):
	#print(data_name_list)
	if leave_out_1st_het_chan and len(data_name_list)>2:
		het = data_name_list[-1]
		first_het = het.index[data_name_list[-1]['Primary_energy']< 0.7].tolist()
		het = het.drop(first_het, axis = 0)
		het.reset_index(drop=True, inplace=True)
		data_name_list[-1] = het 
	
	combined_csv = pd.concat(data_name_list)
	combined_csv.reset_index(drop=True, inplace=True)
	combined_csv = combined_csv.drop(columns = 'Energy_channel')
	
	rows_to_delete = combined_csv.index[combined_csv[fit_to+'_significance'] <sigma].tolist()
	combined_csv = combined_csv.drop(rows_to_delete, axis = 0)
	combined_csv.reset_index(drop=True, inplace=True)
	rows_to_delete = combined_csv.index[combined_csv['rel_backsub_peak_err']> rel_err].tolist()
	combined_csv = combined_csv.drop(rows_to_delete, axis = 0)
	combined_csv.reset_index(drop=True, inplace=True)
	rows_to_delete = combined_csv.index[combined_csv['frac_nonan']<frac_nan_threshold].tolist()
	combined_csv = combined_csv.drop(rows_to_delete, axis = 0)
	combined_csv.reset_index(drop=True, inplace=True)
	
	combined_csv.to_csv(path, sep = ';')
	
	return(combined_csv)

def low_sigma_threshold(data_name_list, sigma = 3, leave_out_1st_het_chan = False, fit_to = 'Peak'):
	combined_csv = pd.concat(data_name_list)
	combined_csv.reset_index(drop=True, inplace=True)
	combined_csv = combined_csv.drop(columns = 'Energy_channel')
	
	if leave_out_1st_het_chan:
		het = data_name_list[-1]
		first_het = het.index[data_name_list[-1]['Primary_energy']< 0.7].tolist()
		het = het.drop(first_het, axis = 0)
		het.reset_index(drop=True, inplace=True)
		data_name_list[-1] = het 
	

	rows_to_delete = combined_csv.index[combined_csv[fit_to+'_significance'] >sigma ].tolist()
	combined_csv = combined_csv.drop(rows_to_delete, axis = 0)
	combined_csv.reset_index(drop=True, inplace=True)
	
	return(combined_csv)

def too_many_nans(data_name_list, frac_nan_threshold = 0.9, leave_out_1st_het_chan = False):
	combined_csv = pd.concat(data_name_list)
	combined_csv.reset_index(drop=True, inplace=True)
	combined_csv = combined_csv.drop(columns = 'Energy_channel')
	
	if leave_out_1st_het_chan:
		het = data_name_list[-1]
		first_het = het.index[data_name_list[-1]['Primary_energy']< 0.7].tolist()
		het = het.drop(first_het, axis = 0)
		het.reset_index(drop=True, inplace=True)
		data_name_list[-1] = het 
	

	rows_to_delete = combined_csv.index[combined_csv['frac_nonan']>frac_nan_threshold].tolist()
	combined_csv = combined_csv.drop(rows_to_delete, axis = 0)
	combined_csv.reset_index(drop=True, inplace=True)
	
	return(combined_csv)

def high_rel_err(data_name_list, rel_err = 0.5, leave_out_1st_het_chan = False):
	combined_csv = pd.concat(data_name_list)
	combined_csv.reset_index(drop=True, inplace=True)
	combined_csv = combined_csv.drop(columns = 'Energy_channel')
	
	if leave_out_1st_het_chan:
		het = data_name_list[-1]
		first_het = het.index[data_name_list[-1]['Primary_energy']< 0.7].tolist()
		het = het.drop(first_het, axis = 0)
		het.reset_index(drop=True, inplace=True)
		data_name_list[-1] = het 
	

	rows_to_delete = combined_csv.index[combined_csv['rel_backsub_peak_err']< rel_err ].tolist()
	combined_csv = combined_csv.drop(rows_to_delete, axis = 0)
	combined_csv.reset_index(drop=True, inplace=True)
	
	return(combined_csv)
	
def delete_bad_data(data, sigma = 3, rel_err = 0.5, frac_nan_threshold = 0.9, leave_out_1st_het_chan = False, fit_to = 'Peak'):
	data = data.drop(columns = 'Energy_channel')
	rows_to_delete = data.index[data[fit_to+'_significance'] <sigma].tolist()
	data = data.drop(rows_to_delete, axis = 0)
	data.reset_index(drop=True, inplace=True)
	rows_to_delete = data.index[data['rel_backsub_peak_err']> rel_err].tolist()
	data = data.drop(rows_to_delete, axis = 0)
	data.reset_index(drop=True, inplace=True)
	rows_to_delete = data.index[data['frac_nonan']<frac_nan_threshold].tolist()
	data = data.drop(rows_to_delete, axis = 0)
	data.reset_index(drop=True, inplace=True)
	
	if leave_out_1st_het_chan:
		first_het = data.index[data['Primary_energy']< 0.7].tolist()
		data = data.drop(first_het, axis = 0)
		data.reset_index(drop=True, inplace=True)

	return(data)
	

def first_het_chan(data):
	first_het = data.index[data['Primary_energy']> 0.7].tolist()
	data = data.drop(first_het, axis = 0)
	data.reset_index(drop=True, inplace=True)
	return(data)
	
