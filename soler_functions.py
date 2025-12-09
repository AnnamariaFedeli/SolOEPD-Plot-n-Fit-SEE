import numpy as np
import pandas as pd

def exclude_channels(data, channels_to_exclude):
	"""This function excludes chosen channels from a dataframe and outputs two dataframes: one with the channels that should not be excluded from the fit and one with the excluded channles.
	One will be the input to mske the fit, the other one is just for plotting the excluded channels in gray.

    Args:
        data (pd.DataFrame): the data that you want to fit
        channels_to_exclude (list of integers): a list containing the indices corresponding to the channels you wish to exclude

    Returns:
        list with two pd.DataFrames: first dataframe contains the data to be fit and the second one the excluded channels.
    """
	
	dataframe_to_fit = data
	dataframe_to_exclude = data
	dataframe_to_fit = dataframe_to_fit.drop(channels_to_exclude, axis = 0)
	
	dataframe_to_fit = dataframe_to_fit.reset_index()
	
	channels_to_keep = []
	for i in range(len(data)):
		channels_to_keep.append(i)
	for j in channels_to_exclude:
		channels_to_keep.remove(j)
	
	dataframe_to_exclude = dataframe_to_exclude.drop(channels_to_keep, axis = 0)
	dataframe_to_exclude = dataframe_to_exclude.reset_index()
	return [dataframe_to_fit, dataframe_to_exclude]# return two dataframes. one is the dataframe that will be fit and one will be the one plotted in gray that has the excluded channels.
    

