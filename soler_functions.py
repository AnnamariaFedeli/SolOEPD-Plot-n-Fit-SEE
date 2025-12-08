import numpy as np
import pandas as pd

def exclude_channels(data, channels_to_exclude):
	"""_summary_

    Args:
        data (_type_): _description_
        channels_to_exclude (_type_): _description_

    Returns:
        _type_: _description_
    """
	dataframe_to_fit = data
	dataframe_to_exclude = data
	dataframe_to_fit = dataframe_to_fit.drop(channels_to_exclude, axis = 0)
	dataframe_to_fit = dataframe_to_fit.reset_index(drop=True, inplace=True)
	channels_to_keep = []
	for i in range(len(data)):
		channels_to_keep.append(i)
	for j in channels_to_exclude:
		channels_to_keep.remove(j)
	dataframe_to_exclude = dataframe_to_exclude.drop(channels_to_keep, axis = 0)
	dataframe_to_exclude = dataframe_to_exclude.reset_index(drop=True, inplace=True)
	return [dataframe_to_fit, dataframe_to_exclude]# return two dataframes. one is the dataframe that will be fit and one will be the one plotted in gray that has the excluded channels.
    

