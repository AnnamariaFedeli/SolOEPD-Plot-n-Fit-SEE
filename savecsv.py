import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from pytimeparse.timeparse import timeparse


def save_info_plot(path, plot_start, plot_end, t_inj, bgstart , bgend, bg_distance_from_window , 
bg_period, travel_distance, travel_distance_second_slope, fixed_window, data_type, 
averaging_mode, averaging, masking, ion_conta_corr, dist, spiral_len, traveltime_min, 
traveltime_max, light_tt):
     """_summary_

     Args:
         path (_type_): _description_
         plot_start (_type_): _description_
         plot_end (_type_): _description_
         t_inj (_type_): _description_
         bgstart (_type_): _description_
         bgend (_type_): _description_
         bg_distance_from_window (_type_): _description_
         bg_period (_type_): _description_
         travel_distance (_type_): _description_
         travel_distance_second_slope (_type_): _description_
         fixed_window (_type_): _description_
         data_type (_type_): _description_
         averaging_mode (_type_): _description_
         averaging (_type_): _description_
         masking (_type_): _description_
         ion_conta_corr (_type_): _description_
         dist (_type_): _description_
         spiral_len (_type_): _description_
         traveltime_min (_type_): _description_
         traveltime_max (_type_): _description_
         light_tt (_type_): _description_

     Returns:
         _type_: _description_
     """
     df = pd.DataFrame({"Plot Start" : plot_start, "Plot end": plot_end, 
     "Injection time at Sun": t_inj, "Background start" : bgstart, "Background end" : bgend, 
     "Bg distance from window [min]": bg_distance_from_window, "Bg period [min]": bg_period, 
     "Travel distance first slope [AU]" : travel_distance, 
     "Travel distance second slope [AU]" : travel_distance_second_slope , 
     "Fixed window [min]": fixed_window, "data type" : data_type, 
     "Averaging mode" : averaging_mode, "Averaging [s]": timeparse(averaging), "Masking" : masking, 
     "Ion contamination corection":ion_conta_corr, "Distance of s/c [AU]":dist,
     "Length of Parker Spiral [AU]":spiral_len, "Travel time 4keV [min]" :traveltime_min,
     "Traveltime 10MeV [min]": traveltime_max, "Traveltimeof light at distance D [min]": light_tt}, index = [0])
     
     df.to_csv(path, sep = ';')

     return df
     

def save_info_fit(path, date_string, averaging, direction, data_product, dist, step, ept, het,
                  sigma, rel_err, frac_nan_threshold, leave_out_1st_het_chan, shift_factor, fit_type, fit_to, which_fit, e_min, e_max, g1_guess, g2_guess, c1_guess, alpha_guess, break_guess,
                  cut_guess,use_random, iterations):

    df = pd.DataFrame({"Date": date_string, "Averaging [s]":timeparse(averaging), "Direction":direction,
    "Data type":data_product, "Distance [AU]":dist, "STEP":step, "EPT":ept, "HET":het, 
    "Sigma":sigma, "Relative error":rel_err, "Fraction of nan":frac_nan_threshold,
    "Leave first HET channel out":leave_out_1st_het_chan, "Shift STEP data": shift_factor,
    "Type of fit":fit_type, "Fit to":fit_to, "Which fit":which_fit , 
    "Min energy": e_min, "Max energy": e_max, "Gamma1 guess":g1_guess, "Gamma2 guess":g2_guess,
    "c1 guess": c1_guess, "Alpha guess": alpha_guess, "Break guess [MeV]":break_guess, 
    "Cutoff point guess [MeV]":cut_guess,
    "Use random":use_random, "Iterations":iterations}, index = [0])

    df.to_csv(path, sep = ";")

    return df

