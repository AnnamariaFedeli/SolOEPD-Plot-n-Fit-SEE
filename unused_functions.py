import numpy as np

def average_flux_error(flux_err: pd.DataFrame) -> pd.Series:
    """Calculate the average flux error for each column.

    Parameters
    ----------
    flux_err : pandas.DataFrame
        Flux uncertainties, with each column representing a quantity
        for which the average error is calculated.

    Returns
    -------
    pandas.Series
        Average flux error for each column.
    """
    return np.sqrt((flux_err**2).sum(axis=0)) / len(flux_err.values)

def find_c1(spec_e, spec_flux, e_min, e_max):
    """Find the value of c1 from the spectrum near the maximum energy.

    Args:
        spec_e: Energy values of the spectrum.
        spec_flux: Flux values corresponding to ``spec_e``.
        e_min: Minimum energy of the fitting range.
        e_max: Maximum energy of the fitting range.

    Returns:
        The calculated value of c1.
    """
    absolute_val_array = np.abs(spec_e - e_max)
    smallest_difference_index = absolute_val_array.argmin()
    closest_element = spec_e[smallest_difference_index]

    x1 = np.log10(spec_e[smallest_difference_index - 5])
    y1 = np.log10(spec_flux[smallest_difference_index - 5])

    x2 = np.log(spec_e[smallest_difference_index])
    y2 = np.log10(spec_flux[smallest_difference_index])

    m = (y1 - y2) / (x1 - x2)
    q = (x1 * y2 - x2 * y1) / (x1 - x2)

    c1 = m * 1.0 + q

    print("x1", x1, "y1", y1, "x2", x2, "y2", y2)

    return c1

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
