# solo_functions.py
import numpy as np
import pandas as pd
import my_power_law_fits_odr as pl_fit
from scipy.stats import t as studentt
#from lmfit.models import GaussianModel
import pickle

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

def closest_values(array, value):
    """Find the values closest to a given value.

    The number of returned values depends on the length of the input
    array:

    - Up to 10 values: return all available values.
    - 11 to 20 values: return half of the input values, rounded.
    - More than 20 values: return 10 values.

    The value equal to ``value`` is excluded from the search.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing the values to search.
    value : float
        Value for which the closest values are found.

    Returns
    -------
    list
        Sorted values closest to ``value``.
    """
    array_length = len(array)

    if array_length <= 10:
        n_closest = array_length
    elif array_length <= 20:
        n_closest = round(array_length / 2)
    else:
        n_closest = 10

    array = np.delete(array, np.where(array == value))

    closest_values_array = np.array(())

    for _ in range(n_closest):
        absolute_val_array = np.abs(array - value)
        smallest_difference_index = absolute_val_array.argmin()
        closest_element = array[smallest_difference_index]

        closest_values_array = np.append(
            closest_values_array,
            closest_element,
        )

        array = np.delete(array, np.where(array == closest_element))

    return sorted(closest_values_array)


def check_redchi_old(
    spec_e,
    spec_flux,
    e_err,
    flux_err,
    gamma1=-1,
    gamma2=-2,
    gamma3=-4,
    c1=1000,
    alpha=10,
    beta=10,
    E_break_low=0.06,
    E_break_high=0.1,
    E_cut=None,
    exponent=2,
    fit="best",
    maxit=10000,
    e_min=None,
    e_max=None,
):
    """Check reduced chi-squared values for different spectral models.

    Parameters
    ----------
    spec_e, spec_flux : array-like
        Energy and flux data to be fitted.
    e_err, flux_err : array-like
        Uncertainties in energy and flux, respectively.
    gamma1, gamma2, gamma3 : float, optional
        Initial guesses for the power-law indices.
    c1 : float, optional
        Initial guess for the normalization.
    alpha, beta : float, optional
        Initial guesses for the smoothness parameters.
    E_break_low, E_break_high : float, optional
        Initial guesses for the lower and upper break energies.
    E_cut : float or None, optional
        Initial guess for the cut-off energy.
    exponent : float, optional
        Initial guess for the cut-off exponent.
    fit : str, optional
        Determines which fitting procedure is used.
    maxit : int, optional
        Maximum number of ODR iterations.
    e_min, e_max : float or None, optional
        Minimum and maximum energy limits used when checking fitted
        break and cut-off positions.

    Returns
    -------
    list
        A list containing the selected fit name, reduced chi-squared
        value, and ODR result.
    """
    # The function also checks if the break point is outside of the
    # energy array (also the cutoff point).
    # The min and max energies cannot be the last and/or first points
    # because it wouldn't be a physical result.

    emin = spec_e[2]
    emax = spec_e[len(spec_e) - 3]

    if e_min is None or e_min == spec_e[0]:
        emin = spec_e[2]
    else:
        emin = e_min

    if e_max is None or e_max == spec_e[len(spec_e) - 1]:
        emax = spec_e[len(spec_e) - 3]
    else:
        emax = e_max

    if fit == "best":
        result_triple = pl_fit.triple_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            gamma3=gamma3,
            c1=c1,
            alpha=alpha,
            beta=beta,
            E_break_low=E_break_low,
            E_break_high=E_break_high,
            maxit=maxit,
        )
        redchi_triple = result_triple.res_var
        breakp_low = result_triple.beta[6]
        breakp_high = result_triple.beta[7]
        difference_triple = np.abs(breakp_high - breakp_low)
        alpha = result_triple.beta[4]
        beta = result_triple.beta[5]

        if alpha > 0:
            gamma1 = result_triple.beta[1]
        elif alpha <= 0:
            gamma1 = result_triple.beta[2]

        result_cut_break = pl_fit.cut_break_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            E_cut=E_cut,
            exponent=exponent,
            print_report=False,
            maxit=maxit,
        )
        redchi_cut_break = result_cut_break.res_var
        breakp_cut = result_cut_break.beta[4]
        cut_b = result_cut_break.beta[5]
        difference_cut = np.abs(breakp_cut - cut_b)
        exponent_cut_break = result_cut_break.beta[6]

        result_cut = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            E_cut=E_cut,
            exponent=exponent,
            maxit=maxit,
        )
        redchi_cut = result_cut.res_var
        cut = result_cut.beta[2]
        exponent_cut = result_cut.beta[3]

        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
        )
        redchi_double = result_double.res_var
        breakp = result_double.beta[4]

        result_single_pl = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
        )
        redchi_single = result_single_pl.res_var

        chis = {
            "triple": redchi_triple,
            "double_cut": redchi_cut_break,
            "cut": redchi_cut,
            "double": redchi_double,
            "single": redchi_single,
        }

        sorted_chis = dict(
            sorted(chis.items(), key=lambda x: x[1], reverse=False)
        )

        if exponent_cut == 0:
            sorted_chis.pop("cut")

        if exponent_cut_break == 0:
            sorted_chis.pop("double_cut")

        # Check if there are values with zero chi sq. If so, delete
        # from dict. Then check if dict is empty. If yes: loop again
        # through the results.
        #
        # Temporary solution will be implemented better later 14.04.26
        if pl_fit.check_odr_output(result_triple) is False:
            sorted_chis.pop("triple")

        if pl_fit.check_odr_output(result_cut_break) is False:
            sorted_chis.pop("double_cut")

        if pl_fit.check_odr_output(result_double) is False:
            sorted_chis.pop("double")

        if pl_fit.check_odr_output(result_single_pl) is False:
            sorted_chis.pop("single")

        list_zero_chi = []

        for i in sorted_chis:
            if sorted_chis[i] == 0.0:
                list_zero_chi.append(i)

        for i in list_zero_chi:
            sorted_chis.pop(i)

        smallest_value = list(sorted_chis.keys())[0]

        for i in range(len(sorted_chis)):
            # Make if statements to check values etc. of breaks and so on
            # and change the smallest value after checking everything.
            if smallest_value == "triple":
                if (
                    emin < breakp_low < emax
                    and emin < breakp_high < emax
                    and breakp_low < breakp_high
                ):
                    absolute_val_array = np.abs(spec_e - breakp_low)
                    smallest_difference_index = absolute_val_array.argmin()
                    low = ""
                    high = ""

                    if smallest_difference_index == len(spec_e) - 1:
                        low = (
                            spec_e[smallest_difference_index - 1]
                            - e_err[smallest_difference_index - 1]
                        )
                        high = (
                            spec_e[smallest_difference_index]
                            + e_err[smallest_difference_index]
                        )
                    else:
                        low = (
                            spec_e[smallest_difference_index]
                            - e_err[smallest_difference_index]
                        )
                        high = (
                            spec_e[smallest_difference_index + 1]
                            + e_err[smallest_difference_index + 1]
                        )

                    difference_triple_energy = high - low

                    # The triple PL is defined so that the two breaks are
                    # actually interchangeable. This means that it can
                    # happen that the 'high' break becomes the low break.
                    # These cases have to be deleted because it messes with
                    # the meaning of the parameters of the fit.

                    if (
                        difference_triple > difference_triple_energy
                        and gamma1 < 0
                    ):
                        if alpha > 0 or beta > 0:
                            which_fit = "triple"
                            redchi = redchi_triple
                            result = result_triple
                            return [which_fit, redchi, result]
                        else:
                            smallest_value = list(sorted_chis.keys())[i]
                            # There are cases in which these statements lead
                            # to no return because all options are bad.
                            # This needs to be fixed by somehow adding options
                            # to the list 1.11.23.
                    else:
                        smallest_value = list(sorted_chis.keys())[i]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "double_cut":
                if (
                    emin < cut_b < emax
                    and emin < breakp_cut < emax
                    and cut_b > breakp_cut
                ):
                    absolute_val_array = np.abs(spec_e - breakp_cut)
                    smallest_difference_index = absolute_val_array.argmin()
                    low = ""
                    high = ""

                    if smallest_difference_index == len(spec_e) - 1:
                        low = (
                            spec_e[smallest_difference_index - 1]
                            - e_err[smallest_difference_index - 1]
                        )
                        high = (
                            spec_e[smallest_difference_index]
                            + e_err[smallest_difference_index]
                        )
                    else:
                        low = (
                            spec_e[smallest_difference_index]
                            - e_err[smallest_difference_index]
                        )
                        high = (
                            spec_e[smallest_difference_index + 1]
                            + e_err[smallest_difference_index + 1]
                        )

                    difference_cut_energy = high - low

                    if (
                        gamma1 < 0
                        and difference_cut > difference_cut_energy
                    ):
                        which_fit = "double_cut"
                        redchi = redchi_cut_break
                        result = result_cut_break
                        return [which_fit, redchi, result]
                    else:
                        smallest_value = list(sorted_chis.keys())[i]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "cut":
                if emin <= cut <= emax:
                    which_fit = "cut"
                    redchi = redchi_cut
                    result = result_cut
                    return [which_fit, redchi, result]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "double":
                if emin <= breakp <= emax:
                    which_fit = "double"
                    redchi = redchi_double
                    result = result_double
                    return [which_fit, redchi, result]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "single":
                which_fit = "single"
                redchi = redchi_single
                result = result_single_pl
                return [which_fit, redchi, result]

        # Redo loop either because list is already empty or because none
        # of the previous options worked.

    if fit == "triple":
        result_triple = pl_fit.triple_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            gamma3=gamma3,
            c1=c1,
            alpha=alpha,
            beta=beta,
            E_break_low=E_break_low,
            E_break_high=E_break_high,
            maxit=maxit,
        )
        redchi_triple = result_triple.res_var
        breakp_low = result_triple.beta[6]
        breakp_high = result_triple.beta[7]
        difference_triple = breakp_high - breakp_low

        if (
            breakp_low < emax
            and breakp_low > emin
            and breakp_high < emax
            and breakp_high > emin
        ):
            absolute_val_array = np.abs(spec_e - breakp_low)
            smallest_difference_index = absolute_val_array.argmin()
            low = ""
            high = ""

            if smallest_difference_index == len(spec_e) - 1:
                low = (
                    spec_e[smallest_difference_index - 1]
                    - e_err[smallest_difference_index - 1]
                )
                high = (
                    spec_e[smallest_difference_index]
                    + e_err[smallest_difference_index]
                )
            else:
                low = (
                    spec_e[smallest_difference_index]
                    - e_err[smallest_difference_index]
                )
                high = (
                    spec_e[smallest_difference_index + 1]
                    + e_err[smallest_difference_index + 1]
                )

            difference_triple_energy = high - low

            if (
                breakp_high > breakp_low
                and difference_triple > difference_triple_energy
            ):
                which_fit = "triple"
                redchi = redchi_triple
                result = result_triple
                return [which_fit, redchi, result]

            else:
                fit = "double_cut"

        else:
            fit = "double_cut"

    if fit == "double_cut":
        result_cut_break = pl_fit.cut_break_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            E_cut=E_cut,
            exponent=exponent,
            print_report=False,
            maxit=maxit,
        )
        redchi_cut_break = result_cut_break.res_var
        breakp_cut = result_cut_break.beta[4]

        # The cut of the break + cutoff.
        cut_b = result_cut_break.beta[5]
        difference_cut = breakp_cut - cut_b

        if (
            breakp_cut <= emax
            and breakp_cut > emin
            and cut_b <= emax
            and cut_b > emin
        ):
            absolute_val_array = np.abs(spec_e - breakp_cut)
            smallest_difference_index = absolute_val_array.argmin()
            low = ""
            high = ""

            if smallest_difference_index == len(spec_e) - 1:
                low = (
                    spec_e[smallest_difference_index - 1]
                    - e_err[smallest_difference_index - 1]
                )
                high = (
                    spec_e[smallest_difference_index]
                    + e_err[smallest_difference_index]
                )
            else:
                low = (
                    spec_e[smallest_difference_index]
                    - e_err[smallest_difference_index]
                )
                high = (
                    spec_e[smallest_difference_index + 1]
                    + e_err[smallest_difference_index + 1]
                )

            difference_cut_energy = high - low

            if cut_b > breakp_cut and difference_cut > difference_cut_energy:
                which_fit = "double_cut"
                redchi = redchi_cut_break
                result = result_cut_break
                return [which_fit, redchi, result]

            else:
                fit = "best_cb"

        else:
            fit = "best_cb"

    if fit == "best_sb":
        result_single_pl = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
        )
        redchi_single = result_single_pl.res_var

        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
        )
        redchi_double = result_double.res_var
        breakp = result_double.beta[4]

        if redchi_double <= redchi_single:
            if breakp < emin or breakp > emax:
                which_fit = "single"
                redchi = redchi_single
                result = result_single_pl
                return [which_fit, redchi, result]

            if emin <= breakp <= emax:
                which_fit = "double"
                redchi = redchi_double
                result = result_double
                return [which_fit, redchi, result]

        if redchi_double > redchi_single:
            which_fit = "single"
            redchi = redchi_single
            result = result_single_pl
            return [which_fit, redchi, result]

    if fit == "best_cb":
        result_cut = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            E_cut=E_cut,
            exponent=exponent,
            maxit=maxit,
        )
        redchi_cut = result_cut.res_var

        # Should maybe make distinction between cut from cut PL
        # and cut from cut double PL.
        cut = result_cut.beta[2]

        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
        )
        redchi_double = result_double.res_var
        breakp = result_double.beta[4]

        if redchi_double <= redchi_cut:
            if breakp < emin or breakp > emax:
                fit = "single"

            if emin <= breakp <= emax:
                which_fit = "double"
                redchi = redchi_double
                result = result_double
                return [which_fit, redchi, result]

        if redchi_double > redchi_cut:
            if cut < emin or cut > emax:
                fit = "single"

            if emin <= cut <= emax:
                which_fit = "cut"
                redchi = redchi_cut
                result = result_cut
                return [which_fit, redchi, result]

    if fit == "cut":
        result_cut = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            E_cut=E_cut,
            exponent=exponent,
            maxit=maxit,
        )
        redchi_cut = result_cut.res_var

        # Should maybe make distinction between cut from cut PL
        # and cut from cut double PL.
        cut = result_cut.beta[2]

        if cut < emin or cut > emax:
            fit = "single"

        if emin <= cut <= emax:
            which_fit = "cut"
            redchi = redchi_cut
            result = result_cut
            return [which_fit, redchi, result]

    if fit == "double":
        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
        )
        redchi_double = result_double.res_var
        breakp = result_double.beta[4]

        if breakp < emin or breakp > emax:
            fit = "single"

        if emin <= breakp <= emax:
            which_fit = "double"
            redchi = redchi_double
            result = result_double
            return [which_fit, redchi, result]

    if fit == "single":
        result_single_pl = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
        )
        redchi_single = result_single_pl.res_var

        which_fit = "single"
        redchi = redchi_single
        result = result_single_pl
        return [which_fit, redchi, result]    
    
def find_c1(spec_e, spec_flux, e_min, e_max):
    """_summary_

    Args:
        spec_e (_type_): _description_
        spec_flux (_type_): _description_
        e_min (_type_): _description_
        e_max (_type_): _description_
    """
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
    
def _break_energy_interval(spec_e, e_err, break_energy):
    """Return the energy interval surrounding the data point nearest a break."""
    absolute_val_array = np.abs(spec_e - break_energy)
    smallest_difference_index = absolute_val_array.argmin()

    if smallest_difference_index == len(spec_e) - 1:
        low = (
            spec_e[smallest_difference_index - 1]
            - e_err[smallest_difference_index - 1]
        )
        high = (
            spec_e[smallest_difference_index]
            + e_err[smallest_difference_index]
        )
    else:
        low = (
            spec_e[smallest_difference_index]
            - e_err[smallest_difference_index]
        )
        high = (
            spec_e[smallest_difference_index + 1]
            + e_err[smallest_difference_index + 1]
        )

    return low, high


def check_redchi(
    spec_e,
    spec_flux,
    e_err,
    flux_err,
    gamma1=-1,
    gamma2=-2,
    gamma3=-4,
    c1=1000,
    alpha=10,
    beta=10,
    E_break_low=0.06,
    E_break_high=0.1,
    E_cut=None,
    exponent=2,
    fit="best",
    maxit=10000,
    e_min=None,
    e_max=None,
):
    """Check reduced chi-squared values for different spectral models.

    Parameters
    ----------
    spec_e, spec_flux : array-like
        Energy and flux data to be fitted.
    e_err, flux_err : array-like
        Uncertainties in energy and flux, respectively.
    gamma1, gamma2, gamma3 : float, optional
        Initial guesses for the power-law indices.
    c1 : float, optional
        Initial guess for the normalization.
    alpha, beta : float, optional
        Initial guesses for the smoothness parameters.
    E_break_low, E_break_high : float, optional
        Initial guesses for the lower and upper break energies.
    E_cut : float or None, optional
        Initial guess for the cut-off energy.
    exponent : float, optional
        Initial guess for the cut-off exponent.
    fit : str, optional
        Determines which fitting procedure is used.
    maxit : int, optional
        Maximum number of ODR iterations.
    e_min, e_max : float or None, optional
        Minimum and maximum energy limits used when checking fitted
        break and cut-off positions.

    Returns
    -------
    list
        A list containing the selected fit name, reduced chi-squared
        value, and ODR result.
    """
    # The function also checks if the break point is outside of the
    # energy array (also the cutoff point).
    # The min and max energies cannot be the last and/or first points
    # because it wouldn't be a physical result.

    emin = spec_e[2]
    emax = spec_e[len(spec_e) - 3]

    if e_min is None or e_min == spec_e[0]:
        emin = spec_e[2]
    else:
        emin = e_min

    if e_max is None or e_max == spec_e[len(spec_e) - 1]:
        emax = spec_e[len(spec_e) - 3]
    else:
        emax = e_max

    # ------------------------------------------------------------------
    # Select the best model based on reduced chi-squared.
    # ------------------------------------------------------------------
    if fit == "best":
        result_triple = pl_fit.triple_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            gamma3=gamma3,
            c1=c1,
            alpha=alpha,
            beta=beta,
            E_break_low=E_break_low,
            E_break_high=E_break_high,
            maxit=maxit,
        )
        redchi_triple = result_triple.res_var
        breakp_low = result_triple.beta[6]
        breakp_high = result_triple.beta[7]
        difference_triple = np.abs(breakp_high - breakp_low)

        alpha = result_triple.beta[4]
        beta = result_triple.beta[5]

        if alpha > 0:
            gamma1 = result_triple.beta[1]
        elif alpha <= 0:
            gamma1 = result_triple.beta[2]

        result_cut_break = pl_fit.cut_break_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            E_cut=E_cut,
            exponent=exponent,
            print_report=False,
            maxit=maxit,
        )
        redchi_cut_break = result_cut_break.res_var
        breakp_cut = result_cut_break.beta[4]
        cut_b = result_cut_break.beta[5]
        difference_cut = np.abs(breakp_cut - cut_b)
        exponent_cut_break = result_cut_break.beta[6]

        result_cut = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            E_cut=E_cut,
            exponent=exponent,
            maxit=maxit,
        )
        redchi_cut = result_cut.res_var
        cut = result_cut.beta[2]
        exponent_cut = result_cut.beta[3]

        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
        )
        redchi_double = result_double.res_var
        breakp = result_double.beta[4]

        result_single_pl = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
        )
        redchi_single = result_single_pl.res_var

        chis = {
            "triple": redchi_triple,
            "double_cut": redchi_cut_break,
            "cut": redchi_cut,
            "double": redchi_double,
            "single": redchi_single,
        }

        sorted_chis = dict(
            sorted(chis.items(), key=lambda item: item[1])
        )

        if exponent_cut == 0:
            sorted_chis.pop("cut")

        if exponent_cut_break == 0:
            sorted_chis.pop("double_cut")

        # Check whether the ODR fits converged successfully.
        #
        # Temporary solution will be implemented better later 14.04.26
        if pl_fit.check_odr_output(result_triple) is False:
            sorted_chis.pop("triple")

        if pl_fit.check_odr_output(result_cut_break) is False:
            sorted_chis.pop("double_cut")

        if pl_fit.check_odr_output(result_double) is False:
            sorted_chis.pop("double")

        if pl_fit.check_odr_output(result_single_pl) is False:
            sorted_chis.pop("single")

        # Check if there are values with zero chi-squared.
        # If so, remove them from the dictionary.
        list_zero_chi = []

        for model_name in sorted_chis:
            if sorted_chis[model_name] == 0.0:
                list_zero_chi.append(model_name)


        for model_name in list_zero_chi:
            sorted_chis.pop(model_name)

        if not sorted_chis:
            return None

        smallest_value = list(sorted_chis.keys())[0]

        for i in range(len(sorted_chis)):
            # Check the candidate models in order of increasing
            # reduced chi-squared. The physical validity of each
            # fitted break/cutoff is checked before accepting it.

            if smallest_value == "triple":
                if (
                    breakp_low < emax
                    and breakp_low > emin
                    and breakp_high < emax
                    and breakp_high > emin
                    and breakp_low < breakp_high
                ):
                    low, high = _break_energy_interval(
                        spec_e,
                        e_err,
                        breakp_low,
                    )
                    difference_triple_energy = high - low

                    # The triple PL is defined so that the two breaks are
                    # actually interchangeable. This means that it can
                    # happen that the 'high' break becomes the low break.
                    # These cases have to be deleted because it messes with
                    # the meaning of the parameters of the fit.

                    if (
                        difference_triple > difference_triple_energy
                        and gamma1 < 0
                    ):
                        if alpha > 0 or beta > 0:
                            return [
                                "triple",
                                redchi_triple,
                                result_triple,
                            ]
                        else:
                            smallest_value = list(sorted_chis.keys())[i]
                            # There are cases in which these statements lead
                            # to no return because all options are bad.
                            # This needs to be fixed by somehow adding options
                            # to the list 1.11.23.
                    else:
                        smallest_value = list(sorted_chis.keys())[i]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "double_cut":
                if (
                    cut_b > emin
                    and cut_b < emax
                    and breakp_cut > emin
                    and breakp_cut < emax
                    and cut_b > breakp_cut
                ):
                    low, high = _break_energy_interval(
                        spec_e,
                        e_err,
                        breakp_cut,
                    )
                    difference_cut_energy = high - low

                    if (
                        gamma1 < 0
                        and difference_cut > difference_cut_energy
                    ):
                        return [
                            "double_cut",
                            redchi_cut_break,
                            result_cut_break,
                        ]
                    else:
                        smallest_value = list(sorted_chis.keys())[i]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "cut":
                if cut >= emin and cut <= emax:
                    return ["cut", redchi_cut, result_cut]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "double":
                if breakp >= emin and breakp <= emax:
                    return ["double", redchi_double, result_double]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "single":
                return ["single", redchi_single, result_single_pl]

        # Redo loop either because list is already empty or because none
        # of the previous options worked.

    # ------------------------------------------------------------------
    # Explicit triple fit.
    # ------------------------------------------------------------------
    if fit == "triple":
        result_triple = pl_fit.triple_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            gamma3=gamma3,
            c1=c1,
            alpha=alpha,
            beta=beta,
            E_break_low=E_break_low,
            E_break_high=E_break_high,
            maxit=maxit,
        )
        redchi_triple = result_triple.res_var
        breakp_low = result_triple.beta[6]
        breakp_high = result_triple.beta[7]
        difference_triple = breakp_high - breakp_low

        if (
            breakp_low < emax
            and breakp_low > emin
            and breakp_high < emax
            and breakp_high > emin
        ):
            low, high = _break_energy_interval(
                spec_e,
                e_err,
                breakp_low,
            )
            difference_triple_energy = high - low

            if (
                breakp_high > breakp_low
                and difference_triple > difference_triple_energy
            ):
                return ["triple", redchi_triple, result_triple]

            else:
                fit = "double_cut"

        else:
            fit = "double_cut"

    # ------------------------------------------------------------------
    # Explicit double power law with cutoff.
    # ------------------------------------------------------------------
    if fit == "double_cut":
        result_cut_break = pl_fit.cut_break_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            E_cut=E_cut,
            exponent=exponent,
            print_report=False,
            maxit=maxit,
        )
        redchi_cut_break = result_cut_break.res_var
        breakp_cut = result_cut_break.beta[4]

        # The cut of the break + cutoff.
        cut_b = result_cut_break.beta[5]
        difference_cut = breakp_cut - cut_b

        if (
            breakp_cut <= emax
            and breakp_cut > emin
            and cut_b <= emax
            and cut_b > emin
        ):
            low, high = _break_energy_interval(
                spec_e,
                e_err,
                breakp_cut,
            )
            difference_cut_energy = high - low

            if (
                cut_b > breakp_cut
                and difference_cut > difference_cut_energy
            ):
                return [
                    "double_cut",
                    redchi_cut_break,
                    result_cut_break,
                ]

            else:
                fit = "best_cb"

        else:
            fit = "best_cb"

    # ------------------------------------------------------------------
    # Select between a single and double power law.
    # ------------------------------------------------------------------
    if fit == "best_sb":
        result_single_pl = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
        )
        redchi_single = result_single_pl.res_var

        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
        )
        redchi_double = result_double.res_var
        breakp = result_double.beta[4]

        if redchi_double <= redchi_single:
            if breakp < emin or breakp > emax:
                return ["single", redchi_single, result_single_pl]

            if breakp >= emin and breakp <= emax:
                return ["double", redchi_double, result_double]

        if redchi_double > redchi_single:
            return ["single", redchi_single, result_single_pl]

    # ------------------------------------------------------------------
    # Select between a cutoff and double power law.
    # ------------------------------------------------------------------
    if fit == "best_cb":
        result_cut = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            E_cut=E_cut,
            exponent=exponent,
            maxit=maxit,
        )
        redchi_cut = result_cut.res_var

        # Should maybe make distinction between cut from cut PL
        # and cut from cut double PL.
        cut = result_cut.beta[2]

        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
        )
        redchi_double = result_double.res_var
        breakp = result_double.beta[4]

        if redchi_double <= redchi_cut:
            if breakp < emin or breakp > emax:
                fit = "single"

            if breakp >= emin and breakp <= emax:
                return ["double", redchi_double, result_double]

        if redchi_double > redchi_cut:
            if cut < emin or cut > emax:
                fit = "single"

            if cut >= emin and cut <= emax:
                return ["cut", redchi_cut, result_cut]

    # ------------------------------------------------------------------
    # Explicit cutoff fit.
    # ------------------------------------------------------------------
    if fit == "cut":
        result_cut = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            E_cut=E_cut,
            exponent=exponent,
            maxit=maxit,
        )
        redchi_cut = result_cut.res_var

        # Should maybe make distinction between cut from cut PL
        # and cut from cut double PL.
        cut = result_cut.beta[2]

        if cut < emin or cut > emax:
            fit = "single"

        if cut >= emin and cut <= emax:
            return ["cut", redchi_cut, result_cut]

    # ------------------------------------------------------------------
    # Explicit double power-law fit.
    # ------------------------------------------------------------------
    if fit == "double":
        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
        )
        redchi_double = result_double.res_var
        breakp = result_double.beta[4]

        if breakp < emin or breakp > emax:
            fit = "single"

        if breakp >= emin and breakp <= emax:
            return ["double", redchi_double, result_double]

    # ------------------------------------------------------------------
    # Single power-law fallback.
    # ------------------------------------------------------------------
    if fit == "single":
        result_single_pl = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
        )
        redchi_single = result_single_pl.res_var

        return ["single", redchi_single, result_single_pl]        
    
    

def MAKE_THE_FIT_old(
    spec_e,
    spec_flux,
    e_err,
    flux_err,
    ax,
    direction="sun",
    which_fit="best",
    e_min=None,
    e_max=None,
    g1_guess=-2.0,
    g2_guess=None,
    g3_guess=None,
    alpha_guess=5.0,
    beta_guess=5,
    break_low_guess=0.065,
    break_high_guess=0.12,
    cut_guess=0.12,
    c1_guess=None,
    exponent_guess=2,
    use_random=False,
    iterations=10,
    path=None,
    path2=None,
    detailed_legend=False,
):
    """Fit the data to a single, double, or break+cut power law.

    The fit type can be chosen between: single, double, cut, or best.
    The best option checks between all available options and chooses
    between them by comparing the reduced chi-squared values.

    When the double or cut options are chosen, the function also checks
    whether the fitted break or cut-off points are outside the energy
    range. In such cases, a single power law is fitted instead.
    """

    # CHANGE GUESS VALUES OF GAMMA1
    if g2_guess is None:
        g2_guess = g1_guess - 0.1

    if g3_guess is None:
        g3_guess = g2_guess - 0.1

    if e_min is None:
        # e_min = min(spec_e)
        e_min = spec_e[0]

    if e_max is None:
        # e_max = max(spec_e)
        e_max = spec_e[-1]

    if c1_guess is None:
        absolute_val_array = np.abs(spec_e - 1)
        smallest_difference_index = absolute_val_array.argmin()
        c1_guess = spec_flux[smallest_difference_index]

    # The break guess should be between min and max energy.

    # Have to construct the guesses logarithmically.
    g1_start_value = -np.abs(g1_guess) * 10.0
    g2_start_value = -np.abs(g2_guess) * 10.0
    g3_start_value = -np.abs(g3_guess) * 10.0

    g1_end_value = np.abs(g1_guess) * 10.0
    g2_end_value = np.abs(g2_guess) * 10.0
    g3_end_value = np.abs(g3_guess) * 10.0

    g1_step = np.abs(g1_guess / 4.0)
    g2_step = np.abs(g2_guess / 4.0)
    g3_step = np.abs(g3_guess / 4.0)

    if use_random:
        gamma1_array = closest_values(
            np.arange(g1_start_value, g1_end_value, g1_step),
            g1_guess,
        )
        gamma2_array = closest_values(
            np.arange(g2_start_value, g2_end_value, g2_step),
            g2_guess,
        )
        gamma3_array = closest_values(
            np.arange(g3_start_value, g3_end_value, g3_step),
            g3_guess,
        )

        # c1_array... we want to get a good approximation of the flux
        # at 1, whatever 1 is in your plot.
        c1_array = np.arange(
            c1_guess / 100.0,
            c1_guess * 100.0,
            c1_guess / 500.0,
        )

        # alpha array
        a1_array = np.arange(0.01, 0.1, 0.01)
        a2_array = np.arange(0.1, 1.0, 0.05)
        a3_array = np.arange(1, 10, 0.5)
        a4_array = np.arange(10, 100, 10)
        a5_array = np.arange(100, 220, 20)

        alpha_array = np.hstack((a1_array, a2_array, a3_array, a4_array, a5_array))
        alpha_array = closest_values(alpha_array, alpha_guess)

        beta_array = np.hstack((a1_array, a2_array, a3_array, a4_array, a5_array))
        beta_array = closest_values(beta_array, beta_guess)

        # break array
        # cut array = break_array * 1.8

        if e_max < 0.1:
            break_array_low = np.arange(e_min, e_max, 0.001)

        if e_max >= 0.1 and e_max < 1.0:
            b1_array = np.arange(e_min, 0.1, 0.001)
            b2_array = np.arange(0.1, e_max, 0.005)
            break_array_low = np.hstack((b1_array, b2_array))

        if e_max >= 1 and e_max < 10:
            b1_array = np.arange(e_min, 0.1, 0.001)
            b2_array = np.arange(0.1, 1, 0.005)
            b3_array = np.arange(1, e_max, 0.01)
            break_array_low = np.hstack((b1_array, b2_array, b3_array))

        if e_max >= 10:
            b1_array = np.arange(e_min, 0.1, 0.001)
            b2_array = np.arange(0.1, 1, 0.005)
            b3_array = np.arange(1, 10, 0.01)
            b4_array = np.arange(10, e_max, 1)
            break_array_low = np.hstack((b1_array, b2_array, b3_array, b4_array))

        break_array_high = break_array_low[1:]
        cut_array = break_array_low[1:]

        break_array_low = closest_values(break_array_low, break_low_guess)
        break_array_high = closest_values(break_array_high, break_high_guess)
        cut_array = closest_values(cut_array, cut_guess)

    # print(c1_guess)

    color = {
        "sun": "crimson",
        "asun": "orange",
        "north": "darkslateblue",
        "south": "c",
    }

    spec_e = np.array(spec_e)
    spec_flux = np.array(spec_flux)
    e_err = np.array(e_err)
    flux_err = np.array(flux_err)

    xplot = np.logspace(
        np.log10(np.nanmin(spec_e)),
        np.log10(np.nanmax(spec_e)),
        num=500,
    )
    xplot = xplot[
        np.where((xplot >= e_min) & (xplot <= e_max))[0]
    ]

    fit_ind = np.where(
        (spec_e >= e_min)
        & (spec_e <= e_max)
        & np.isfinite(spec_flux)
        & np.isfinite(flux_err)
    )[0]

    spec_e = spec_e[fit_ind]
    spec_flux = spec_flux[fit_ind]
    e_err = e_err[fit_ind]
    flux_err = flux_err[fit_ind]

    # Everything is in a for loop that chooses random values between the
    # closest_values (n times) and checks the redchis and chooses the best one.
    # Try separately the input guesses and then the random ones.

    # Everything is done first with input guess values and then with randoms.

    # Parameters used as final inputs !!!
    # (not the fit result but input)!!!

    which_fit_final = ""

    redchi_final = 0

    result_final = None
    convergence = True

# spec_e, spec_flux, e_err, flux_err, gamma1, gamma2, gamma3, c1, alpha, beta, E_break_low, E_break_high,  E_cut= None, fit = 'best',  maxit=10000, e_min=None, e_max=None):
    if which_fit == "best":
        # First check the redchi and if the break is outside of the
        # energy range using the guess values, then compare the random
        # values to these.
        # If redchi is better, substitute values.

        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            beta=beta_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            gamma3=g3_guess,
            E_break_low=break_low_guess,
            E_break_high=break_high_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="best",
            maxit=10000,
            e_min=e_min,
            e_max=e_max,
        )

        redchi_guess = which_fit_guess[1]

        redchi_final = redchi_guess
        which_fit_final = which_fit_guess[0]
        result_final = which_fit_guess[2]
        convergence = pl_fit.check_odr_output(result_final)

        if use_random:
            # print("USING RANDOM BEST")
            for i in range(iterations):
                # Need [0] because the result is an array.
                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]
                g3_random = np.random.choice(gamma3_array, 1)[0]

                # POSSIBLE ISSUES
                gammas = [g1_random, g2_random, g3_random]
                gammas.sort()
                g1_random = gammas[0]
                g2_random = gammas[1]
                g3_random = gammas[2]

                alpha_random = np.random.choice(alpha_array, 1)[0]
                beta_random = np.random.choice(beta_array, 1)[0]

                break_low_random = np.random.choice(
                    break_array_low, 1
                )[0]
                break_high_random = np.random.choice(
                    break_array_high, 1
                )[0]

                if break_high_random < break_low_random:
                    b = break_low_random
                    break_low_random = break_high_random
                    break_high_random = b

                cut_random = np.random.choice(cut_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    beta=beta_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    gamma3=g3_random,
                    E_break_low=break_low_random,
                    E_break_high=break_high_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    maxit=10000,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    break

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]
                convergence = pl_fit.check_odr_output(result_random)

                if redchi_random < redchi_final and convergence:
                    result_final = which_fit_random[2]
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]



    if which_fit == "triple":
        # First check the redchi and if the break is outside of the
        # energy range using the guess values, then compare the random
        # values to these.
        # If redchi is better, substitute values.

        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            beta=beta_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            gamma3=g3_guess,
            E_break_low=break_low_guess,
            E_break_high=break_high_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="triple",
            maxit=10000,
            e_min=e_min,
            e_max=e_max,
        )

        # If for some reason the fit is not doable, the result will be None.
        # In that case you cannot use redchi_guess = which_fit_guess[1]
        # because you cannot call a None value.
        #
        # This was previously handled by repeating the fit in a while loop,
        # but the current behavior is to handle None through the result
        # checking below.

        redchi_guess = which_fit_guess[1]

        redchi_final = redchi_guess
        which_fit_final = which_fit_guess[0]
        result_final = which_fit_guess[2]

        if use_random:
            for i in range(iterations):
                # Need [0] because the result is an array.
                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0] 
                g3_random = np.random.choice(gamma3_array, 1)[0]
                #removed sorting because gamma2 can be less negative fpr AK triple for example

                alpha_random = np.random.choice(alpha_array, 1)[0]
                beta_random = np.random.choice(beta_array, 1)[0]

                break_low_random = np.random.choice(
                    break_array_low, 1
                )[0]
                break_high_random = np.random.choice(
                    break_array_high, 1
                )[0]

                if break_high_random < break_low_random:
                    b = break_low_random
                    break_low_random = break_high_random
                    break_high_random = b

                cut_random = np.random.choice(cut_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    beta=beta_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    gamma3=g3_random,
                    E_break_low=break_low_random,
                    E_break_high=break_high_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    fit="triple",
                    maxit=10000,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    break

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]
                convergence = pl_fit.check_odr_output(result_random)

                if redchi_random < redchi_final and convergence:
                    result_final = which_fit_random[2]
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    if which_fit == "best_cb":
        # First check the redchi and if the break is outside of the energy
        # range using the guess values, then compare the random values to these.
        # If redchi is better, substitute values.
        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            E_break_low=break_low_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="best_cb",
            maxit=10000,
            e_min=e_min,
            e_max=e_max,
        )

        redchi_guess = which_fit_guess[1]
        redchi_final = redchi_guess
        which_fit_final = which_fit_guess[0]
        result_final = which_fit_guess[2]

        if use_random:
            for i in range(iterations):
                # Need [0] because np.random.choice returns an array here.
                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]

                alpha_random = np.random.choice(alpha_array, 1)[0]
                break_low_random = np.random.choice(break_array_low, 1)[0]
                cut_random = np.random.choice(cut_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    E_break_low=break_low_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    fit="best_cb",
                    maxit=10000,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    break

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]
                convergence = pl_fit.check_odr_output(result_random)

                if redchi_random < redchi_final and convergence:
                    result_final = which_fit_random[2]

                    if which_fit_random[0] in ("single", "double", "cut"):
                        redchi_final = redchi_random
                        which_fit_final = which_fit_random[0]
                        
    if which_fit == "best_sb":
    # First check the redchi and if the break is outside of the energy
    # range using the guess values, then compare the random values to these.
    # If redchi is better, substitute values.
        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            E_break_low=break_low_guess,
            fit="best_sb",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
        )

        redchi_guess = which_fit_guess[1]
        redchi_final = redchi_guess
        which_fit_final = which_fit_guess[0]
        result_final = which_fit_guess[2]

        if use_random:
            for i in range(iterations):
                # Need [0] because np.random.choice returns an array here.
                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]

                alpha_random = np.random.choice(alpha_array, 1)[0]
                break_low_random = np.random.choice(break_array_low, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    E_break_low=break_low_random,
                    fit="best_sb",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    break

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]
                convergence = pl_fit.check_odr_output(result_random)

                if redchi_random < redchi_final and convergence:
                    result_final = which_fit_random[2]

                    if which_fit_random[0] in ("single", "double"):
                        redchi_final = redchi_random
                        which_fit_final = which_fit_random[0]
                        
    
    if which_fit == "double_cut":
        result_cut_guess = pl_fit.cut_break_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=g1_guess,
            gamma2=g2_guess,
            c1=c1_guess,
            alpha=alpha_guess,
            E_break=break_low_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            print_report=False,
            maxit=maxit,
        )

        breakp_cut = result_cut_guess.beta[4]
        cut_b = result_cut_guess.beta[5]

        if breakp_cut < e_min or breakp_cut > e_max:
            print("The break point is outside of the energy range")

            which_fit_guess = check_redchi(
                spec_e,
                spec_flux,
                e_err,
                flux_err,
                c1=c1_guess,
                alpha=alpha_guess,
                gamma1=g1_guess,
                gamma2=g2_guess,
                E_break_low=break_low_guess,
                E_cut=cut_guess,
                exponent=exponent_guess,
                fit="best_cb",
                maxit=maxit,
                e_min=e_min,
                e_max=e_max,
            )

            redchi_guess = which_fit_guess[1]
            redchi_final = redchi_guess
            which_fit_final = which_fit_guess[0]
            result_final = which_fit_guess[2]

        if breakp_cut >= e_min and breakp_cut <= e_max:
            if cut_b <= e_min or cut_b >= e_max:
                # The breaks are checked by redchi.
                which_fit_guess = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_guess,
                    alpha=alpha_guess,
                    beta=beta_guess,
                    gamma1=g1_guess,
                    gamma2=g2_guess,
                    E_break_low=break_low_guess,
                    E_cut=cut_b,
                    exponent=exponent_guess,
                    fit="double_cut",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                )

                redchi_guess = which_fit_guess[1]
                redchi_final = redchi_guess
                which_fit_final = which_fit_guess[0]
                result_final = which_fit_guess[2]

            if cut_b > e_min and cut_b < e_max:
                which_fit_final = "double_cut"
                result_final = result_cut_guess
                redchi_guess = result_cut_guess.res_var
                redchi_final = redchi_guess

        if use_random:
            for i in range(iterations):
                # Need [0] because np.random.choice returns an array here.
                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]

                alpha_random = np.random.choice(alpha_array, 1)[0]
                break_low_random = np.random.choice(break_array_low, 1)[0]
                cut_random = np.random.choice(cut_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    E_break_low=break_low_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    fit="double_cut",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    break

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]
                convergence = pl_fit.check_odr_output(result_random)

                if redchi_random < redchi_final and convergence:
                    result_final = which_fit_random[2]

                    if which_fit_random[0] in (
                        "single",
                        "double",
                        "cut",
                        "double_cut",
                    ):
                        redchi_final = redchi_random
                        which_fit_final = which_fit_random[0]
                                        
    
    if which_fit == "double":
    # Even if which_fit is double, we need to check first if the break
    # point is outside of the energy range. In that case, we have to
    # change it to single.
        result_double_guess = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=g1_guess,
            gamma2=g2_guess,
            c1=c1_guess,
            alpha=alpha_guess,
            E_break=break_low_guess,
            maxit=maxit,
        )
        breakp_1 = result_double_guess.beta[4]

        if breakp_1 < e_min or breakp_1 > e_max:
            print("The break point is outside of the energy range")

            which_fit_final = "single"
            result_single_pl_guess = pl_fit.power_law_fit(
                x=spec_e,
                y=spec_flux,
                xerr=e_err,
                yerr=flux_err,
                gamma1=g1_guess,
                c1=c1_guess,
            )
            result_final = result_single_pl_guess
            redchi_guess = result_single_pl_guess.res_var
            redchi_final = redchi_guess

        if breakp_1 >= e_min and breakp_1 <= e_max:
            which_fit_final = "double"
            result_final = result_double_guess
            redchi_guess = result_double_guess.res_var
            redchi_final = redchi_guess

        if use_random:
            for i in range(iterations):
                # Need [0] because np.random.choice returns an array here.
                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]

                alpha_random = np.random.choice(alpha_array, 1)[0]
                break_low_random = np.random.choice(break_array_low, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                result_double_random = pl_fit.double_pl_fit(
                    x=spec_e,
                    y=spec_flux,
                    xerr=e_err,
                    yerr=flux_err,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    c1=c1_random,
                    alpha=alpha_random,
                    E_break=break_low_random,
                    maxit=maxit,
                )
                breakp_1 = result_double_random.beta[4]

                convergence_double = pl_fit.check_odr_output(
                    result_double_random
                )

                if breakp_1 < e_min or breakp_1 > e_max:
                    result_single_pl_random = pl_fit.power_law_fit(
                        x=spec_e,
                        y=spec_flux,
                        xerr=e_err,
                        yerr=flux_err,
                        gamma1=g1_random,
                        c1=c1_random,
                    )
                    redchi_random = result_single_pl_random.res_var
                    convergence_single = pl_fit.check_odr_output(
                        result_single_pl_random
                    )

                    if redchi_random < redchi_final and convergence_single:
                        which_fit_final = "single"
                        redchi_final = redchi_random
                        result_final = result_single_pl_random

                if breakp_1 >= e_min and breakp_1 <= e_max and convergence_double:
                    redchi_random = result_double_random.res_var

                    if redchi_random < redchi_final:
                        which_fit_final = "double"
                        redchi_final = redchi_random
                        result_final = result_double_random                    

    if which_fit == 'cut':
        # Check first whether the cutoff point is outside the energy range.
        # In that case, fall back to a single power law.
        result_cut_guess = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=g1_guess,
            c1=c1_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            maxit=maxit,
        )
        cut = result_cut_guess.beta[2]

        if cut < e_min or cut > e_max:
            print('The cutoff point is outside of the energy range')

            which_fit_final = 'single'
            result_single_pl_guess = pl_fit.power_law_fit(
                x=spec_e,
                y=spec_flux,
                xerr=e_err,
                yerr=flux_err,
                gamma1=g1_guess,
                c1=c1_guess,
            )
            result_final = result_single_pl_guess
            redchi_final = result_single_pl_guess.res_var

        elif e_min <= cut <= e_max:
            which_fit_final = 'cut'
            result_final = result_cut_guess
            redchi_final = result_cut_guess.res_var

        if use_random:
            for i in range(iterations):
                # Draw random initial guesses.
                g1_random = np.random.choice(gamma1_array, 1)[0]
                cut_random = np.random.choice(cut_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                result_cut_random = pl_fit.cut_pl_fit(
                    x=spec_e,
                    y=spec_flux,
                    xerr=e_err,
                    yerr=flux_err,
                    gamma1=g1_random,
                    c1=c1_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    maxit=maxit,
                )
                cut = result_cut_random.beta[2]

                convergence_cut = pl_fit.check_odr_output(
                    result_cut_random
                )

                if cut < e_min or cut > e_max:
                    result_single_pl_random = pl_fit.power_law_fit(
                        x=spec_e,
                        y=spec_flux,
                        xerr=e_err,
                        yerr=flux_err,
                        gamma1=g1_random,
                        c1=c1_random,
                    )
                    redchi_random = result_single_pl_random.res_var
                    convergence_single = pl_fit.check_odr_output(
                        result_single_pl_random
                    )

                    if redchi_random < redchi_final and convergence_single:
                        which_fit_final = 'single'
                        redchi_final = redchi_random
                        result_final = result_single_pl_random

                elif e_min <= cut <= e_max and convergence_cut:
                    redchi_random = result_cut_random.res_var

                    if redchi_random < redchi_final:
                        which_fit_final = 'cut'
                        redchi_final = redchi_random
                        result_final = result_cut_random
    
    if which_fit == 'single':
        which_fit_final = 'single'

        result_single_pl_guess = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=g1_guess,
            c1=c1_guess,
        )
        result_final = result_single_pl_guess
        redchi_final = result_single_pl_guess.res_var

        if use_random:
            for i in range(iterations):
                # Draw random initial guesses.
                g1_random = np.random.choice(gamma1_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                result_single_pl_random = pl_fit.power_law_fit(
                    x=spec_e,
                    y=spec_flux,
                    xerr=e_err,
                    yerr=flux_err,
                    gamma1=g1_random,
                    c1=c1_random,
                )
                redchi_random = result_single_pl_random.res_var
                convergence_single = pl_fit.check_odr_output(
                    result_single_pl_random
                )

                if redchi_random < redchi_final and convergence_single:
                    redchi_final = redchi_random
                    result_final = result_single_pl_random


        result_dataframe = pd.DataFrame(
        {"Final fit type": which_fit_final},
        index=[0],
    )

    result = result_final

    if which_fit_final == 'single':
        result_single_pl = result_final

        redchi_single = result_single_pl.res_var
        c1 = result_single_pl.beta[0]
        gamma1 = result_single_pl.beta[1]

        dof = len(spec_e) - len(result_single_pl.beta)
        t_val = studentt.interval(0.95, dof)[1]
        errors = t_val * result_single_pl.sd_beta
        gamma1_err = errors[1]

        if detailed_legend:
            ax.plot([], [], ' ', label="Single pl")
            ax.plot(
                [],
                [],
                ' ',
                label=r'$\mathregular{\chi²=}$%5.2f'
                % round(redchi_single, ndigits=2),
            )
            ax.plot(
                [],
                [],
                ' ',
                label=r'$\mathregular{I_0=}$'
                + "{:.2e}".format(c1)
                + "/(s cm² sr MeV)",
            )

        ax.plot(
            xplot,
            pl_fit.simple_pl([c1, gamma1], xplot),
            '-',
            color=color[direction],
            label=(
                r'$\mathregular{\gamma=}$%5.2f'
                % round(gamma1, ndigits=2)
                + r"$\pm$"
                + '{0:.2f}'.format(gamma1_err)
            ),
        )
        ax.plot(
            xplot,
            pl_fit.simple_pl([c1, gamma1], xplot),
            '--k',
            zorder=10,
        )

        result_dataframe["Reduced chi sq"] = redchi_single
        result_dataframe["c1"] = c1
        result_dataframe["c1 err"] = errors[0]
        result_dataframe["Gamma1"] = gamma1
        result_dataframe["Gamma1 err"] = gamma1_err
        result_dataframe["Gamma2"] = None
        result_dataframe["Gamma2 err"] = None
        result_dataframe["Gamma3"] = None
        result_dataframe["Gamma3 err"] = None
        result_dataframe["Break point 1 [MeV]"] = None
        result_dataframe["Break point 1 err [MeV]"] = None
        result_dataframe["Break point 2 [MeV]"] = None
        result_dataframe["Break point 2 err [MeV]"] = None
        result_dataframe["Exponential cutoff point [MeV]"] = None
        result_dataframe["Cutoff err [MeV]"] = None
        result_dataframe["Alpha"] = None
        result_dataframe["Beta"] = None
        result_dataframe["Exponent"] = None

    if which_fit_final == 'double':
        result_double = result_final
        result = result_final

        breakp_1 = result_double.beta[4]
        alpha = result_double.beta[3]
        c1 = result_double.beta[0]

        dof = len(spec_e) - len(result_double.beta)
        redchi_double = result_double.res_var

        t_val = studentt.interval(0.95, dof)[1]
        errors = t_val * result_double.sd_beta

        breakp_1_err = errors[4]

        # The meaning of beta[1] and beta[2] depends on the sign of alpha.
        if alpha > 0:
            gamma1 = result_double.beta[1]
            gamma1_err = errors[1]
            gamma2 = result_double.beta[2]
            gamma2_err = errors[2]

        elif alpha < 0:
            gamma1 = result_double.beta[2]
            gamma1_err = errors[2]
            gamma2 = result_double.beta[1]
            gamma2_err = errors[1]

        # Ensure that the sign of alpha is consistent with the ordering
        # of the two power-law indices.
        if gamma1 < gamma2 and alpha > 0:
            alpha = -abs(alpha)

        elif gamma2 < gamma1 and alpha < 0:
            alpha = abs(alpha)

        fit_plot = pl_fit.double_pl_func(result_double.beta, xplot)
        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:
            ax.plot([], [], ' ', label="Broken pl")
            ax.plot(
                [],
                [],
                ' ',
                label=r'$\mathregular{\chi²=}$%5.2f'
                % round(redchi_double, ndigits=2),
            )
            ax.plot(
                [],
                [],
                ' ',
                label=(
                    r'$\mathregular{I_0=}$'
                    + "{:.2e}".format(c1)
                    + "/(s cm² sr MeV)"
                ),
            )

        ax.plot(
            xplot,
            fit_plot,
            '-b',
            label=(
                r'$\mathregular{\gamma_1=}$%5.2f'
                % round(gamma1, ndigits=2)
                + r"$\pm$"
                + '{0:.2f}'.format(gamma1_err)
                + '\n'
                + r'$\mathregular{\gamma_2=}$%5.2f'
                % round(gamma2, ndigits=2)
                + r"$\pm$"
                + '{0:.2f}'.format(gamma2_err)
                + '\n'
                + r'$\mathregular{\alpha=}$%5.2f'
                % round(alpha, ndigits=2)
            ),
        )

        if len(str(breakp_1 * 1e3).split('.')[0]) > 3:
            ax.axvline(
                x=breakp_1,
                color='blue',
                linestyle='--',
                label=(
                    r'$\mathregular{E_b=}$ '
                    + str(round(breakp_1, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(breakp_1_err, ndigits=1))
                    + ' MeV'
                ),
            )
        else:
            ax.axvline(
                x=breakp_1,
                color='blue',
                linestyle='--',
                label=(
                    r'$\mathregular{E_b=}$ '
                    + str(round(breakp_1 * 1e3, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(breakp_1_err * 1e3, ndigits=1))
                    + ' keV'
                ),
            )

        result_dataframe["Reduced chi sq"] = redchi_double
        result_dataframe["c1"] = c1
        result_dataframe["c1 err"] = errors[0]
        result_dataframe["Gamma1"] = gamma1
        result_dataframe["Gamma1 err"] = gamma1_err
        result_dataframe["Gamma2"] = gamma2
        result_dataframe["Gamma2 err"] = gamma2_err
        result_dataframe["Gamma3"] = None
        result_dataframe["Gamma3 err"] = None
        result_dataframe["Break point 1 [MeV]"] = breakp_1
        result_dataframe["Break point 1 err [MeV]"] = breakp_1_err
        result_dataframe["Break point 2 [MeV]"] = None
        result_dataframe["Break point 2 err [MeV]"] = None
        result_dataframe["Exponential cutoff point [MeV]"] = None
        result_dataframe["Cutoff err [MeV]"] = None
        result_dataframe["Alpha"] = alpha
        result_dataframe["Beta"] = None
        result_dataframe["Exponent"] = None


    if which_fit_final == 'cut':
        result_cut = result_final

        cut = result_cut.beta[2]
        dof = len(spec_e) - len(result_cut.beta)
        redchi_cut = result_cut.res_var

        t_val = studentt.interval(0.95, dof)[1]
        errors = t_val * result_cut.sd_beta

        c1 = result_cut.beta[0]
        gamma1 = result_cut.beta[1]
        gamma1_err = errors[1]
        cut_err = errors[2]
        exponent = result_cut.beta[3]

        fit_plot = pl_fit.cut_pl_func(result_cut.beta, xplot)
        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:
            ax.plot([], [], ' ', label="Single pl + exp cutoff")
            ax.plot([], [], ' ', label="exponent: " + str(round(exponent, ndigits=2)))
            ax.plot(
                [],
                [],
                ' ',
                label=r'$\mathregular{\chi²=}$%5.2f' % round(redchi_cut, ndigits=2)
            )
            ax.plot(
                [],
                [],
                ' ',
                label=r'$\mathregular{I_0=}$' + "{:.2e}".format(c1)
                + "/(s cm² sr MeV)"
            )

        ax.plot(
            xplot,
            fit_plot,
            '-b',
            label=(
                r'$\mathregular{\gamma_1=}$%5.2f'
                % round(gamma1, ndigits=2)
                + r"$\pm$"
                + '{0:.2f}'.format(gamma1_err)
            )
        )

        if len(str(cut * 1e3).split('.')[0]) > 3:
            ax.axvline(
                x=cut,
                color='purple',
                linestyle='--',
                label=(
                    r'$\mathregular{E_c=}$ '
                    + str(round(cut, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(cut_err, ndigits=1))
                    + ' MeV'
                )
            )
        else:
            ax.axvline(
                x=cut,
                color='purple',
                linestyle='--',
                label=(
                    r'$\mathregular{E_c=}$ '
                    + str(round(cut * 1e3, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(cut_err * 1e3, ndigits=1))
                    + ' keV'
                )
            )

        result_dataframe["Reduced chi sq"] = redchi_cut
        result_dataframe["c1"] = c1
        result_dataframe["c1 err"] = errors[0]
        result_dataframe["Gamma1"] = gamma1
        result_dataframe["Gamma1 err"] = gamma1_err
        result_dataframe["Gamma2"] = None
        result_dataframe["Gamma2 err"] = None
        result_dataframe["Gamma3"] = None
        result_dataframe["Gamma3 err"] = None
        result_dataframe["Break point 1 [MeV]"] = None
        result_dataframe["Break point 1 err [MeV]"] = None
        result_dataframe["Break point 2 [MeV]"] = None
        result_dataframe["Break point 2 err [MeV]"] = None
        result_dataframe["Exponential cutoff point [MeV]"] = cut
        result_dataframe["Cutoff err [MeV]"] = cut_err
        result_dataframe["Alpha"] = None
        result_dataframe["Beta"] = None
        result_dataframe["Exponent"] = exponent


    if which_fit_final == 'double_cut':
        result_cut = result_final

        cut = result_cut.beta[5]
        breakp_1 = result_cut.beta[4]
        alpha = result_cut.beta[3]

        dof = len(spec_e) - len(result_cut.beta)
        redchi_cut = result_cut.res_var

        t_val = studentt.interval(0.95, dof)[1]
        errors = t_val * result_cut.sd_beta

        breakp_1_err = errors[4]
        cut_err = errors[5]

        c1 = result_cut.beta[0]
        exponent = result_cut.beta[6]

        # Keep the existing alpha-dependent gamma assignment.
        if alpha > 0:
            gamma1 = result_cut.beta[1]
            gamma1_err = errors[1]
            gamma2 = result_cut.beta[2]
            gamma2_err = errors[2]

        if alpha < 0:
            gamma1 = result_cut.beta[2]
            gamma1_err = errors[2]
            gamma2 = result_cut.beta[1]
            gamma2_err = errors[1]

        # Preserve the existing sign convention for alpha.
        if gamma1 < gamma2 and alpha > 0:
            alpha = -abs(alpha)
        elif gamma2 < gamma1 and alpha < 0:
            alpha = abs(alpha)

        fit_plot = pl_fit.cut_break_pl_func(result_cut.beta, xplot)
        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:
            ax.plot([], [], ' ', label="Broken pl + exp cutoff")
            ax.plot(
                [],
                [],
                ' ',
                label="exponent: " + str(round(exponent, ndigits=2))
            )
            ax.plot(
                [],
                [],
                ' ',
                label=r'$\mathregular{\chi²=}$%5.2f'
                % round(redchi_cut, ndigits=2)
            )
            ax.plot(
                [],
                [],
                ' ',
                label=r'$\mathregular{I_0=}$' + "{:.2e}".format(c1)
                + "/(s cm² sr MeV)"
            )

        ax.plot(
            xplot,
            fit_plot,
            '-b',
            label=(
                r'$\mathregular{\gamma_1=}$%5.2f'
                % round(gamma1, ndigits=2)
                + r"$\pm$"
                + '{0:.2f}'.format(gamma1_err)
                + '\n'
                + r'$\mathregular{\gamma_2=}$%5.2f'
                % round(gamma2, ndigits=2)
                + r"$\pm$"
                + '{0:.2f}'.format(gamma2_err)
                + '\n'
                + r'$\mathregular{\alpha=}$%5.2f'
                % round(alpha, ndigits=2)
            )
        )

        if len(str(breakp_1 * 1e3).split('.')[0]) > 3:
            ax.axvline(
                x=breakp_1,
                color='blue',
                linestyle='--',
                label=(
                    r'$\mathregular{E_b=}$ '
                    + str(round(breakp_1, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(breakp_1_err, ndigits=1))
                    + ' MeV'
                )
            )
        else:
            ax.axvline(
                x=breakp_1,
                color='blue',
                linestyle='--',
                label=(
                    r'$\mathregular{E_b=}$ '
                    + str(round(breakp_1 * 1e3, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(breakp_1_err * 1e3, ndigits=1))
                    + ' keV'
                )
            )

        if len(str(cut * 1e3).split('.')[0]) > 3:
            ax.axvline(
                x=cut,
                color='purple',
                linestyle='--',
                label=(
                    r'$\mathregular{E_c=}$ '
                    + str(round(cut, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(cut_err, ndigits=1))
                    + ' MeV'
                )
            )
        else:
            ax.axvline(
                x=cut,
                color='purple',
                linestyle='--',
                label=(
                    r'$\mathregular{E_c=}$ '
                    + str(round(cut * 1e3, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(cut_err * 1e3, ndigits=1))
                    + ' keV'
                )
            )

        result_dataframe["Reduced chi sq"] = redchi_cut
        result_dataframe["c1"] = c1
        result_dataframe["c1 err"] = errors[0]
        result_dataframe["Gamma1"] = gamma1
        result_dataframe["Gamma1 err"] = gamma1_err
        result_dataframe["Gamma2"] = gamma2
        result_dataframe["Gamma2 err"] = gamma2_err
        result_dataframe["Gamma3"] = None
        result_dataframe["Gamma3 err"] = None
        result_dataframe["Break point 1 [MeV]"] = breakp_1
        result_dataframe["Break point 1 err [MeV]"] = breakp_1_err
        result_dataframe["Break point 2 [MeV]"] = None
        result_dataframe["Break point 2 err [MeV]"] = None
        result_dataframe["Exponential cutoff point [MeV]"] = cut
        result_dataframe["Cutoff err [MeV]"] = cut_err
        result_dataframe["Alpha"] = alpha
        result_dataframe["Beta"] = None
        result_dataframe["Exponent"] = exponent


    if which_fit_final == 'triple':
        result_triple = result_final

        breakp_2 = result_triple.beta[7]
        breakp_1 = result_triple.beta[6]
        alpha = result_triple.beta[4]
        beta = result_triple.beta[5]

        dof = len(spec_e) - len(result_triple.beta)
        redchi_triple = result_triple.res_var

        t_val = studentt.interval(0.95, dof)[1]
        errors = t_val * result_triple.sd_beta

        breakp_1_err = errors[6]
        breakp_2_err = errors[7]

        c1 = result_triple.beta[0]

        # The gamma assignment depends on the signs of alpha and beta.
        # Keep this logic unchanged because it defines the physical
        # ordering/convention of the triple-power-law parameters.
        if alpha > 0 and beta > 0:
            gamma1 = result_triple.beta[1]
            gamma1_err = errors[1]
            gamma2 = result_triple.beta[2]
            gamma2_err = errors[2]
            gamma3 = result_triple.beta[3]
            gamma3_err = errors[3]

        if alpha < 0 and beta > 0:
            gamma1 = result_triple.beta[2]
            gamma1_err = errors[2]
            gamma2 = result_triple.beta[1]
            gamma2_err = errors[1]
            gamma3 = result_triple.beta[3]
            gamma3_err = errors[3]

        if beta < 0 and alpha > 0:
            gamma1 = result_triple.beta[1]
            gamma1_err = errors[1]
            gamma2 = result_triple.beta[3]
            gamma2_err = errors[3]
            gamma3 = result_triple.beta[2]
            gamma3_err = errors[2]

        if alpha < 0 and beta < 0:
            gamma1 = result_triple.beta[3]
            gamma1_err = errors[3]
            gamma2 = result_triple.beta[2]
            gamma2_err = errors[2]
            gamma3 = result_triple.beta[1]
            gamma3_err = errors[1]

        # Preserve the existing sign convention for alpha and beta.
        if gamma1 > gamma2 and gamma2 > gamma3:
            if alpha < 0 and beta > 0:
                alpha = abs(alpha)

            if alpha > 0 and beta < 0:
                beta = abs(beta)

            if alpha < 0 and beta < 0:
                alpha = abs(alpha)
                beta = abs(beta)

        if gamma1 > gamma2 and gamma2 < gamma3:
            if alpha > 0 and beta > 0:
                beta = -abs(beta)

            if alpha < 0 and beta > 0:
                a = alpha
                b = beta
                alpha = b
                beta = a

            if alpha < 0 and beta < 0:
                alpha = abs(alpha)

        if gamma1 < gamma2 and gamma2 > gamma3:
            if alpha > 0 and beta > 0:
                alpha = -abs(alpha)

            if alpha > 0 and beta < 0:
                a = alpha
                b = beta
                alpha = b
                beta = a

            if alpha < 0 and beta < 0:
                beta = -abs(beta)

        fit_plot = pl_fit.triple_pl_func(result_triple.beta, xplot)
        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:
            ax.plot([], [], ' ', label="Triple pl")
            ax.plot(
                [],
                [],
                ' ',
                label=r'$\mathregular{\chi²=}$%5.2f'
                % round(redchi_triple, ndigits=2)
            )
            ax.plot(
                [],
                [],
                ' ',
                label=r'$\mathregular{I_0=}$' + "{:.2e}".format(c1)
                + "/(s cm² sr MeV)"
            )

        ax.plot(
            xplot,
            fit_plot,
            '-b',
            label=(
                r'$\mathregular{\gamma_1=}$%5.2f'
                % round(gamma1, ndigits=2)
                + r"$\pm$"
                + '{0:.2f}'.format(gamma1_err)
                + '\n'
                + r'$\mathregular{\gamma_2=}$%5.2f'
                % round(gamma2, ndigits=2)
                + r"$\pm$"
                + '{0:.2f}'.format(gamma2_err)
                + '\n'
                + r'$\mathregular{\gamma_3=}$%5.2f'
                % round(gamma3, ndigits=2)
                + r"$\pm$"
                + '{0:.2f}'.format(gamma3_err)
                + '\n'
                + r'$\mathregular{\alpha=}$%5.2f'
                % round(alpha, ndigits=2)
                + '\n'
                + r'$\mathregular{\beta=}$%5.2f'
                % round(beta, ndigits=2)
            )
        )

        if len(str(breakp_1 * 1e3).split('.')[0]) > 3:
            ax.axvline(
                x=breakp_1,
                color='blue',
                linestyle='--',
                label=(
                    r'$\mathregular{E_b1=}$ '
                    + str(round(breakp_1, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(breakp_1_err, ndigits=1))
                    + ' MeV'
                )
            )
        else:
            ax.axvline(
                x=breakp_1,
                color='blue',
                linestyle='--',
                label=(
                    r'$\mathregular{E_b1=}$ '
                    + str(round(breakp_1 * 1e3, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(breakp_1_err * 1e3, ndigits=1))
                    + ' keV'
                )
            )

        if len(str(breakp_2 * 1e3).split('.')[0]) > 3:
            ax.axvline(
                x=breakp_2,
                color='purple',
                linestyle='--',
                label=(
                    r'$\mathregular{E_b2=}$ '
                    + str(round(breakp_2, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(breakp_2_err, ndigits=1))
                    + ' MeV'
                )
            )
        else:
            ax.axvline(
                x=breakp_2,
                color='purple',
                linestyle='--',
                label=(
                    r'$\mathregular{E_b2=}$ '
                    + str(round(breakp_2 * 1e3, ndigits=1))
                    + '\n'
                    + r"$\pm$"
                    + str(round(breakp_2_err * 1e3, ndigits=1))
                    + ' keV'
                )
            )

        result_dataframe["Reduced chi sq"] = redchi_triple
        result_dataframe["c1"] = c1
        result_dataframe["c1 err"] = errors[0]
        result_dataframe["Gamma1"] = gamma1
        result_dataframe["Gamma1 err"] = gamma1_err
        result_dataframe["Gamma2"] = gamma2
        result_dataframe["Gamma2 err"] = gamma2_err
        result_dataframe["Gamma3"] = gamma3
        result_dataframe["Gamma3 err"] = gamma3_err
        result_dataframe["Break point 1 [MeV]"] = breakp_1
        result_dataframe["Break point 1 err [MeV]"] = breakp_1_err
        result_dataframe["Break point 2 [MeV]"] = breakp_2
        result_dataframe["Break point 2 err [MeV]"] = breakp_2_err
        result_dataframe["Exponential cutoff point [MeV]"] = None
        result_dataframe["Cutoff err [MeV]"] = None
        result_dataframe["Alpha"] = alpha
        result_dataframe["Beta"] = beta
        result_dataframe["Exponent"] = None

    result_dataframe["E min [MeV]"] = e_min
    result_dataframe["E max [MeV]"] = e_max

    # Save result to pickle file.
    if path is not None:
        with open(path, 'wb') as f:
            pickle.dump(result, f)

    # Save fitting variables.
    if path2 is not None:
        result_dataframe.to_csv(path2, sep=";")

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


import pickle
import numpy as np
import pandas as pd

from scipy.stats import t as studentt


# ============================================================================
# Helpers
# ============================================================================

def _get_fit_errors(result):
    """
    Return the 95% confidence-interval-scaled ODR parameter errors.

    This preserves the original behavior:
        errors = student_t * result.sd_beta
    """
    dof = result.y.size - len(result.beta)

    t_val = studentt.interval(0.95, dof)[1]
    return t_val * result.sd_beta


def _update_result_dataframe(
    result_dataframe,
    *,
    redchi,
    c1,
    c1_err,
    gamma1,
    gamma1_err,
    gamma2=None,
    gamma2_err=None,
    gamma3=None,
    gamma3_err=None,
    breakp_1=None,
    breakp_1_err=None,
    breakp_2=None,
    breakp_2_err=None,
    cutoff=None,
    cutoff_err=None,
    alpha=None,
    beta=None,
    exponent=None,
    e_min=None,
    e_max=None,
):
    """
    Store the final reported/physical fit parameters in result_dataframe.

    IMPORTANT:
    The gamma values passed here are the reordered/reporting convention,
    not necessarily result.beta[1:4].
    """

    result_dataframe["Reduced chi sq"] = redchi

    result_dataframe["c1"] = c1
    result_dataframe["c1 err"] = c1_err

    result_dataframe["Gamma1"] = gamma1
    result_dataframe["Gamma1 err"] = gamma1_err

    result_dataframe["Gamma2"] = gamma2
    result_dataframe["Gamma2 err"] = gamma2_err

    result_dataframe["Gamma3"] = gamma3
    result_dataframe["Gamma3 err"] = gamma3_err

    result_dataframe["Break point 1 [MeV]"] = breakp_1
    result_dataframe["Break point 1 err [MeV]"] = breakp_1_err

    result_dataframe["Break point 2 [MeV]"] = breakp_2
    result_dataframe["Break point 2 err [MeV]"] = breakp_2_err

    result_dataframe["Exponential cutoff point [MeV]"] = cutoff
    result_dataframe["Cutoff err [MeV]"] = cutoff_err

    result_dataframe["Alpha"] = alpha
    result_dataframe["Beta"] = beta
    result_dataframe["Exponent"] = exponent

    result_dataframe["E min [MeV]"] = e_min
    result_dataframe["E max [MeV]"] = e_max


def _plot_energy_marker(
    ax,
    energy,
    error,
    *,
    color,
    linestyle="--",
    label_prefix="E_b=",
):
    """
    Plot an energy marker in MeV or keV.

    The original code chose the unit according to whether the energy
    was below or above 1 MeV. This helper preserves that behavior.
    """

    if energy >= 1:
        value = energy
        value_err = error
        unit = "MeV"
    else:
        value = energy * 1e3
        value_err = error * 1e3
        unit = "keV"

    ax.axvline(
        x=energy,
        color=color,
        linestyle=linestyle,
        label=(
            rf"$\mathregular{{{label_prefix}}}$ "
            + f"{value:.1f}"
            + "\n"
            + r"$\pm$"
            + f"{value_err:.1f}"
            + f" {unit}"
        ),
    )


def _reorder_double_gammas(
    raw_gamma1,
    raw_gamma2,
    raw_gamma1_err,
    raw_gamma2_err,
    alpha,
):
    """
    Convert the raw mathematical-function gamma parameters into the
    physical/plot-order Gamma1 and Gamma2 convention.

    IMPORTANT:
    raw_gamma1/raw_gamma2 refer to the parameters in result.beta.
    They do NOT necessarily correspond to the first/second spectral
    power law in the plotted spectrum.

    For a double power law, the correspondence depends on alpha.
    """

    if alpha > 0:
        gamma1 = raw_gamma1
        gamma1_err = raw_gamma1_err

        gamma2 = raw_gamma2
        gamma2_err = raw_gamma2_err

    else:
        gamma1 = raw_gamma2
        gamma1_err = raw_gamma2_err

        gamma2 = raw_gamma1
        gamma2_err = raw_gamma1_err

    return (
        gamma1,
        gamma1_err,
        gamma2,
        gamma2_err,
    )


def _reorder_triple_gammas(
    raw_gamma1,
    raw_gamma2,
    raw_gamma3,
    raw_gamma1_err,
    raw_gamma2_err,
    raw_gamma3_err,
    alpha,
    beta,
):
    """
    Convert the raw ODR gamma parameters into the physical/plot-order
    Gamma1, Gamma2 and Gamma3 convention.

    IMPORTANT
    ---------
    The triple convolution can exchange the spectral components depending
    on the signs of alpha and beta.

    Therefore:

        result.beta[1] -> raw_gamma1
        result.beta[2] -> raw_gamma2
        result.beta[3] -> raw_gamma3

    are the gamma parameters of the MATHEMATICAL FUNCTION, whereas
    Gamma1/Gamma2/Gamma3 returned here are the physical/plot-order
    spectral indices.

    The plotted function must therefore continue to use:

        pl_fit.triple_pl_func(result.beta, xplot)

    and NOT the reordered Gamma1/Gamma2/Gamma3.
    """

    if alpha > 0 and beta > 0:

        gamma1 = raw_gamma1
        gamma1_err = raw_gamma1_err

        gamma2 = raw_gamma2
        gamma2_err = raw_gamma2_err

        gamma3 = raw_gamma3
        gamma3_err = raw_gamma3_err

    elif alpha < 0 and beta > 0:

        gamma1 = raw_gamma2
        gamma1_err = raw_gamma2_err

        gamma2 = raw_gamma1
        gamma2_err = raw_gamma1_err

        gamma3 = raw_gamma3
        gamma3_err = raw_gamma3_err

    elif beta < 0 and alpha > 0:

        gamma1 = raw_gamma1
        gamma1_err = raw_gamma1_err

        gamma2 = raw_gamma3
        gamma2_err = raw_gamma3_err

        gamma3 = raw_gamma2
        gamma3_err = raw_gamma2_err

    elif alpha < 0 and beta < 0:

        gamma1 = raw_gamma3
        gamma1_err = raw_gamma3_err

        gamma2 = raw_gamma2
        gamma2_err = raw_gamma2_err

        gamma3 = raw_gamma1
        gamma3_err = raw_gamma1_err

    else:
        raise ValueError(
            "alpha and beta cannot be zero when determining "
            "the triple-power-law gamma ordering."
        )

    return (
        gamma1,
        gamma1_err,
        gamma2,
        gamma2_err,
        gamma3,
        gamma3_err,
    )


def _apply_triple_sign_convention(
    gamma1,
    gamma2,
    gamma3,
    alpha,
    beta,
):
    """
    Preserve the existing alpha/beta sign convention used by the code.

    This function ONLY changes the reported alpha/beta convention.

    It does NOT modify the raw ODR parameter vector and therefore does
    not modify the plotted mathematical function.
    """

    if gamma1 > gamma2 and gamma2 > gamma3:

        if alpha < 0 and beta > 0:
            alpha = abs(alpha)

        if alpha > 0 and beta < 0:
            beta = abs(beta)

        if alpha < 0 and beta < 0:
            alpha = abs(alpha)
            beta = abs(beta)

    elif gamma1 > gamma2 and gamma2 < gamma3:

        if alpha > 0 and beta > 0:
            beta = -abs(beta)

        if alpha < 0 and beta > 0:
            a = alpha
            b = beta
            alpha = b
            beta = a

        if alpha < 0 and beta < 0:
            alpha = abs(alpha)

    elif gamma1 < gamma2 and gamma2 > gamma3:

        if alpha > 0 and beta > 0:
            alpha = -abs(alpha)

        if alpha > 0 and beta < 0:
            a = alpha
            b = beta
            alpha = b
            beta = a

        if alpha < 0 and beta < 0:
            beta = -abs(beta)

    return alpha, beta


def _is_valid_fit(result):
    """Check whether an ODR result exists and reports successful convergence."""
    if result is None:
        return False

    return pl_fit.check_odr_output(result)


def _make_result_dataframe(which_fit_final):
    """Create the one-row result DataFrame."""
    return pd.DataFrame(
        {"Final fit type": which_fit_final},
        index=[0],
    )

def MAKE_THE_FIT(
    spec_e,
    spec_flux,
    e_err,
    flux_err,
    ax,
    direction="sun",
    which_fit="best",
    e_min=None,
    e_max=None,
    g1_guess=-2.0,
    g2_guess=None,
    g3_guess=None,
    alpha_guess=5.0,
    beta_guess=5,
    break_low_guess=0.065,
    break_high_guess=0.12,
    cut_guess=0.12,
    c1_guess=None,
    exponent_guess=2,
    use_random=False,
    iterations=10,
    maxit=10000,
    path=None,
    path2=None,
    detailed_legend=False,
):
    """
    Fit the data to a single, double, triple, cutoff, or
    broken-power-law + cutoff model.

    The mathematical ODR parameters and the reported physical parameters
    are deliberately kept separate.

    In particular, for the triple model:

        result_final.beta[1:4]

    are the raw gamma parameters of the mathematical convolution.

    The reported Gamma1/Gamma2/Gamma3 values are reordered according
    to alpha and beta so that they correspond to the physical/plot-order
    spectral components.

    The plotted curve ALWAYS uses the raw ODR parameter vector.
    """

    # ========================================================================
    # 1. Prepare guesses
    # ========================================================================

    if g2_guess is None:
        g2_guess = g1_guess - 0.1

    if g3_guess is None:
        g3_guess = g2_guess - 0.1

    spec_e = np.asarray(spec_e)
    spec_flux = np.asarray(spec_flux)
    e_err = np.asarray(e_err)
    flux_err = np.asarray(flux_err)

    if e_min is None:
        e_min = spec_e[0]

    if e_max is None:
        e_max = spec_e[-1]

    if c1_guess is None:
        absolute_val_array = np.abs(spec_e - 1)
        smallest_difference_index = absolute_val_array.argmin()
        c1_guess = spec_flux[smallest_difference_index]

    # ========================================================================
    # 2. Generate random-search arrays
    # ========================================================================

    g1_start_value = -np.abs(g1_guess) * 10.0
    g2_start_value = -np.abs(g2_guess) * 10.0
    g3_start_value = -np.abs(g3_guess) * 10.0

    g1_end_value = np.abs(g1_guess) * 10.0
    g2_end_value = np.abs(g2_guess) * 10.0
    g3_end_value = np.abs(g3_guess) * 10.0

    g1_step = np.abs(g1_guess / 4.0)
    g2_step = np.abs(g2_guess / 4.0)
    g3_step = np.abs(g3_guess / 4.0)

    if use_random:

        gamma1_array = closest_values(
            np.arange(g1_start_value, g1_end_value, g1_step),
            g1_guess,
        )

        gamma2_array = closest_values(
            np.arange(g2_start_value, g2_end_value, g2_step),
            g2_guess,
        )

        gamma3_array = closest_values(
            np.arange(g3_start_value, g3_end_value, g3_step),
            g3_guess,
        )

        c1_array = np.arange(
            c1_guess / 100.0,
            c1_guess * 100.0,
            c1_guess / 500.0,
        )

        a1_array = np.arange(0.01, 0.1, 0.01)
        a2_array = np.arange(0.1, 1.0, 0.05)
        a3_array = np.arange(1, 10, 0.5)
        a4_array = np.arange(10, 100, 10)
        a5_array = np.arange(100, 220, 20)

        alpha_array = np.hstack(
            (a1_array, a2_array, a3_array, a4_array, a5_array)
        )
        alpha_array = closest_values(alpha_array, alpha_guess)

        beta_array = np.hstack(
            (a1_array, a2_array, a3_array, a4_array, a5_array)
        )
        beta_array = closest_values(beta_array, beta_guess)

        if e_max < 0.1:

            break_array_low = np.arange(
                e_min,
                e_max,
                0.001,
            )

        elif e_max < 1.0:

            b1_array = np.arange(e_min, 0.1, 0.001)
            b2_array = np.arange(0.1, e_max, 0.005)

            break_array_low = np.hstack(
                (b1_array, b2_array)
            )

        elif e_max < 10:

            b1_array = np.arange(e_min, 0.1, 0.001)
            b2_array = np.arange(0.1, 1, 0.005)
            b3_array = np.arange(1, e_max, 0.01)

            break_array_low = np.hstack(
                (b1_array, b2_array, b3_array)
            )

        else:

            b1_array = np.arange(e_min, 0.1, 0.001)
            b2_array = np.arange(0.1, 1, 0.005)
            b3_array = np.arange(1, 10, 0.01)
            b4_array = np.arange(10, e_max, 1)

            break_array_low = np.hstack(
                (b1_array, b2_array, b3_array, b4_array)
            )

        break_array_high = break_array_low[1:]
        cut_array = break_array_low[1:]

        break_array_low = closest_values(
            break_array_low,
            break_low_guess,
        )

        break_array_high = closest_values(
            break_array_high,
            break_high_guess,
        )

        cut_array = closest_values(
            cut_array,
            cut_guess,
        )

    # ========================================================================
    # 3. Plot/data preparation
    # ========================================================================

    color = {
        "sun": "crimson",
        "asun": "orange",
        "north": "darkslateblue",
        "south": "c",
    }

    xplot = np.logspace(
        np.log10(np.nanmin(spec_e)),
        np.log10(np.nanmax(spec_e)),
        num=500,
    )

    xplot = xplot[
        np.where(
            (xplot >= e_min)
            & (xplot <= e_max)
        )[0]
    ]

    fit_ind = np.where(
        (spec_e >= e_min)
        & (spec_e <= e_max)
        & np.isfinite(spec_flux)
        & np.isfinite(flux_err)
    )[0]

    spec_e = spec_e[fit_ind]
    spec_flux = spec_flux[fit_ind]
    e_err = e_err[fit_ind]
    flux_err = flux_err[fit_ind]

    # ========================================================================
    # 4. Variables used to select the final fit
    # ========================================================================

    which_fit_final = ""
    redchi_final = np.inf
    result_final = None

    # ========================================================================
    # 5. BEST
    # ========================================================================

    if which_fit == "best":

        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            beta=beta_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            gamma3=g3_guess,
            E_break_low=break_low_guess,
            E_break_high=break_high_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="best",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
        )

        if which_fit_guess is not None:

            redchi_final = which_fit_guess[1]
            which_fit_final = which_fit_guess[0]
            result_final = which_fit_guess[2]

        if use_random:

            for i in range(iterations):

                # Keep [0] deliberately.
                g1_random = np.random.choice(
                    gamma1_array, 1
                )[0]

                g2_random = np.random.choice(
                    gamma2_array, 1
                )[0]

                g3_random = np.random.choice(
                    gamma3_array, 1
                )[0]

                alpha_random = np.random.choice(
                    alpha_array, 1
                )[0]

                beta_random = np.random.choice(
                    beta_array, 1
                )[0]

                break_low_random = np.random.choice(
                    break_array_low, 1
                )[0]

                break_high_random = np.random.choice(
                    break_array_high, 1
                )[0]

                if break_high_random < break_low_random:

                    b = break_low_random
                    break_low_random = break_high_random
                    break_high_random = b

                cut_random = np.random.choice(
                    cut_array, 1
                )[0]

                c1_random = np.random.choice(
                    c1_array, 1
                )[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    beta=beta_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    gamma3=g3_random,
                    E_break_low=break_low_random,
                    E_break_high=break_high_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                convergence = _is_valid_fit(result_random)

                if (
                    redchi_random < redchi_final
                    and convergence
                ):
                    result_final = result_random
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    # ========================================================================
    # 6. TRIPLE
    # ========================================================================

    if which_fit == "triple":

        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            beta=beta_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            gamma3=g3_guess,
            E_break_low=break_low_guess,
            E_break_high=break_high_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="triple",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
        )

        if which_fit_guess is not None:

            redchi_final = which_fit_guess[1]
            which_fit_final = which_fit_guess[0]
            result_final = which_fit_guess[2]

        if use_random:

            for i in range(iterations):

                # Keep these exactly in the explicit form requested.
                g1_random = np.random.choice(
                    gamma1_array, 1
                )[0]

                g2_random = np.random.choice(
                    gamma2_array, 1
                )[0]

                g3_random = np.random.choice(
                    gamma3_array, 1
                )[0]

                alpha_random = np.random.choice(
                    alpha_array, 1
                )[0]

                beta_random = np.random.choice(
                    beta_array, 1
                )[0]

                break_low_random = np.random.choice(
                    break_array_low, 1
                )[0]

                break_high_random = np.random.choice(
                    break_array_high, 1
                )[0]

                if break_high_random < break_low_random:

                    b = break_low_random
                    break_low_random = break_high_random
                    break_high_random = b

                cut_random = np.random.choice(
                    cut_array, 1
                )[0]

                c1_random = np.random.choice(
                    c1_array, 1
                )[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    beta=beta_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    gamma3=g3_random,
                    E_break_low=break_low_random,
                    E_break_high=break_high_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    fit="triple",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                convergence = _is_valid_fit(result_random)

                if (
                    redchi_random < redchi_final
                    and convergence
                ):
                    result_final = result_random
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    # ========================================================================
    # 7. BEST_CB
    # ========================================================================

    if which_fit == "best_cb":

        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            E_break_low=break_low_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="best_cb",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
        )

        if which_fit_guess is not None:

            redchi_final = which_fit_guess[1]
            which_fit_final = which_fit_guess[0]
            result_final = which_fit_guess[2]

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(
                    gamma1_array, 1
                )[0]

                g2_random = np.random.choice(
                    gamma2_array, 1
                )[0]

                alpha_random = np.random.choice(
                    alpha_array, 1
                )[0]

                break_low_random = np.random.choice(
                    break_array_low, 1
                )[0]

                cut_random = np.random.choice(
                    cut_array, 1
                )[0]

                c1_random = np.random.choice(
                    c1_array, 1
                )[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    E_break_low=break_low_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    fit="best_cb",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                convergence = _is_valid_fit(result_random)

                if (
                    redchi_random < redchi_final
                    and convergence
                    and which_fit_random[0]
                    in ("single", "double", "cut")
                ):
                    result_final = result_random
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    # ========================================================================
    # 8. BEST_SB
    # ========================================================================

    if which_fit == "best_sb":

        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            E_break_low=break_low_guess,
            fit="best_sb",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
        )

        if which_fit_guess is not None:

            redchi_final = which_fit_guess[1]
            which_fit_final = which_fit_guess[0]
            result_final = which_fit_guess[2]

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(
                    gamma1_array, 1
                )[0]

                g2_random = np.random.choice(
                    gamma2_array, 1
                )[0]

                alpha_random = np.random.choice(
                    alpha_array, 1
                )[0]

                break_low_random = np.random.choice(
                    break_array_low, 1
                )[0]

                c1_random = np.random.choice(
                    c1_array, 1
                )[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    E_break_low=break_low_random,
                    fit="best_sb",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                convergence = _is_valid_fit(result_random)

                if (
                    redchi_random < redchi_final
                    and convergence
                    and which_fit_random[0]
                    in ("single", "double")
                ):
                    result_final = result_random
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    # ========================================================================
    # 9. DOUBLE_CUT
    # ========================================================================

    if which_fit == "double_cut":

        result_cut_guess = pl_fit.cut_break_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=g1_guess,
            gamma2=g2_guess,
            c1=c1_guess,
            alpha=alpha_guess,
            E_break=break_low_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            print_report=False,
            maxit=maxit,
        )

        breakp_cut = result_cut_guess.beta[4]
        cut_b = result_cut_guess.beta[5]

        if breakp_cut < e_min or breakp_cut > e_max:

            print(
                "The break point is outside of the energy range"
            )

            which_fit_guess = check_redchi(
                spec_e,
                spec_flux,
                e_err,
                flux_err,
                c1=c1_guess,
                alpha=alpha_guess,
                beta=beta_guess,
                gamma1=g1_guess,
                gamma2=g2_guess,
                E_break_low=break_low_guess,
                E_cut=cut_guess,
                exponent=exponent_guess,
                fit="best_cb",
                maxit=maxit,
                e_min=e_min,
                e_max=e_max,
            )

            if which_fit_guess is not None:
                redchi_final = which_fit_guess[1]
                which_fit_final = which_fit_guess[0]
                result_final = which_fit_guess[2]

        elif e_min <= breakp_cut <= e_max:

            if cut_b <= e_min or cut_b >= e_max:

                which_fit_guess = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_guess,
                    alpha=alpha_guess,
                    beta=beta_guess,
                    gamma1=g1_guess,
                    gamma2=g2_guess,
                    E_break_low=break_low_guess,
                    E_cut=cut_b,
                    exponent=exponent_guess,
                    fit="double_cut",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_guess is not None:
                    redchi_final = which_fit_guess[1]
                    which_fit_final = which_fit_guess[0]
                    result_final = which_fit_guess[2]

            elif e_min < cut_b < e_max:

                which_fit_final = "double_cut"
                result_final = result_cut_guess
                redchi_final = result_cut_guess.res_var

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(
                    gamma1_array, 1
                )[0]

                g2_random = np.random.choice(
                    gamma2_array, 1
                )[0]

                alpha_random = np.random.choice(
                    alpha_array, 1
                )[0]

                break_low_random = np.random.choice(
                    break_array_low, 1
                )[0]

                cut_random = np.random.choice(
                    cut_array, 1
                )[0]

                c1_random = np.random.choice(
                    c1_array, 1
                )[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    E_break_low=break_low_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    fit="double_cut",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                convergence = _is_valid_fit(result_random)

                if (
                    redchi_random < redchi_final
                    and convergence
                    and which_fit_random[0]
                    in (
                        "single",
                        "double",
                        "cut",
                        "double_cut",
                    )
                ):
                    result_final = result_random
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    # ========================================================================
    # 10. DOUBLE
    # ========================================================================

    if which_fit == "double":

        result_double_guess = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=g1_guess,
            gamma2=g2_guess,
            c1=c1_guess,
            alpha=alpha_guess,
            E_break=break_low_guess,
            maxit=maxit,
        )

        breakp_1 = result_double_guess.beta[4]

        if breakp_1 < e_min or breakp_1 > e_max:

            print(
                "The break point is outside of the energy range"
            )

            which_fit_final = "single"

            result_final = pl_fit.power_law_fit(
                x=spec_e,
                y=spec_flux,
                xerr=e_err,
                yerr=flux_err,
                gamma1=g1_guess,
                c1=c1_guess,
            )

            redchi_final = result_final.res_var

        else:

            which_fit_final = "double"
            result_final = result_double_guess
            redchi_final = result_double_guess.res_var

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(
                    gamma1_array, 1
                )[0]

                g2_random = np.random.choice(
                    gamma2_array, 1
                )[0]

                alpha_random = np.random.choice(
                    alpha_array, 1
                )[0]

                break_low_random = np.random.choice(
                    break_array_low, 1
                )[0]

                c1_random = np.random.choice(
                    c1_array, 1
                )[0]

                result_double_random = pl_fit.double_pl_fit(
                    x=spec_e,
                    y=spec_flux,
                    xerr=e_err,
                    yerr=flux_err,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    c1=c1_random,
                    alpha=alpha_random,
                    E_break=break_low_random,
                    maxit=maxit,
                )

                breakp_1 = result_double_random.beta[4]

                convergence_double = _is_valid_fit(
                    result_double_random
                )

                if breakp_1 < e_min or breakp_1 > e_max:

                    result_single_pl_random = pl_fit.power_law_fit(
                        x=spec_e,
                        y=spec_flux,
                        xerr=e_err,
                        yerr=flux_err,
                        gamma1=g1_random,
                        c1=c1_random,
                    )

                    redchi_random = (
                        result_single_pl_random.res_var
                    )

                    convergence_single = _is_valid_fit(
                        result_single_pl_random
                    )

                    if (
                        redchi_random < redchi_final
                        and convergence_single
                    ):
                        which_fit_final = "single"
                        redchi_final = redchi_random
                        result_final = result_single_pl_random

                elif (
                    breakp_1 >= e_min
                    and breakp_1 <= e_max
                    and convergence_double
                ):

                    redchi_random = (
                        result_double_random.res_var
                    )

                    if redchi_random < redchi_final:

                        which_fit_final = "double"
                        redchi_final = redchi_random
                        result_final = result_double_random

    # ========================================================================
    # 11. CUT
    # ========================================================================

    if which_fit == "cut":

        result_cut_guess = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=g1_guess,
            c1=c1_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            maxit=maxit,
        )

        cut = result_cut_guess.beta[2]

        if cut < e_min or cut > e_max:

            print(
                "The cutoff point is outside of the energy range"
            )

            which_fit_final = "single"

            result_final = pl_fit.power_law_fit(
                x=spec_e,
                y=spec_flux,
                xerr=e_err,
                yerr=flux_err,
                gamma1=g1_guess,
                c1=c1_guess,
            )

            redchi_final = result_final.res_var

        else:

            which_fit_final = "cut"
            result_final = result_cut_guess
            redchi_final = result_cut_guess.res_var

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(
                    gamma1_array, 1
                )[0]

                cut_random = np.random.choice(
                    cut_array, 1
                )[0]

                c1_random = np.random.choice(
                    c1_array, 1
                )[0]

                result_cut_random = pl_fit.cut_pl_fit(
                    x=spec_e,
                    y=spec_flux,
                    xerr=e_err,
                    yerr=flux_err,
                    gamma1=g1_random,
                    c1=c1_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    maxit=maxit,
                )

                cut = result_cut_random.beta[2]

                convergence_cut = _is_valid_fit(
                    result_cut_random
                )

                if cut < e_min or cut > e_max:

                    result_single_pl_random = (
                        pl_fit.power_law_fit(
                            x=spec_e,
                            y=spec_flux,
                            xerr=e_err,
                            yerr=flux_err,
                            gamma1=g1_random,
                            c1=c1_random,
                        )
                    )

                    redchi_random = (
                        result_single_pl_random.res_var
                    )

                    convergence_single = _is_valid_fit(
                        result_single_pl_random
                    )

                    if (
                        redchi_random < redchi_final
                        and convergence_single
                    ):
                        which_fit_final = "single"
                        redchi_final = redchi_random
                        result_final = result_single_pl_random

                elif (
                    e_min <= cut <= e_max
                    and convergence_cut
                ):

                    redchi_random = (
                        result_cut_random.res_var
                    )

                    if redchi_random < redchi_final:

                        which_fit_final = "cut"
                        redchi_final = redchi_random
                        result_final = result_cut_random

    # ========================================================================
    # 12. SINGLE
    # ========================================================================

    if which_fit == "single":

        which_fit_final = "single"

        result_final = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=g1_guess,
            c1=c1_guess,
        )

        redchi_final = result_final.res_var

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(
                    gamma1_array, 1
                )[0]

                c1_random = np.random.choice(
                    c1_array, 1
                )[0]

                result_single_pl_random = (
                    pl_fit.power_law_fit(
                        x=spec_e,
                        y=spec_flux,
                        xerr=e_err,
                        yerr=flux_err,
                        gamma1=g1_random,
                        c1=c1_random,
                    )
                )

                redchi_random = (
                    result_single_pl_random.res_var
                )

                convergence_single = _is_valid_fit(
                    result_single_pl_random
                )

                if (
                    redchi_random < redchi_final
                    and convergence_single
                ):
                    redchi_final = redchi_random
                    result_final = result_single_pl_random

    # ========================================================================
    # 13. Check that something was actually fitted
    # ========================================================================

    if result_final is None:

        raise RuntimeError(
            "No valid fit was obtained for "
            f"which_fit='{which_fit}'."
        )

    result = result_final

    result_dataframe = _make_result_dataframe(
        which_fit_final
    )

    # ========================================================================
    # 14. SINGLE RESULT
    # ========================================================================

    if which_fit_final == "single":

        result_single_pl = result_final

        redchi_single = result_single_pl.res_var

        c1 = result_single_pl.beta[0]
        gamma1 = result_single_pl.beta[1]

        errors = _get_fit_errors(result_single_pl)

        gamma1_err = errors[1]

        if detailed_legend:

            ax.plot(
                [],
                [],
                " ",
                label="Single pl",
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    r"$\mathregular{\chi²=}$"
                    f"{redchi_single:.2f}"
                ),
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    r"$\mathregular{I_0=}$"
                    + f"{c1:.2e}"
                    + "/(s cm² sr MeV)"
                ),
            )

        fit_plot = pl_fit.simple_pl(
            [c1, gamma1],
            xplot,
        )

        ax.plot(
            xplot,
            fit_plot,
            "-",
            color=color[direction],
            label=(
                r"$\mathregular{\gamma=}$"
                f"{gamma1:.2f}"
                r"$\pm$"
                f"{gamma1_err:.2f}"
            ),
        )

        ax.plot(
            xplot,
            fit_plot,
            "--k",
            zorder=10,
        )

        _update_result_dataframe(
            result_dataframe,
            redchi=redchi_single,
            c1=c1,
            c1_err=errors[0],
            gamma1=gamma1,
            gamma1_err=gamma1_err,
            e_min=e_min,
            e_max=e_max,
        )

    # ========================================================================
    # 15. DOUBLE RESULT
    # ========================================================================

    elif which_fit_final == "double":

        result_double = result_final

        breakp_1 = result_double.beta[4]
        alpha = result_double.beta[3]
        c1 = result_double.beta[0]

        redchi_double = result_double.res_var

        errors = _get_fit_errors(result_double)

        breakp_1_err = errors[4]

        # ------------------------------------------------------------
        # Raw mathematical gamma parameters.
        #
        # These are NOT necessarily the first and second physical
        # power-law segments in the plotted spectrum.
        # ------------------------------------------------------------

        raw_gamma1 = result_double.beta[1]
        raw_gamma2 = result_double.beta[2]

        raw_gamma1_err = errors[1]
        raw_gamma2_err = errors[2]

        (
            gamma1,
            gamma1_err,
            gamma2,
            gamma2_err,
        ) = _reorder_double_gammas(
            raw_gamma1,
            raw_gamma2,
            raw_gamma1_err,
            raw_gamma2_err,
            alpha,
        )

        # Preserve the existing alpha convention.
        if gamma1 < gamma2 and alpha > 0:
            alpha = -abs(alpha)

        elif gamma2 < gamma1 and alpha < 0:
            alpha = abs(alpha)

        # IMPORTANT:
        # The plot uses the ORIGINAL ODR parameter vector.
        fit_plot = pl_fit.double_pl_func(
            result_double.beta,
            xplot,
        )

        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:

            ax.plot(
                [],
                [],
                " ",
                label="Broken pl",
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    r"$\mathregular{\chi²=}$"
                    f"{redchi_double:.2f}"
                ),
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    r"$\mathregular{I_0=}$"
                    + f"{c1:.2e}"
                    + "/(s cm² sr MeV)"
                ),
            )

        ax.plot(
            xplot,
            fit_plot,
            "-b",
            label=(
                r"$\mathregular{\gamma_1=}$"
                f"{gamma1:.2f}"
                r"$\pm$"
                f"{gamma1_err:.2f}"
                "\n"
                r"$\mathregular{\gamma_2=}$"
                f"{gamma2:.2f}"
                r"$\pm$"
                f"{gamma2_err:.2f}"
                "\n"
                r"$\mathregular{\alpha=}$"
                f"{alpha:.2f}"
            ),
        )

        _plot_energy_marker(
            ax,
            breakp_1,
            breakp_1_err,
            color="blue",
            label_prefix="E_b=",
        )

        _update_result_dataframe(
            result_dataframe,
            redchi=redchi_double,
            c1=c1,
            c1_err=errors[0],
            gamma1=gamma1,
            gamma1_err=gamma1_err,
            gamma2=gamma2,
            gamma2_err=gamma2_err,
            breakp_1=breakp_1,
            breakp_1_err=breakp_1_err,
            alpha=alpha,
            e_min=e_min,
            e_max=e_max,
        )

    # ========================================================================
    # 16. CUT RESULT
    # ========================================================================

    elif which_fit_final == "cut":

        result_cut = result_final

        cut = result_cut.beta[2]

        redchi_cut = result_cut.res_var

        errors = _get_fit_errors(result_cut)

        c1 = result_cut.beta[0]
        gamma1 = result_cut.beta[1]

        gamma1_err = errors[1]
        cut_err = errors[2]
        exponent = result_cut.beta[3]

        fit_plot = pl_fit.cut_pl_func(
            result_cut.beta,
            xplot,
        )

        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:

            ax.plot(
                [],
                [],
                " ",
                label="Single pl + exp cutoff",
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    "exponent: "
                    f"{exponent:.2f}"
                ),
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    r"$\mathregular{\chi²=}$"
                    f"{redchi_cut:.2f}"
                ),
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    r"$\mathregular{I_0=}$"
                    + f"{c1:.2e}"
                    + "/(s cm² sr MeV)"
                ),
            )

        ax.plot(
            xplot,
            fit_plot,
            "-b",
            label=(
                r"$\mathregular{\gamma_1=}$"
                f"{gamma1:.2f}"
                r"$\pm$"
                f"{gamma1_err:.2f}"
            ),
        )

        _plot_energy_marker(
            ax,
            cut,
            cut_err,
            color="purple",
            label_prefix="E_c=",
        )

        _update_result_dataframe(
            result_dataframe,
            redchi=redchi_cut,
            c1=c1,
            c1_err=errors[0],
            gamma1=gamma1,
            gamma1_err=gamma1_err,
            cutoff=cut,
            cutoff_err=cut_err,
            exponent=exponent,
            e_min=e_min,
            e_max=e_max,
        )

    # ========================================================================
    # 17. DOUBLE + CUTOFF RESULT
    # ========================================================================

    elif which_fit_final == "double_cut":

        result_cut = result_final

        breakp_1 = result_cut.beta[4]
        cut = result_cut.beta[5]
        alpha = result_cut.beta[3]

        redchi_cut = result_cut.res_var

        errors = _get_fit_errors(result_cut)

        breakp_1_err = errors[4]
        cut_err = errors[5]

        c1 = result_cut.beta[0]
        exponent = result_cut.beta[6]

        # ------------------------------------------------------------
        # Raw mathematical gamma parameters.
        # ------------------------------------------------------------

        raw_gamma1 = result_cut.beta[1]
        raw_gamma2 = result_cut.beta[2]

        raw_gamma1_err = errors[1]
        raw_gamma2_err = errors[2]

        (
            gamma1,
            gamma1_err,
            gamma2,
            gamma2_err,
        ) = _reorder_double_gammas(
            raw_gamma1,
            raw_gamma2,
            raw_gamma1_err,
            raw_gamma2_err,
            alpha,
        )

        # Preserve existing alpha sign convention.
        if gamma1 < gamma2 and alpha > 0:
            alpha = -abs(alpha)

        elif gamma2 < gamma1 and alpha < 0:
            alpha = abs(alpha)

        # IMPORTANT:
        # Use raw ODR parameters for the mathematical function.
        fit_plot = pl_fit.cut_break_pl_func(
            result_cut.beta,
            xplot,
        )

        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:

            ax.plot(
                [],
                [],
                " ",
                label="Broken pl + exp cutoff",
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    "exponent: "
                    f"{exponent:.2f}"
                ),
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    r"$\mathregular{\chi²=}$"
                    f"{redchi_cut:.2f}"
                ),
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    r"$\mathregular{I_0=}$"
                    + f"{c1:.2e}"
                    + "/(s cm² sr MeV)"
                ),
            )

        ax.plot(
            xplot,
            fit_plot,
            "-b",
            label=(
                r"$\mathregular{\gamma_1=}$"
                f"{gamma1:.2f}"
                r"$\pm$"
                f"{gamma1_err:.2f}"
                "\n"
                r"$\mathregular{\gamma_2=}$"
                f"{gamma2:.2f}"
                r"$\pm$"
                f"{gamma2_err:.2f}"
                "\n"
                r"$\mathregular{\alpha=}$"
                f"{alpha:.2f}"
            ),
        )

        _plot_energy_marker(
            ax,
            breakp_1,
            breakp_1_err,
            color="blue",
            label_prefix="E_b=",
        )

        _plot_energy_marker(
            ax,
            cut,
            cut_err,
            color="purple",
            label_prefix="E_c=",
        )

        _update_result_dataframe(
            result_dataframe,
            redchi=redchi_cut,
            c1=c1,
            c1_err=errors[0],
            gamma1=gamma1,
            gamma1_err=gamma1_err,
            gamma2=gamma2,
            gamma2_err=gamma2_err,
            breakp_1=breakp_1,
            breakp_1_err=breakp_1_err,
            cutoff=cut,
            cutoff_err=cut_err,
            alpha=alpha,
            exponent=exponent,
            e_min=e_min,
            e_max=e_max,
        )

    # ========================================================================
    # 18. TRIPLE RESULT
    # ========================================================================

    elif which_fit_final == "triple":

        result_triple = result_final

        breakp_1 = result_triple.beta[6]
        breakp_2 = result_triple.beta[7]

        alpha = result_triple.beta[4]
        beta = result_triple.beta[5]

        redchi_triple = result_triple.res_var

        errors = _get_fit_errors(result_triple)

        breakp_1_err = errors[6]
        breakp_2_err = errors[7]

        c1 = result_triple.beta[0]

        # ================================================================
        # RAW MATHEMATICAL GAMMAS
        # ================================================================
        #
        # These are the parameters that belong to the mathematical
        # convolution function.
        #
        # They are deliberately NOT assumed to be the physical spectral
        # Gamma1/Gamma2/Gamma3 ordering.
        #
        # The plotted curve below uses these parameters directly through
        #
        #     pl_fit.triple_pl_func(result_triple.beta, xplot)
        #
        # ================================================================

        raw_gamma1 = result_triple.beta[1]
        raw_gamma2 = result_triple.beta[2]
        raw_gamma3 = result_triple.beta[3]

        raw_gamma1_err = errors[1]
        raw_gamma2_err = errors[2]
        raw_gamma3_err = errors[3]

        # ================================================================
        # PHYSICAL / PLOT-ORDER GAMMAS
        # ================================================================
        #
        # Because alpha and beta can be positive or negative, the
        # convolution can exchange which raw mathematical gamma appears
        # as the first, second, or third physical power-law component.
        #
        # Therefore the reported Gamma1/Gamma2/Gamma3 are reordered here.
        #
        # THIS DOES NOT CHANGE THE FUNCTION BEING PLOTTED.
        # ================================================================

        (
            gamma1,
            gamma1_err,
            gamma2,
            gamma2_err,
            gamma3,
            gamma3_err,
        ) = _reorder_triple_gammas(
            raw_gamma1,
            raw_gamma2,
            raw_gamma3,
            raw_gamma1_err,
            raw_gamma2_err,
            raw_gamma3_err,
            alpha,
            beta,
        )

        # ================================================================
        # Reported alpha/beta convention
        # ================================================================

        alpha, beta = _apply_triple_sign_convention(
            gamma1,
            gamma2,
            gamma3,
            alpha,
            beta,
        )

        # ================================================================
        # PLOT THE ACTUAL MATHEMATICAL FUNCTION
        # ================================================================
        #
        # DO NOT replace result_triple.beta by gamma1/gamma2/gamma3.
        #
        # The plotted function is the actual ODR mathematical model.
        # ================================================================

        fit_plot = pl_fit.triple_pl_func(
            result_triple.beta,
            xplot,
        )

        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:

            ax.plot(
                [],
                [],
                " ",
                label="Triple pl",
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    r"$\mathregular{\chi²=}$"
                    f"{redchi_triple:.2f}"
                ),
            )

            ax.plot(
                [],
                [],
                " ",
                label=(
                    r"$\mathregular{I_0=}$"
                    + f"{c1:.2e}"
                    + "/(s cm² sr MeV)"
                ),
            )

        ax.plot(
            xplot,
            fit_plot,
            "-b",
            label=(
                r"$\mathregular{\gamma_1=}$"
                f"{gamma1:.2f}"
                r"$\pm$"
                f"{gamma1_err:.2f}"
                "\n"
                r"$\mathregular{\gamma_2=}$"
                f"{gamma2:.2f}"
                r"$\pm$"
                f"{gamma2_err:.2f}"
                "\n"
                r"$\mathregular{\gamma_3=}$"
                f"{gamma3:.2f}"
                r"$\pm$"
                f"{gamma3_err:.2f}"
                "\n"
                r"$\mathregular{\alpha=}$"
                f"{alpha:.2f}"
                "\n"
                r"$\mathregular{\beta=}$"
                f"{beta:.2f}"
            ),
        )

        _plot_energy_marker(
            ax,
            breakp_1,
            breakp_1_err,
            color="blue",
            label_prefix="E_b1=",
        )

        _plot_energy_marker(
            ax,
            breakp_2,
            breakp_2_err,
            color="purple",
            label_prefix="E_b2=",
        )

        _update_result_dataframe(
            result_dataframe,
            redchi=redchi_triple,
            c1=c1,
            c1_err=errors[0],
            gamma1=gamma1,
            gamma1_err=gamma1_err,
            gamma2=gamma2,
            gamma2_err=gamma2_err,
            gamma3=gamma3,
            gamma3_err=gamma3_err,
            breakp_1=breakp_1,
            breakp_1_err=breakp_1_err,
            breakp_2=breakp_2,
            breakp_2_err=breakp_2_err,
            alpha=alpha,
            beta=beta,
            e_min=e_min,
            e_max=e_max,
        )

    # ========================================================================
    # 19. Save results
    # ========================================================================

    if path is not None:

        with open(path, "wb") as f:
            pickle.dump(result, f)

    if path2 is not None:

        result_dataframe.to_csv(
            path2,
            sep=";",
        )

    return result