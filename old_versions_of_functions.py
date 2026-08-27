
import pickle
import numpy as np
import pandas as pd

from scipy.stats import t as studentt


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

        sorted_chis = dict(sorted(chis.items(), key=lambda x: x[1], reverse=False))

        if exponent_cut == 0:
            sorted_chis.pop("cut")

        if exponent_cut_break == 0:
            sorted_chis.pop("double_cut")

        # Check if there are values with zero chi sq. If so, delete
        # from dict. Then check if dict is empty. If yes: loop again
        # through the results.
        # The make the fit function takes care of the possibly empty list
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
                if (emin < breakp_low < emax
                    and emin < breakp_high < emax
                    and breakp_low < breakp_high):

                    absolute_val_array = np.abs(spec_e - breakp_low)
                    smallest_difference_index = absolute_val_array.argmin()
                    low = ""
                    high = ""

                    if smallest_difference_index == len(spec_e) - 1:
                        low = (spec_e[smallest_difference_index - 1]
                            - e_err[smallest_difference_index - 1])
                        high = (spec_e[smallest_difference_index]
                            + e_err[smallest_difference_index])
                    else:
                        low = (spec_e[smallest_difference_index]
                            - e_err[smallest_difference_index])
                        high = (spec_e[smallest_difference_index + 1]
                            + e_err[smallest_difference_index + 1])

                    difference_triple_energy = high - low

                    # The triple PL is defined so that the two breaks are
                    # actually interchangeable. This means that it can
                    # happen that the 'high' break becomes the low break.
                    # These cases have to be deleted because it messes with
                    # the meaning of the parameters of the fit.

                    if (difference_triple > difference_triple_energy and gamma1 < 0):

                        if alpha > 0 or beta > 0:
                            which_fit = "triple"
                            redchi = redchi_triple
                            result = result_triple
                            return [which_fit, redchi, result]
                        else:
                            smallest_value = list(sorted_chis.keys())[i]

                    else:
                        smallest_value = list(sorted_chis.keys())[i]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "double_cut":
                if (emin < cut_b < emax
                    and emin < breakp_cut < emax
                    and cut_b > breakp_cut):

                    absolute_val_array = np.abs(spec_e - breakp_cut)
                    smallest_difference_index = absolute_val_array.argmin()
                    low = ""
                    high = ""

                    if smallest_difference_index == len(spec_e) - 1:
                        low = (spec_e[smallest_difference_index - 1]
                            - e_err[smallest_difference_index - 1])
                        high = (spec_e[smallest_difference_index]
                            + e_err[smallest_difference_index])
                        
                    else:
                        low = (spec_e[smallest_difference_index]
                            - e_err[smallest_difference_index])
                        high = (spec_e[smallest_difference_index + 1]
                            + e_err[smallest_difference_index + 1])

                    difference_cut_energy = high - low

                    if (gamma1 < 0 and difference_cut > difference_cut_energy):

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

        if (breakp_low < emax
            and breakp_low > emin
            and breakp_high < emax
            and breakp_high > emin):

            absolute_val_array = np.abs(spec_e - breakp_low)
            smallest_difference_index = absolute_val_array.argmin()
            low = ""
            high = ""

            if smallest_difference_index == len(spec_e) - 1:
                low = (spec_e[smallest_difference_index - 1]
                    - e_err[smallest_difference_index - 1])
                high = (spec_e[smallest_difference_index]
                    + e_err[smallest_difference_index])
            else:
                low = (spec_e[smallest_difference_index]
                    - e_err[smallest_difference_index])
                high = (spec_e[smallest_difference_index + 1]
                    + e_err[smallest_difference_index + 1])

            difference_triple_energy = high - low

            if (breakp_high > breakp_low and difference_triple > difference_triple_energy):
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

        if (breakp_cut <= emax
            and breakp_cut > emin
            and cut_b <= emax
            and cut_b > emin):

            absolute_val_array = np.abs(spec_e - breakp_cut)
            smallest_difference_index = absolute_val_array.argmin()
            low = ""
            high = ""

            if smallest_difference_index == len(spec_e) - 1:
                low = (spec_e[smallest_difference_index - 1]
                    - e_err[smallest_difference_index - 1])
                high = (spec_e[smallest_difference_index]
                    + e_err[smallest_difference_index])
            else:
                low = (spec_e[smallest_difference_index]
                    - e_err[smallest_difference_index])
                high = (spec_e[smallest_difference_index + 1]
                    + e_err[smallest_difference_index + 1])

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

    if fit == "best_sd":
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

        # c1_array... we want to get a good approximation of the particle intensity
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
        # cut array 

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

    
    color = {"sun": "crimson", "asun": "orange", "north": "darkslateblue", "south": "c"}

    spec_e = np.array(spec_e)
    spec_flux = np.array(spec_flux)
    e_err = np.array(e_err)
    flux_err = np.array(flux_err)

    xplot = np.logspace(
        np.log10(np.nanmin(spec_e)),
        np.log10(np.nanmax(spec_e)),
        num=500,
        )
    xplot = xplot[np.where((xplot >= e_min) & (xplot <= e_max))[0]]

    fit_ind = np.where((spec_e >= e_min)
            & (spec_e <= e_max)
            & np.isfinite(spec_flux)
            & np.isfinite(flux_err)
            )[0]

    spec_e = spec_e[fit_ind]
    spec_flux = spec_flux[fit_ind]
    e_err = e_err[fit_ind]
    flux_err = flux_err[fit_ind]

    # Everything is done first with input guess values and then with randoms.

    which_fit_final = ""

    redchi_final = 0

    result_final = None
    convergence = True

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
            maxit=1000,
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
                        
    if which_fit == "best_sd":
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
            fit="best_sd",
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
                    fit="best_sd",
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
