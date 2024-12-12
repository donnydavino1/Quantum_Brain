import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import Qobj
from numpy.typing import NDArray
from scipy import integrate
from scipy.optimize import curve_fit

import operators as op
from coeff_groups_class import CoefficientGroups

""" PULSEE imports """
from matplotlib import colorbar as clrbar, colors as clrs
import matplotlib.colors
from fractions import Fraction


def plot_spectra_together(freqs, data, names, xlims, l_freq, r_freq, int_width=None, share_y=False):
    assert len(data) == len(names), "Dimension of data and names do not match!"
    assert len(freqs) == data.shape[1], "Dimension of the given frequency array does not match the spectrum data!"

    # Indices of frequency x limits
    i_L = np.absolute(freqs - xlims[0]).argmin()
    i_R = np.absolute(freqs - xlims[1]).argmin()

    n_rows = len(data) // 2
    n_cols = 2
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=(10, 10), sharey=share_y)

    for i in range(len(data)):
        ix = i % 2
        iy = i // 2
        ax = axs[iy, ix]
        ax.plot(freqs[i_L: i_R + 1], data[i][i_L: i_R + 1])
        ax.grid(alpha=0.5)
        ax.set_title(names[i])
        # x label just for the bottom 2 plots
        if i >= len(names) - 2:
            ax.set(xlabel='Offset Freq (Hz)')
        ax.tick_params(direction='in')

        ax.axvline(l_freq, color='r', alpha=0.5)
        ax.axvline(r_freq, color='r', alpha=0.5)

        if int_width is not None:
            ax.axvspan(l_freq - int_width / 2, l_freq + int_width / 2, color='orange', alpha=0.5)
            ax.axvspan(r_freq - int_width / 2, r_freq + int_width / 2, color='orange', alpha=0.5)
    return fig, axs


# Given an index, returns the element type "P1" or "P2"
def index_to_element(index: int, index_cutoff=14) -> str:
    if index < index_cutoff:
        return 'P1'
    return 'P2'


""" integration functions """


def integrate_simpson(freqs: NDArray,
                      all_spectra: NDArray,
                      p1_freqs: tuple[int],
                      p2_freqs: tuple[int],
                      int_width: float,
                      p1_offset: float = 0,
                      p2_offset: float = 0,
                      use_abs_trace: bool = True,
                      return_error: bool = False,
                      positive_diag: bool = False
) -> Qobj:

    freq_spacing = float(freqs[-1] - freqs[0]) / (len(freqs) - 1)
    p1_L_left_idx = np.abs(freqs - (p1_freqs[0] - int_width / 2 + p1_offset)).argmin()
    p1_L_right_idx = np.abs(freqs - (p1_freqs[0] + int_width / 2 + p1_offset)).argmin()
    p1_R_left_idx = np.abs(freqs - (p1_freqs[1] - int_width / 2 + p1_offset)).argmin()
    p1_R_right_idx = np.abs(freqs - (p1_freqs[1] + int_width / 2 + p1_offset)).argmin()

    p2_L_left_idx = np.abs(freqs - (p2_freqs[0] - int_width / 2 + p2_offset)).argmin()
    p2_L_right_idx = np.abs(freqs - (p2_freqs[0] + int_width / 2 + p2_offset)).argmin()
    p2_R_left_idx = np.abs(freqs - (p2_freqs[1] - int_width / 2 + p2_offset)).argmin()
    p2_R_right_idx = np.abs(freqs - (p2_freqs[1] + int_width / 2 + p2_offset)).argmin()

    coeff_groups_simpson = CoefficientGroups()
    for (i, spectrum) in enumerate(all_spectra):
        if index_to_element(i) == 'P1':
            L_left_i, L_right_i, R_left_i, R_right_i = p1_L_left_idx, p1_L_right_idx, p1_R_left_idx, p1_R_right_idx
        elif index_to_element(i) == 'P2':
            L_left_i, L_right_i, R_left_i, R_right_i = p2_L_left_idx, p2_L_right_idx, p2_R_left_idx, p2_R_right_idx
        else:
            raise ValueError('not correctly identifying `index_to_element`')

        L = integrate.simpson(spectrum[L_left_i: L_right_i + 1], dx=freq_spacing)
        R = integrate.simpson(spectrum[R_left_i: R_right_i + 1], dx=freq_spacing)
        coeff_groups_simpson.add_coefficient(op.product_operators[i][0], L + R, index_to_element(i))
        coeff_groups_simpson.add_coefficient(op.product_operators[i][1], L - R, index_to_element(i))
        # print(f"Spectrum {thermal_col_names[i]} L is: {L:.1e}, R is {R:.1e}")
        # print(f"Spectrum {thermal_col_names[i]} L+R is: {(L+R):.1e}, L-R is {(L-R):.1e}")

    if return_error:
        return coeff_groups_simpson.get_error()

    rho_simpson = coeff_groups_simpson.reconstruct_rho(abs_trace=use_abs_trace, positive_diag=positive_diag)
    return rho_simpson


def integrate_optimized(
        freqs: NDArray,
        all_spectra: NDArray,
        p1_freqs: tuple[int],
        p2_freqs: tuple[int],
        rho_theory: Qobj,
        p1_range: NDArray = np.arange(-5, 6, 1),
        p2_range: NDArray = np.arange(-5, 6, 1),
        width_range: NDArray = np.arange(1, 11, 1),
        use_abs_trace: bool = True,
        return_error: bool = False,
        positive_diag: bool = False,
        projection: str = "fortunato"):

    best_projection = 0
    best_offsets = tuple()
    best_int_width = 0
    best_rho_simpson = 0

    for p1_offset in p1_range:
        for p2_offset in p2_range:
            for int_width in width_range:
                curr_rho_simpson = integrate_simpson(freqs, all_spectra, p1_freqs, p2_freqs, int_width, p1_offset,
                                                     p2_offset, use_abs_trace, positive_diag=positive_diag)
                # try minimizing the negatives in the diagonal, and check if it gives us better fidelity.
                curr_rho_zeroed = zero_negatives(curr_rho_simpson)

                for curr_rho in [curr_rho_simpson, curr_rho_zeroed]:
                    if projection == "fortunato":
                        curr_projection = projection_fortunato(curr_rho, rho_theory)
                        assert np.imag(curr_projection) < 1e-6, f"Imaginary part of projection too large! {curr_projection}"
                        curr_projection = np.real(curr_projection)
                    else:  # "jozsa"
                        curr_projection = projection_2(curr_rho, rho_theory)
                        assert np.imag(curr_projection) < 1e-6, f"Imaginary part of projection too large! {curr_projection}"
                        curr_projection = np.real(curr_projection)

                    if curr_projection > best_projection:
                        best_offsets = (p1_offset, p2_offset)
                        best_int_width = int_width
                        best_projection = curr_projection
                        best_rho_simpson = curr_rho



    # usually return here
    if not return_error:
        return best_rho_simpson, best_projection, best_offsets, best_int_width

    # for error propagation: note the argument `return_error=True`.
    rho_error = integrate_simpson(freqs, all_spectra, p1_freqs, p2_freqs, best_int_width, best_offsets[0],
                                  best_offsets[1], use_abs_trace, return_error=True, positive_diag=positive_diag)
    return best_rho_simpson, best_projection, best_offsets, best_int_width, rho_error


def zero_negatives(rho: Qobj) -> Qobj:
    negatives = [elem for elem in rho.diag() if elem < 0]
    neg_avg = np.mean(negatives)
    rho_new = rho - op.IDENTITY * neg_avg
    return normalize_diag(rho_new)


def normalize_diag(rho: Qobj) -> Qobj:
    diag_sum = np.sum(rho.diag())
    return rho / diag_sum


def projection_fortunato(rho1: Qobj, rho2: Qobj) -> float:
    if (rho1 ** 2).tr() == 0 or (rho2 ** 2).tr() == 0:  # Avoid divide by zero!
        return 0
    return (rho1 * rho2).tr() / np.sqrt((rho1 ** 2).tr() * (rho2 ** 2).tr())


def projection_2(rho1: Qobj, rho2: Qobj) -> float:
    """
    Assumes rho1 and rho2 are both positive semi-definite matrices!
    """
    assert is_positive_semidefinite(rho1)
    assert is_positive_semidefinite(rho2), f"{rho2.eigenenergies()}"
    # assert rho1.sqrtm() * rho1.sqrtm().trans() == rho1
    return (((rho1.sqrtm()) * rho2 * (rho1.sqrtm())).sqrtm()).tr() ** 2


def is_positive_semidefinite(rho: Qobj) -> bool:
    eigenvalues = rho.eigenenergies()
    return np.all([eig >=0 for eig in eigenvalues])



""" scipy fit stuff """


# sum of absorptive (1) and dispersive (2) Lorentzians
def lorentzian(w, width, center, amp1, amp2):
    absorptive = amp1 * width / (width ** 2 + (w - center) ** 2)
    dispersive = -amp2 * (w - center) / (width ** 2 + (w - center) ** 2)
    return absorptive + dispersive


def plot_guess(freqs_window, spectrum, name, guesses, xlims, l_freq, r_freq):
    width = guesses[0]
    left_guess = guesses[1:4]
    right_guess = guesses[4:7]
    L_peak = lorentzian(freqs_window, width, *left_guess)
    R_peak = lorentzian(freqs_window, width, *right_guess)

    plt.title(name)
    plt.axvline(guesses[1], color='gray', alpha=0.5)
    plt.axvline(guesses[4], color='gray', alpha=0.5)
    plt.scatter(freqs_window, spectrum, s=15, label='spectrum', color='C0')
    plt.fill_between(freqs_window, L_peak, label='left guess', color='C1', alpha=0.3)
    plt.fill_between(freqs_window, R_peak, label='right guess', color='C2', alpha=0.3)
    plt.plot(freqs_window, L_peak + R_peak, label='total guess', color='red')
    plt.legend()
    plt.xlim(xlims)


# ASSUMING WIDTHS OF BOTH LEFT AND RIGHT PEAKS ARE SAME
def three_peaks(w, width, cen_L, amp1_L, amp2_L, cen_R, amp1_R, amp2_R):
    return (lorentzian(w, width, cen_L, amp1_L, amp2_L)  # left peak
            + lorentzian(w, width, cen_R, amp1_R, amp2_R))  # right peak


def fit_3peaks(freqs_window, spectrum, name, initial_guess, xlims_display, l_freq, r_freq):
    l_freq = initial_guess[1]
    r_freq = initial_guess[4]
    lower_bounds = [0, l_freq - np.absolute(l_freq) / 5, -1e10, -1e10, r_freq - r_freq / 5, -1e10, -1e10]
    upper_bounds = [50, l_freq + np.absolute(l_freq) / 5, 1e10, 1e10, r_freq + r_freq / 5, 1e10, 1e10]

    popt, pcov = curve_fit(three_peaks, freqs_window, spectrum, p0=initial_guess,
                           bounds=(lower_bounds, upper_bounds), maxfev=10000)

    [width_opt,
     cen_L_opt, amp1_L_opt, amp2_L_opt,
     cen_R_opt, amp1_R_opt, amp2_R_opt] = popt

    # For plotting only
    freqs_fit = np.linspace(freqs_window[0], freqs_window[-1], len(freqs_window) * 2)

    popt_left = [width_opt, cen_L_opt, amp1_L_opt, amp2_L_opt]
    left_fit = lorentzian(freqs_fit, *popt_left)

    popt_right = [width_opt, cen_R_opt, amp1_R_opt, amp2_R_opt]
    right_fit = lorentzian(freqs_fit, *popt_right)

    plt.title(name)
    plt.grid(alpha=0.5)
    plt.axvline(cen_L_opt, color='black', alpha=0.5)
    plt.axvline(cen_R_opt, color='black', alpha=0.5)
    plt.scatter(freqs_window, spectrum, s=1, label='spectrum', color='black')
    # plt.plot(freqs_fine, left_fit, color='green')
    plt.fill_between(freqs_fit, left_fit, label='left peak fit', facecolor='green', alpha=0.3)
    # plt.plot(freqs_fine, right_fit, color='yellow')
    plt.fill_between(freqs_fit, right_fit, label='right peak fit', facecolor='orange', alpha=0.3)
    plt.plot(freqs_fit, left_fit + right_fit, label='total fit', color='red')

    plt.xlim(xlims_display)
    plt.legend()
    plt.show()
    return popt, pcov


def calculate_r2(freqs_window, all_spectra_window, popts):
    r2s = []
    for i in range(len(popts)):
        spectrum = all_spectra_window[i]
        ss_res = np.sum((spectrum - three_peaks(freqs_window, *popts[i])) ** 2)
        ss_tot = np.sum((spectrum - np.mean(spectrum)) ** 2)
        r2 = 1 - ss_res / ss_tot
        r2s.append(r2)
    return r2s


""" Error Stuff """


def multiply_error_matrix(
        rho1_qobj: Qobj,
        err1_qobj: Qobj,
        rho2_qobj: Qobj,
        err2_qobj: Qobj
) -> Qobj:
    assert rho1_qobj.shape == rho2_qobj.shape == err1_qobj.shape == err2_qobj.shape
    assert np.array_equal(err1_qobj, np.real(err1_qobj)) and np.array_equal(err2_qobj, np.real(err2_qobj))

    rho1, rho2 = rho1_qobj.full(), rho2_qobj.full()
    err1, err2 = np.real(err1_qobj.full()), np.real(err2_qobj.full())
    value_matrix = np.empty(rho1.shape, dtype=np.complex128)
    error_matrix = np.empty(rho1.shape, dtype=float)
    for i in range(rho1.shape[0]):
        for j in range(rho1.shape[1]):
            row_val, row_err = rho1[i, :], err1[i, :]
            col_val, col_err = rho2[:, j], err2[:, j]
            val_list = row_val * col_val
            # if any element of row_val or col_val is 0, dividing by this will give NaN. Using `numpy_divide` to
            # instead give 0 instead of NaN.
            row_err_reduced = np.divide(row_err, np.abs(row_val), out=np.zeros_like(row_err), where=(row_val != 0))
            col_err_reduced = np.divide(col_err, np.abs(col_val), out=np.zeros_like(col_err), where=(col_val != 0))

            err_list = np.abs(val_list) * np.sqrt(row_err_reduced ** 2 + col_err_reduced ** 2)
            err = np.sqrt(np.sum(err_list ** 2))

            error_matrix[i, j] = err
            value_matrix[i, j] = np.sum(val_list)

    # if not np.allclose(value_matrix, (rho_1_qobj * rho_2_qobj).full()):
    #     print(f"value_matrix is: \n{value_matrix}")
    #     print(f"Qobj multiplication: \n{(rho_1_qobj * rho_2_qobj).full()}")

    assert np.allclose(value_matrix, (rho1_qobj * rho2_qobj).full())
    return error_matrix


def tr_error(rho_err: Qobj) -> float:
    return np.sqrt(np.sum(np.diag(rho_err) ** 2))


def fortunato_error(rho1: Qobj, err1: Qobj, rho2: Qobj, err2: Qobj) -> float:
    numer = (rho1 * rho2).tr()
    denom = np.sqrt((rho1 ** 2).tr() * (rho2 ** 2).tr())
    err_numer = tr_error(multiply_error_matrix(rho1, err1, rho2, err2))
    err_denom_l = tr_error(multiply_error_matrix(rho1, err1, rho1, err1))
    err_denom_r = tr_error(multiply_error_matrix(rho2, err2, rho2, err2))
    err_denom_0 = ((rho1 ** 2).tr() * (rho2 ** 2).tr()
                   * np.sqrt((err_denom_l / (rho1 ** 2).tr()) ** 2 + (err_denom_r / (rho2 ** 2).tr()) ** 2))
    err_denom = 1/2 * err_denom_0 / np.sqrt(denom)

    assert (numer / denom) == projection_fortunato(rho1, rho2)
    err_final = numer / denom * np.sqrt((err_numer / numer) ** 2 + (err_denom / denom) ** 2)
    return err_final

