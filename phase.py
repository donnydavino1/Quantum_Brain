import nmrglue as ng
import numpy as np
from numpy.typing import NDArray
from scipy import integrate


def auto_phase_all_spectra(ffts_avg: NDArray, phase_width: int, state_name: str):
    """
    Automatic 0th order phase. A bit hard coded since we use the P1 X1 imag spectra (L+R = I_1z, L-R = I_1z I_2z) and
    P2 X2 imag spectra (L+R = I_2z, L-R = I_1z I_2z) to phase all the P1 and P2 spectra (respectively) since we
    `know' theoretically what they should look like. (ex. the '1000' up-up state should have only absorption peaks in
    P1X1 and P2X2 spectra.
    :param ffts_avg:
    :param phase_width:
    :param state_name:
    :return:
    """
    mid = ffts_avg.shape[0] // 2

    if state_name == "1000":
        PH0_P1_OFFSET, PH0_P2_OFFSET = 0, 0
    elif state_name == "0100":
        PH0_P1_OFFSET, PH0_P2_OFFSET = 0, 180
    elif state_name == "0010":
        PH0_P1_OFFSET, PH0_P2_OFFSET = 180, 0
    elif state_name == "0001":
        PH0_P1_OFFSET, PH0_P2_OFFSET = 180, 180
    else:
        raise ValueError(f"Unknown state_name: {state_name}. Should be one of ('1000', '0100', '0010', '0001')")

    spectrum_p1x1 = ffts_avg[1]  # P1 X1 (L+R = I_1z, L-R = I_1z I_2z)
    ph_0_p1 = autophase_0(spectrum_p1x1, phase_width)
    ffts_avg[:mid] = np.array(
        [ng.proc_base.ps(s, p0=ph_0_p1 + PH0_P1_OFFSET, p1=0) for s in ffts_avg[:mid]])
    spectrum_p2x2 = ffts_avg[10]  # P2 X2 (L+R = I_2z, L-R = I_1z I_2z)
    ph_0_p2 = autophase_0(spectrum_p2x2, phase_width)
    ffts_avg[mid:] = np.array(
        [ng.proc_base.ps(s, p0=ph_0_p2 + PH0_P2_OFFSET, p1=0) for s in ffts_avg[mid:]])


def autophase_0(spectrum: NDArray,
                width: int = None,
                phase_range: NDArray = None,
                ) -> float:
    if width is not None:
        mid = len(spectrum) // 2
        spectrum = spectrum[mid - width // 2: mid + width // 2]

    if phase_range is None:
        phase_range = np.linspace(-180, 180, 721)

    ph0_best = None
    area_best = 0

    for ph0 in phase_range:
        spectrum_curr_real = ng.proc_base.ps(spectrum, ph0).imag  # Since we're looking at P1 X1 IMAG / P2 X2 IMAG
        area_curr = integrate.simpson(spectrum_curr_real)

        if area_curr > area_best:
            ph0_best = ph0
            area_best = area_curr

    return ph0_best

# def autophase_0(spectrum: NDArray,
#                 width: int = None,
#                 phase_range: NDArray = None,
#                 show: bool = False
#                 ) -> float:
#     # Since we're looking at P1 X1 / P2 X2 spectra,
#     # want to maximize the imag (absorption) integral and minimize the real (dispersion) integral
#
#     if width is not None:
#         mid = len(spectrum) // 2
#         spectrum = spectrum[mid - width // 2: mid + width // 2]
#
#     if show:
#         plt.plot(spectrum)
#
#     if phase_range is None:
#         phase_range = np.linspace(-180, 180, 721)
#
#     ph0_real, ph0_imag = None, None
#     area_max, area_min = 0, np.inf
#
#     for ph0 in phase_range:
#         spectrum_curr = ng.proc_base.ps(spectrum, ph0)
#         area = integrate.simpson(spectrum_curr)
#
#         if area.imag > area_max:
#             ph0_imag = ph0
#             area_max = area.imag
#         if np.abs(area.real) < area_min:
#             ph0_real = ph0
#             area_min = np.abs(area.real)
#
#     ph0_mean = (ph0_real + ph0_imag) / 2
#     print(f"ph0_real = {ph0_real}, ph0_imag = {ph0_imag}. Average: {ph0_mean}")
#     return ph0_mean


# def autophase_all(spectra_complex: NDArray,
#                   state_name: str,
#                   phase_width: int = 200) -> NDArray:
#     l = spectra_complex.shape[0] // 2
#
#     spectrum_p1x1 = spectra_complex[1]  # P1 X1 (L+R = I_1z, L-R = I_1z I_2z)
#     ph_0_p1 = autophase_0(spectrum_p1x1, phase_width)
#
#     spectrum_p2x2 = spectra_complex[10]  # P2 X2 (L+R = I_2z, L-R = I_1z I_2z)
#     ph_0_p2 = autophase_0(spectrum_p2x2, phase_width)
#
#     # if ("1000" in state_name) and ("CN" not in state_name):
#
#     spectra_complex[:l] = np.array([ng.proc_base.ps(s, p0=ph_0_p1 + 90, p1=0) for s in spectra_complex[:l]])
#     spectra_complex[l:] = np.array([ng.proc_base.ps(s, p0=ph_0_p2 + 90, p1=0) for s in spectra_complex[l:]])
