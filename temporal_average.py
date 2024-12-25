import os

import nmrglue as ng
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.fft import fftfreq, fftshift
""" Global Variables """
ACQ_TIME = 1.6# Acquistion time (s)
SWH = 2000  # Spectral Width [Hz]
SF = 202.4765750 # Spectrometer frequency [Mhz]
SFO1 = 202.4751924  # Transmitter frequency [Mhz]
LEFT_SHIFT_INDEX = 63
XLIM_DISPLAY = -50, 50


def temporal_average(data_folder: str,
                     display_fid: bool = False,
                     display_fft: bool = False,
                     display_avg: bool = False,
                     DISPLAY_I: int = 0,
                     phase_0=-44):
    folder_nums = os.listdir(f"{data_folder}")
    folder_nums = [str(s) for s in (sorted([int(x) for x in folder_nums]))]  # ordering by increasing number value
    file_names = [fr"{data_folder}\{fn}" for fn in folder_nums]

    ffts_proc = []
    freqs = get_3_spectra(file_names[0])[0]
    for fn in file_names:
        ffts_proc.append(get_3_spectra(fn, display_fid=display_fid, display_fft=display_fft, phase_0=phase_0)[1])


    ffts_proc = np.array(ffts_proc)
    print(ffts_proc.shape)
    ffts_avg = np.mean(ffts_proc, axis=1)

    if display_avg:
        fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True, figsize=(8, 12))
        plot_complex(axs[0], freqs, ffts_proc[DISPLAY_I, 0, :])
        plot_complex(axs[1], freqs, ffts_proc[DISPLAY_I, 1, :])
        plot_complex(axs[2], freqs, ffts_proc[DISPLAY_I, 2, :])
        plot_complex(axs[3], freqs, ffts_avg[DISPLAY_I, :])
        axs[3].set_title("Averaged:")

        for ax in axs.flat:
            ax.grid(True)
            ax.legend()
            ax.set(xlabel="Freq (Hz)", ylabel="intensity (a.u.)")
            ax.tick_params(direction='in')
            ax.label_outer()
            ax.set_xlim(XLIM_DISPLAY)

        fig.tight_layout()
        plt.show()

    print(ffts_avg.shape)
    return freqs, ffts_avg


def get_3_spectra(data_folder: str,
                  display_fid: bool = False,
                  display_fft: bool = False,
                  phase_0: float = -44):
    all_data = ng.fileio.bruker.read(data_folder)[1]

    # Trying to figure out a way to read data because folder 3 is giving us problems
    # all_data = []
    # for i in range(1, 5):
    #     all_data.append(ng.fileio.bruker.read_pdata(f"{data_folder}/pdata/{i}")[1])

    all_data = np.array(all_data)
    assert all_data.shape[0] == 4  # the first three arrays are data, and the last array is a placeholder!
    all_data = all_data[:3, :]
    # (Bruker needs them in factors of 2 or something)

    # Phase shift, because our receiver is at -y instead of the standard x
    all_data = ng.proc_base.ps(all_data, phase_0 + 180)
    # all_data = ng.process.proc_base.zf_auto(all_data)

    # print(np.abs(all_data[0, :100]))
    # print(np.abs(all_data[0, 63:65]))
    # print(63 * ACQ_TIME / (all_data.shape[1] - 1))

    fids_processed = prepare_fid(all_data, display_fid)

    ffts_raw = fft_fids(all_data)
    ffts_proc = fft_fids(fids_processed)
    freqs = fftshift(fftfreq(all_data.shape[1], d=ACQ_TIME / (all_data.shape[1] - 1)))

    if display_fft:
        n_rows, n_cols = 3, 2
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True, figsize=(10, 8))
        for i in range(n_rows):
            plot_complex(axs[i, 0], freqs, ffts_raw[i, :])  # Raw spectra
            plot_complex(axs[i, 1], freqs, ffts_proc[i, :])  # Processed spectra

        for ax in axs.flat:
            ax.grid(True)
            ax.legend()
            ax.set(xlabel="Freq (Hz)", ylabel="intensity (a.u.)")
            ax.tick_params(direction='in')
            ax.label_outer()
            ax.set_xlim(XLIM_DISPLAY)
            # ax.set_ylim(-2e10, 2e10)

        axs[0, 0].set_title("Raw FFT")
        axs[0, 1].set_title("Processed FFT (left shift)")

        fig.tight_layout()
        plt.show()

    return freqs, ffts_proc


def prepare_fid(all_data: NDArray, display: bool = False) -> NDArray:
    times = np.linspace(0, ACQ_TIME, all_data.shape[1])

    # I've noticed most FID start with a lot of zero-points (not sure why). Left-shifting data so the FID starts at
    # the largest absolute value

    # i_max = np.argmax(np.abs(all_data), axis=1)
    # data_ls = np.array([ng.process.proc_base.ls(all_data[i], i_max[i]) for i in range(all_data.shape[0])])
    data_ls = np.array([ng.process.proc_base.ls(all_data[i], LEFT_SHIFT_INDEX) for i in range(all_data.shape[0])])

    if display:
        n_rows, n_cols = 3, 2
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True, figsize=(10, 8))
        for i in range(n_rows):
            plot_complex(axs[i, 0], times, all_data[i, :])  # Raw FID
            plot_complex(axs[i, 1], times, data_ls[i, :])  # Processed FID

        for ax in axs.flat:
            ax.grid(True)
            ax.legend()
            ax.set(xlabel="time (s)", ylabel="intensity (a.u.)", xlim=(0, ACQ_TIME))
            ax.tick_params(direction='in')
            ax.label_outer()
            ax.set_xlim(0.35, 0.45)

        axs[0, 0].set_title("Raw FID")
        axs[0, 1].set_title("Processed FID (left shift)")

        fig.tight_layout()
        plt.show()

    return data_ls


def fft_fids(fids):
    n_1d, n_2d = np.shape(fids)
    fft_arr = np.zeros_like(fids)
    # for i in range(n_1d):
    #     fft_arr[i, :] = fftshift(fft(fids[i, :]))

    fft_arr = ng.proc_base.fft(fids)

    return fft_arr


def plot_complex(ax, xs, ys):
    ax.plot(xs, ys.real, label="real", color="green")
    ax.plot(xs, ys.imag, label="imag", color="red")
    ax.plot(xs, np.abs(ys), label="abs", color="blue")


# DATA_FOLDER = r"C:\Users\lemon\OneDrive - Brown University\CNOT project\MIT\data\2024_12_13\1000_V6_NS4_2D_2ch\3"
# # get_3_spectra(DATA_FOLDER, display_fid=True, display_fft=True)
#
# DATA_FOLDER = r"C:\Users\lemon\OneDrive - Brown University\CNOT project\MIT\data\2024_12_13\1000_V6_NS4_2D_2ch"
# temporal_average(DATA_FOLDER, display_avg=True)
