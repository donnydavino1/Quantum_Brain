import os

import nmrglue as ng
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.fft import fftfreq, fftshift

from phase import auto_phase_all_spectra

""" Specifying `ns` will now do the phase cycling averaging (assuming they're in the convention Donny has in his 
notebook `ns4_array.ipynb`). """


def temporal_average(
        data_folder: str,
        AQ: float = 1.6,
        PHASE_0: float = -44,
        LEFT_SHIFT_INDEX: int = 0,
        remove_digital: bool = False,
        ns: int | None = None,
        auto_phase: bool = False,
        phase_width: int = 150,
        state_name: str = "1000",
        display_fid: bool = False,
        display_fft: bool = False,
        display_avg: bool = False,
        DISPLAY_I: int = 0,
        XLIM_DISPLAY: tuple[float, float] = (-50, 50)
) -> tuple[NDArray, NDArray]:
    # SWH = 2000,  # Spectral Width [Hz]
    # SF = 202.4765750,  # Spectrometer frequency [Mhz]
    # SFO1 = 202.4751988,  # Transmitter frequency [Mhz]
    folder_nums = os.listdir(f"{data_folder}")
    folder_nums = [str(s) for s in (sorted([int(x) for x in folder_nums]))]  # ordering by increasing number value
    file_names = [fr"{data_folder}\{fn}" for fn in folder_nums]

    ffts_proc = []
    freqs = process_combined_data(file_names[0], PHASE_0=PHASE_0, ACQ_TIME=AQ, XLIM_DISPLAY=XLIM_DISPLAY,
                                  LEFT_SHIFT_INDEX=LEFT_SHIFT_INDEX, remove_digital=remove_digital)[0]
    for fn in file_names:
        ffts_proc.append(
            process_combined_data(fn, display_fid=display_fid, display_fft=display_fft, PHASE_0=PHASE_0,
                                  ACQ_TIME=AQ, XLIM_DISPLAY=XLIM_DISPLAY, LEFT_SHIFT_INDEX=LEFT_SHIFT_INDEX,
                                  remove_digital=remove_digital)[1]
        )
    ffts_proc = np.array(ffts_proc)

    # Phase cycling:
    if ns is None:  # no phase cycling (the old usual method)
        assert ffts_proc.shape[:2] == (14, 3), "Data is not the expected shape of (14, 3, num_points)"
        pass
    else:  # newly added method for averaging different phase cycled spectra
        assert ffts_proc.shape[0] == 14 * ns, \
            f"There should be 14*ns number of folders. Currently has: {ffts_proc.shape[0]}"
        assert ffts_proc.shape[1] == 3

        ffts_averaged = np.zeros((14, 3, ffts_proc.shape[2]), dtype=ffts_proc.dtype)
        for i in range(ffts_proc.shape[0]):  # 14
            ffts_averaged[i] = np.mean(ffts_proc[i * ns: i * ns + ns], axis=0)
        ffts_proc = ffts_averaged

    # Temporal 'Averaging'
    ffts_avg = np.mean(ffts_proc, axis=1)

    # After `remove_digital`, the 0th order phase gets messed up (since it changes the left-shifting of the FID),
    # so I have to manually phase the spectra again
    if auto_phase:
        auto_phase_all_spectra(ffts_avg, phase_width, state_name)

    if display_avg:
        plot_avg(freqs, ffts_avg, ffts_proc, DISPLAY_I, XLIM_DISPLAY)

    return freqs, ffts_avg


def process_combined_data(
        data_folder: str,
        display_fid: bool = False,
        display_fft: bool = False,
        PHASE_0: float = -40,
        ACQ_TIME: float = 1.6,
        XLIM_DISPLAY: tuple[float, float] = (-50, 50),
        LEFT_SHIFT_INDEX: int = None,
        remove_digital: bool = False
) -> tuple[NDArray, NDArray]:
    dic_data, all_data = ng.fileio.bruker.read(data_folder)

    if remove_digital:
        all_data = ng.bruker.remove_digital_filter(dic_data, all_data)

    # Trying to figure out a way to read data because folder 3 is giving us problems
    # all_data = []
    # for i in range(1, 5):
    #     all_data.append(ng.fileio.bruker.read_pdata(f"{data_folder}/pdata/{i}")[1])

    all_data = np.array(all_data)
    # print(f"all_data.shape = {all_data.shape}")
    assert all_data.shape[0] == 4  # the first three arrays are data, and the last array is a placeholder!
    all_data = all_data[:3, :]
    # (Bruker needs them in factors of 2 or something)

    # Phase shift, because our receiver is at -y instead of the standard x
    all_data = ng.proc_base.ps(all_data, PHASE_0 + 180)
    # all_data = ng.process.proc_base.zf_auto(all_data)

    # print(np.abs(all_data[0, :100]))
    # print(np.abs(all_data[0, 63:65]))
    # print(63 * ACQ_TIME / (all_data.shape[1] - 1))

    fids_processed = prepare_fid(all_data, display_fid, ACQ_TIME=ACQ_TIME, LEFT_SHIFT_INDEX=LEFT_SHIFT_INDEX)

    ffts_raw = fft_fids(all_data)
    ffts_proc = fft_fids(fids_processed)
    freqs = fftshift(fftfreq(all_data.shape[1], d=ACQ_TIME / (all_data.shape[1] - 1)))

    if display_fft:
        plot_fft(freqs, ffts_raw, ffts_proc, XLIM_DISPLAY)

    return freqs, ffts_proc


def prepare_fid(all_data: NDArray, display: bool = False, ACQ_TIME=1.6, LEFT_SHIFT_INDEX=63) -> NDArray:
    times = np.linspace(0, ACQ_TIME, all_data.shape[1])
    # Bruker's FID has such a large signal-to-noise that apodization and baseline seems unnecessary.
    # So only doing left-shift for processing.

    # I've noticed most FID start with a lot of zero-points (not sure why). Left-shifting data so the FID starts at
    # the largest absolute value

    # i_max = np.argmax(np.abs(all_data), axis=1)
    # data_ls = np.array([ng.process.proc_base.ls(all_data[i], i_max[i]) for i in range(all_data.shape[0])])

    # if LEFT_SHIFT_INDEX is 0, nmrglue's left shift function transforms all data into 0 !! (unexpected behavior)
    if LEFT_SHIFT_INDEX == 0:
        data_ls = all_data
    else:
        data_ls = np.array([ng.process.proc_base.ls(all_data[i], LEFT_SHIFT_INDEX) for i in range(all_data.shape[0])])
        assert data_ls.shape == all_data.shape

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
            ax.set_xlim(0, 0.1)

        axs[0, 0].set_title("Raw FID")
        axs[0, 1].set_title("Processed FID (left shift)")

        fig.tight_layout()
        plt.show()

    return data_ls


def fft_fids(fids: NDArray) -> NDArray:
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


def test_run():
    DATA_FOLDER = r"C:\Users\lemon\OneDrive - Brown University\CNOT project\MIT\data\2024_12_13\1000_V6_NS4_2D_2ch\3"
    process_combined_data(DATA_FOLDER, display_fid=True, display_fft=True)

    DATA_FOLDER = r"C:\Users\lemon\OneDrive - Brown University\CNOT project\MIT\data\2024_12_13\1000_V6_NS4_2D_2ch"
    temporal_average(DATA_FOLDER, display_avg=True)
    return


""" Plotting function """


def plot_avg(freqs, ffts_avg, ffts_proc, DISPLAY_I, XLIM_DISPLAY):
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


def plot_fft(freqs, ffts_raw, ffts_proc, XLIM_DISPLAY):
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
