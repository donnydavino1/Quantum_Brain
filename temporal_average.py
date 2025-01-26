import os
AQ=1.6# Acquistion time (s)
SWH=2000 #Spectral Width [Hz]
SF=202.4765750 #Spectrometer frequency [Mhz]
SFO1=202.4751988 #Transmitter frequency [Mhz]
p0=-32
p1=0
center=-(SF-SFO1)*10**6
plot_width=30

import pandas as pd
import nmrglue as ng
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.fft import fftfreq, fftshift

""" The 'new' function that works with 3 separate Bruker files to combine & process the data into a single spectra. """

# def temporal_average_separate(
#         data_folder: str,
#         display_fid: bool = False,
#         display_fft: bool = False,
#         display_avg: bool = False,
#         DISPLAY_I: int = 0,
#         AQ=1.6,  # Acquistion time (s)
#         SWH=2000,  # Spectral Width [Hz]
#         SF=202.4765750,  # Spectrometer frequency [Mhz]
#         SFO1=202.4751988,  # Transmitter frequency [Mhz]
#         p0=-40,
#         plot_width=30
# ) -> tuple[NDArray, NDArray]:
#     folder_nums = os.listdir(f"{data_folder}")
#     folder_nums = [str(s) for s in (sorted([int(x) for x in folder_nums]))]  # ordering by increasing number value
#     file_names = [fr"{data_folder}\{fn}" for fn in folder_nums]
#
#     ffts_proc = []
#     freqs = process_combined_data(file_names[0])[0]
#     for fn in file_names:
#         ffts_proc.append(process_combined_data(fn, display_fid=display_fid, display_fft=display_fft)[1])
#
#     ffts_proc = np.array(ffts_proc)
#     print(ffts_proc.shape)
#     ffts_avg = np.mean(ffts_proc, axis=1)
#
#     if display_avg:
#         fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=True, figsize=(8, 12))
#         plot_complex(axs[0], freqs, ffts_proc[DISPLAY_I, 0, :])
#         plot_complex(axs[1], freqs, ffts_proc[DISPLAY_I, 1, :])
#         plot_complex(axs[2], freqs, ffts_proc[DISPLAY_I, 2, :])
#         plot_complex(axs[3], freqs, ffts_avg[DISPLAY_I, :])
#         axs[3].set_title("Averaged:")
#
#         for ax in axs.flat:
#             ax.grid(True)
#             ax.legend()
#             ax.set(xlabel="Freq (Hz)", ylabel="intensity (a.u.)")
#             ax.tick_params(direction='in')
#             ax.label_outer()
#             ax.set_xlim(XLIM_DISPLAY)
#
#         fig.tight_layout()
#         plt.show()
#
#     print(ffts_avg.shape)
#     return freqs, ffts_avg
#


""" Specifying `ns` will now do the phase cycling averaging (assuming they're in the convention Donny has in his 
notebook `ns4_array.ipynb`). """


def temporal_average(
        data_folder: str,
        ns: int | None = None,
        display_fid: bool = False,
        display_fft: bool = False,
        display_avg: bool = False,
        DISPLAY_I: int = 0,
        XLIM_DISPLAY=(-50, 50),
        LEFT_SHIFT_INDEX=63,
        AQ=1.6,  # Acquistion time (s)
        SWH=2000,  # Spectral Width [Hz]
        SF=202.4765750,  # Spectrometer frequency [Mhz]
        SFO1=202.4751988,  # Transmitter frequency [Mhz]
        PHASE_0=-44,
) -> tuple[NDArray, NDArray]:
    folder_nums = os.listdir(f"{data_folder}")
    folder_nums = [str(s) for s in (sorted([int(x) for x in folder_nums]))]  # ordering by increasing number value
    file_names = [fr"{data_folder}\{fn}" for fn in folder_nums]

    ffts_proc = []
    freqs = process_combined_data(file_names[0], PHASE_0=PHASE_0, ACQ_TIME=AQ, XLIM_DISPLAY=XLIM_DISPLAY,
                                  LEFT_SHIFT_INDEX=LEFT_SHIFT_INDEX)[0]
    for fn in file_names:
        ffts_proc.append(
            process_combined_data(fn, display_fid=display_fid, display_fft=display_fft, PHASE_0=PHASE_0,
                                  ACQ_TIME=AQ, XLIM_DISPLAY=XLIM_DISPLAY, LEFT_SHIFT_INDEX=LEFT_SHIFT_INDEX)[1])

    ffts_proc = np.array(ffts_proc)
    if ns is None:  # no phase cycling (the old usual method)
        assert ffts_proc.shape[:2] == (14, 3), "Data is not the expected shape of (14, 3, num_points)"
    else:  # newly added method
        assert ffts_proc.shape[0] == 14 * ns, (f"There should be 14*ns number of folders. Currently has:"
                                               f" {ffts_proc.shape[0]}")
        assert ffts_proc.shape[1] == 3

        ffts_averaged = np.zeros((14, 3, ffts_proc.shape[2]), dtype=ffts_proc.dtype)
        for i in range(ffts_proc.shape[0]):  # 14
            ffts_averaged[i] = np.mean(ffts_proc[i * ns: i * ns + ns], axis=0)
        ffts_proc = ffts_averaged

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

    # print(ffts_avg.shape)
    return freqs, ffts_avg


def process_combined_data(data_folder: str,
                          display_fid: bool = False,
                          display_fft: bool = False,
                          PHASE_0=-40,
                          ACQ_TIME=1.6,
                          XLIM_DISPLAY=(-50, 50),
                          LEFT_SHIFT_INDEX=63) -> tuple[NDArray, NDArray]:
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


def prepare_fid(all_data: NDArray, display: bool = False, ACQ_TIME=1.6, LEFT_SHIFT_INDEX=63) -> NDArray:
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

# DATA_FOLDER = r"C:\Users\lemon\OneDrive - Brown University\CNOT project\MIT\data\2024_12_13\1000_V6_NS4_2D_2ch\3"
# # process_combined_data(DATA_FOLDER, display_fid=True, display_fft=True)
#
# DATA_FOLDER = r"C:\Users\lemon\OneDrive - Brown University\CNOT project\MIT\data\2024_12_13\1000_V6_NS4_2D_2ch"
# temporal_average(DATA_FOLDER, display_avg=True)


def analyze_dataset(all_data):
    # Create a DataFrame with real, imaginary, and time values
    time = np.linspace(0, AQ, len(all_data))
    df = pd.DataFrame({
        "Time": time,
        "Real": all_data.real,
        "Imaginary": all_data.imag
    })
    all_data = ng.proc_base.ps(all_data, p0 + 90,
                                    p1)  # Why is there an extra 90 here? Probably from ph31 convention?
    # Apply exponential apodization
    # apodized_data = ng.proc_base.em(all_data, .0004)
    # *********this function can only take real part and not full complex part. Do we want to edit this way?

    ls_and_apodized_data = ng.proc_base.ls(all_data, 63)
    # Plot the results

    spectrum = ng.proc_base.fft(ls_and_apodized_data)
    n = len(all_data.real)
    freq = np.linspace(-SWH / 2 + center, SWH / 2 + center,
                       n)  # I am not sure how to calcualte the bounds of the FT here. But this mehtod lines up with Bruker.
    return (freq, np.real(spectrum), np.imag(spectrum))


def Donnys_custom_ns4_array_function(data_folder, display_avg=True, DISPLAY_I=0):
    """
    Reads NMR data from a set of Bruker files in `data_folder`,
    processes them (via `analyze_dataset`), then combines & averages
    the results in groups of 4. Finally, returns:

      freqs                -> 1D frequency axis (length = 3200)
      all_spectra_complex  -> shape (14, 3, 3200) array

    Parameters
    ----------
    data_folder : str
        The top-level directory containing your Bruker data folders.
    display_avg : bool, optional
        If True, you might add extra printing or plotting here.
    DISPLAY_I   : int, optional
        A possible index for debugging or additional logic.
    """

    # ---------------------------------------------------------
    # 2) Build file paths. Example: from 119 to 174 inclusive
    # ---------------------------------------------------------
    file_paths = [
        os.path.join(data_folder, f"{i}")
        for i in range(119, 175)  # 119..174
    ]

    # ---------------------------------------------------------
    # 3) Process each path, store results
    #    all_results will collect a structure of shape (N, 9)
    # ---------------------------------------------------------
    all_results = []
    for path in file_paths:
        # Read the data (dictionary, data) = ng.bruker.read(path)
        _, all_data = ng.bruker.read(path)

        # all_data is assumed to contain arrays: array0, array1, array2, ...
        array0, array1, array2, _ = all_data

        # Process each array using your own function
        rho_0 = analyze_dataset(array0)
        rho_1 = analyze_dataset(array1)
        rho_2 = analyze_dataset(array2)

        # Combine processed arrays
        x = rho_0[0]  # frequency axis (assumed consistent)
        real_data = rho_0[1] + rho_1[1] + rho_2[1]
        imag_data = rho_0[2] + rho_1[2] + rho_2[2]

        # Sanity check: ensure consistent lengths
        assert len(x) == len(real_data) == len(imag_data) == 3200, \
            "Array dimension mismatch!"

        # Store: [ freq, real0, real1, real2, imag0, imag1, imag2, sum_real, sum_imag ]
        all_results.append([
            x,
            rho_0[1], rho_1[1], rho_2[1],
            rho_0[2], rho_1[2], rho_2[2],
            real_data, imag_data
        ])

    # Convert list -> NumPy array
    results_array = np.array(all_results)
    # results_array shape is (number_of_paths, 9, 3200)

    # ---------------------------------------------------------
    # 4) Group the data in sets of 4 and average
    #    We end up with 14 groups if we have 56 files total
    # ---------------------------------------------------------
    output = []
    for n in range(14):
        # Indices:  4*n, 4*n+1, 4*n+2, 4*n+3
        # [1] => real0, [2] => real1, [3] => real2
        avg_real0 = (results_array[4 * n + 0][1]
                     + results_array[4 * n + 1][1]
                     + results_array[4 * n + 2][1]
                     + results_array[4 * n + 3][1]) / 4.0

        avg_real1 = (results_array[4 * n + 0][2]
                     + results_array[4 * n + 1][2]
                     + results_array[4 * n + 2][2]
                     + results_array[4 * n + 3][2]) / 4.0

        avg_real2 = (results_array[4 * n + 0][3]
                     + results_array[4 * n + 1][3]
                     + results_array[4 * n + 2][3]
                     + results_array[4 * n + 3][3]) / 4.0

        # Append a row with shape (3, 3200)
        output.append([avg_real0, avg_real1, avg_real2])

    # Convert to NumPy => shape (14, 3, 3200)
    ns4_array = np.array(output)
    # Convert to NumPy => shape (14, 3200)
    ns4_avg = np.mean(ns4_array, axis=1)

    # ---------------------------------------------------------
    # 5) Return the frequency axis and the final 14 x 3 x 3200
    #    array. We assume freq is the same for all files, so we
    #    take it from the first row of results_array.
    # ---------------------------------------------------------
    freqs = results_array[0][0]  # The frequency axis from the first dataset
    all_spectra_complex = ns4_avg +1j*ns4_avg

    # Optionally, do something if display_avg = True or DISPLAY_I != 0:
    if display_avg:
        print(f"Computed average spectra for {len(file_paths)} files.")
    if DISPLAY_I != 0:
        print(f"DISPLAY_I = {DISPLAY_I}; no special logic implemented yet.")

    return freqs, all_spectra_complex
