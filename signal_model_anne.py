"""
Signal processing model using ANNE-PSG data to predict sleep apnea.
"""

import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import neurokit2 as nk
from scipy.signal import find_peaks
from scipy.signal import detrend
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import sortednp as snp

RUN_LOCALLY = True
if RUN_LOCALLY:
    DATA_PATH = "anne-psg toy data"
    OUTPUT_PATH = 'anne-psg toy features'
else:
    DATA_PATH = '/scratch/a/alim/alim/overnight_validation/ANNE-PSG231215'
    OUTPUT_PATH = '/scratch/a/alim/yrichard'

SAMPLING_RATE = 100  # Number of samples per second. Integer
WINDOW_TIME = 10  # In seconds. Integer.


def read_files_in_path(path: str) -> list[str]:
    """
    Read dataset files (location specified by path), returning the i-th file's path as the i-th
    element of the return list.

    The data is structured as many folders, where each folder contains a single ELF file of the same
    name as the folder plus a suffix '-features.edf'
    """
    paths = []

    for entry in Path(path).iterdir():
        if not entry.is_dir():
            continue

        for sub_entry in entry.iterdir():
            paths.append(str(sub_entry))

    return paths


def group_data_into_windows(matrix: np.ndarray, window_size: int) -> np.ndarray:
    """
    Group data of shape (N, D) into non-overlapping windows of a specified size K, returning an
    array of shape (N // K, K, D) AKA (# of windows, window size, D).

    Remaining data that doesn't divide the window size evenly is discarded.
    """
    n = window_size * (matrix.shape[0] // window_size)
    return matrix[:n, :].reshape(n // window_size, window_size, -1)


def preprocess(path: str) -> tuple[np.ndarray, list[str]]:
    """
    Preprocess the EDF file at the specified path into a windowed matrix, returning an array of
    size (# of windows, window size, D) matrix and a list of feature names of length D.

    The following features are sampled at a lower frequency:
        - zeroCrossingRate      - ecgSQI        - ppgSQI        - chestTemp     - limbTemp
        - HRmedian              - HR            - SpO2          - RR            - RRsqi
        - PPGamp                - PPGampmedian
    """
    df = read_raw_edf(path, verbose=False).to_data_frame()
    columns = list(df.columns)  # length = 33
    kept_columns = ['time', 'ecgProcessed', 'ppgFiltered', 'PPGamp', 'sleepstage', 'resp_events']
    kept_indices = np.array([columns.index(x) for x in kept_columns])
    data = np.array(df)[:, kept_indices]

    return data, kept_columns


def order_peaks_troughs(peaks: np.ndarray, troughs: np.ndarray) -> tuple[np.ndarray, bool]:
    """Concatenate an array of ordered indices of peaks and troughs into a (2 x n) array such that
    the final array's elements, ordered smallest (1) to largest (2n), are:
    [1, 3, 5, ..., 2n - 1]
    [2, 4, 6, ..., 2n]
    A boolean is also returned indicating whether the first row contains the peaks (if false, troughs)
    """
    assert abs(len(peaks) - len(troughs)) <= 1, 'The # of peaks and troughs should not differ by >1'

    if len(peaks) > len(troughs):
        assert np.all(peaks[1:] > troughs) and troughs[0] > peaks[0]
        return np.stack([peaks, np.concatenate([troughs, [-1]])]), True

    if len(peaks) < len(troughs):
        assert np.all(troughs[1:] > peaks) and peaks[0] > troughs[0]
        return np.stack([troughs, np.concatenate([peaks, [-1]])]), False

    if np.all(peaks > troughs):
        return np.stack([peaks, troughs]), True

    assert np.all(troughs > peaks), 'Peaks/troughs should interleave each other perfectly'
    return np.stack([troughs, peaks]), False


def filter_extrema(signal: np.ndarray, extrema: np.ndarray) -> np.ndarray:
    """Given a 1D signal and output of order_peaks_troughs, return a mask to filters noisy extrema.

    For each peak, find the closest distance to the neighbouring 2 troughs and 2 peaks.
    Filter out the peak if the minimum trough distance < minimum peak distance.
    Also do this for troughs.

    To prevent double peaks/troughs, we keep the one with the largest magnitude.
    """
    extrema_list = snp.merge(extrema[0], extrema[1])

    diff = np.abs(np.ediff1d(signal[extrema_list]))  # (n - 1)
    diff = np.minimum(diff[1:], diff[:-1])[1:-1]  # (n - 4)

    same = np.abs(signal[extrema_list[2:]] - signal[extrema_list[:-2]])  # (n - 2)
    same = np.minimum(same[2:], same[:-2])  # (n - 4)

    mask = np.ones_like(extrema_list, dtype=bool)
    mask[2:-2] = same <= diff

    # Clear double peaks or double troughs
    mask_left = np.roll(mask, 1)
    mask_right = np.roll(mask, -1)
    i = (mask_left == mask_right) & (mask != mask_left)
    i[0] = False
    i[-1] = False

    i_left_ind = np.where(np.roll(i, -1))[0]
    i_right_ind = np.where(np.roll(i, 1))[0]
    real_extrema = np.argmax([np.abs(signal[extrema_list[i_left_ind]]),
                              np.abs(signal[extrema_list[i_right_ind]])], axis=0)
    mask[i_left_ind[real_extrema == 1]] = False
    mask[i_right_ind[real_extrema == 0]] = False

    mask_left = np.roll(mask, 1)
    mask_right = np.roll(mask, -1)
    i = (mask_left == mask_right) & (mask != mask_left)
    i[0] = False
    i[-1] = False

    #


    # # Find signal amplitude difference between neighbouring extrema
    # vertical = np.abs(signal[extrema[0, 1:-1]] - signal[extrema[1, 1:-1]])          # (n - 2)
    # diagonal = np.abs(signal[extrema[0, 1:]] - signal[extrema[1, :-1]])             # (n - 1)
    # top_horizontal = np.abs(signal[extrema[0, 1:]] - signal[extrema[0, :-1]])       # (n - 1)
    # bottom_horizontal = np.abs(signal[extrema[1, 1:]] - signal[extrema[1, :-1]])    # (n - 1)
    #
    # # Compare distances of extrema to neighbouring extrema of same/different type
    # min_diff_neighbours = np.stack([np.minimum(vertical, diagonal[:-1]),
    #                                 np.minimum(vertical, diagonal[1:])])            # (2, n - 2)
    # min_same_neighbours = np.stack([np.minimum(top_horizontal[1:], top_horizontal[:-1]),
    #                                 np.minimum(bottom_horizontal[1:], bottom_horizontal[:-1])])  # (2, n - 2)
    # mask = np.ones_like(extrema, dtype=bool)
    # mask[:, 1:-1] = min_same_neighbours <= min_diff_neighbours

    # def filter_double_extrema() -> None:
    #     """
    #     Pattern 1:
    #     [T, T]  ->  [T, F]  or  [F, T]
    #     [F, T]      [F, T]      [F, T]
    #
    #     Pattern 2:
    #     [T, F]  ->  [T, F]  or  [T, F]
    #     [T, T]      [F, T]      [T, F]
    #     """
    #     pattern_1 = np.logical_and.reduce([mask[0, :-1], mask[0, 1:], mask[1, 1:], ~mask[1, :-1]])
    #     pattern_2 = np.logical_and.reduce([mask[0, :-1], ~mask[0, 1:], mask[1, 1:], mask[1, :-1]])
    #     pattern_1 = np.where(pattern_1)[0]
    #     pattern_2 = np.where(pattern_2)[0]
    #
    #     i = 0
    #     while len(pattern_1) + len(pattern_2) > 0 and i < 500:
    #         real_peak = np.argmax([signal[extrema[0, pattern_1]], signal[extrema[0, pattern_1 + 1]]], axis=0)
    #         real_trough = np.argmin([signal[extrema[1, pattern_2]], signal[extrema[1, pattern_2 + 1]]], axis=0)
    #
    #         mask[0, pattern_1[np.where(real_peak == 0)[0]]] = False
    #         mask[0, pattern_1[np.where(real_peak == 1)[0]] + 1] = False
    #         mask[1, pattern_2[np.where(real_trough == 0)[0]] + 1] = False
    #         mask[1, pattern_2[np.where(real_trough == 1)[0]]] = False
    #
    #         pattern_1 = np.logical_and.reduce([mask[0, :-1], mask[0, 1:], mask[1, 1:], ~mask[1, :-1]])
    #         pattern_2 = np.logical_and.reduce([mask[0, :-1], ~mask[0, 1:], mask[1, 1:], mask[1, :-1]])
    #         pattern_1 = np.where(pattern_1)[0]
    #         pattern_2 = np.where(pattern_2)[0]
    #         i += 1
    #
    #     if i == 500:
    #         raise ValueError()

    filter_double_extrema()
    return mask


def extract_features(path: str) -> pd.DataFrame:
    """
    Perform feature extraction on a file in a given path.
    """
    data, columns = preprocess(path)  # (N, D)

    signal_columns = np.array([1, 2])
    data[:, signal_columns] = detrend(data[:, signal_columns], axis=0)

    ppg = data[:, 2]
    peaks = find_peaks(ppg)[0]
    troughs = find_peaks(-ppg)[0]

    extrema, peaks_first = order_peaks_troughs(peaks, troughs)
    extrema_mask = filter_extrema(ppg, extrema)
    peaks = extrema[0, extrema_mask[0, :]] if peaks_first else extrema[1, extrema_mask[0, :]]
    troughs = extrema[1, extrema_mask[1, :]] if peaks_first else extrema[0, extrema_mask[0, :]]

    # If minimum distance to 2 other-type neighbours closer than 2 same-type neighbours, flag for deletion
    #
    amplitudes =



def plot_peaks_troughs(signal: np.ndarray, peaks: np.ndarray, troughs: np.ndarray,
                       i: int = 0, j: int = 10000) -> None:
    plt.clf()
    plt.plot(range(i, j), signal[i:j], color='black')

    peaks = peaks[(peaks >= i) & (peaks < j)]
    troughs = troughs[(troughs >= i) & (troughs < j)]
    plt.scatter(peaks, signal[peaks], color='red')
    plt.scatter(troughs, signal[troughs], color='blue')
    plt.show()


def plot_extrema(signal: np.ndarray, extrema: np.ndarray, i: int = 0, j: int = 10000) -> None:
    plt.clf()
    plt.plot(range(i, j), signal[i:j], color='black')

    extrema = extrema[(extrema >= i) & (extrema < j)]
    extrema_odd = extrema[range(0, len(extrema), 2)]
    extrema_even = extrema[range(1, len(extrema), 2)]
    plt.scatter(extrema_odd, signal[extrema_odd], color='red')
    plt.scatter(extrema_even, signal[extrema_even], color='blue')
    plt.show()


if __name__ == '__main__':
    print(f"Started job at {datetime.now()}")

    paths = read_files_in_path(DATA_PATH)
    print(f"{len(paths)} files to process")

    # if RUN_LOCALLY:
    #     dfs = [extract_features(path) for path in tqdm(paths)]
    # else:
    #     cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
    #     pool = mp.Pool(processes=cpus)
    #     dfs = pool.map(extract_features, paths)
    #     pool.close()
    #     pool.join()

    # df = pd.concat(dfs)
    # if not os.path.exists(OUTPUT_PATH):
    #     os.makedirs(OUTPUT_PATH)
    # df.to_csv(f"{OUTPUT_PATH}/anne-psg features.csv", index=False)

    print(f"Finished job at {datetime.now()}")

# function to calculate peak-to-peak distances

# find baseline via awake peak-to-peak distances

# 10-15 second windows

#
