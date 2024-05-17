"""
Feature Extraction for ANNE-PSG Data.

With vectorization since the previous one was extremely slow.

Notes:
    - Features are not normalized; it is the responsibility of training code to scale the data
    - Windows are fully disjoint because I don't know how to write code to efficiently do otherwise

Some acronyms:
ECG     electrocardiogram signal of heart
PPG     photoplethysmogram signal of blood volume
SQI     signal quality index
HR      heart rate
RR      respiratory (i.e. breathing) rate
SpO2    oxygen saturation
PAT     pulse arrival time
amp     amplitude
"""

import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import neurokit2 as nk
from scipy.signal import detrend
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt

RUN_LOCALLY = False
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

    Removed:
        - RRsqi             identical to RR
        - PPGampmedian      identical to PPGamp
        - footpeakmedian    identical to footpeak
    """
    df = read_raw_edf(path, verbose=False).to_data_frame()
    columns = list(df.columns)  # length = 33
    kept_columns = ['time', 'zeroCrossingRate', 'ecgRaw', 'ecgProcessed', 'ecgSQI', 'ppgRaw',
                    'ppgFiltered', 'ppgSQI', 'chestTemp', 'limbTemp', 'x', 'y', 'z', 'x_detrend',
                    'y_detrend', 'z_detrend', 'PAT', 'PATmedian', 'PATdetrend', 'footpeak',
                    'HRmedian', 'HR', 'SpO2', 'RR', 'PPGamp', 'sleepstage', 'arousal_events',
                    'resp_events', 'PLM_events', 'SpO2_events']
    kept_indices = np.array([columns.index(x) for x in kept_columns])
    data = np.array(df)[:, kept_indices]

    return data, kept_columns


def extract_features(path: str) -> pd.DataFrame:
    """
    Perform feature extraction on a file in a given path.
    """
    data, columns = preprocess(path)  # (N, D)
    if data.shape[0] == 0:
        return pd.DataFrame()

    # Detrend signals
    signal_columns = np.array([2, 3, 5, 6])
    data[:, signal_columns] = detrend(data[:, signal_columns], axis=0)

    # Group data into time windows of size K
    # Then remove batches with sleepstage == 0 (awake)
    data = group_data_into_windows(data, SAMPLING_RATE * WINDOW_TIME)  # (N // K, K, D)
    data = data[np.all(data[:, :, -5] != 0, axis=1)]
    if data.shape[0] == 0:
        return pd.DataFrame()

    # Reduce more features by getting average + standard deviation
    # non_stats_columns = np.array([0, -5, -4, -3, -2, -1])
    stats_columns = np.array(range(1, 24))
    means = np.mean(data[:, :, stats_columns], axis=1)
    stdevs = np.std(data[:, :, stats_columns], axis=1)

    # Count the number of peaks between R-peaks in ECG
    try:
        sos = scipy.signal.butter(N=2, Wn=[0.5, 40], btype='band', fs=SAMPLING_RATE, output='sos')
        filtered = scipy.signal.sosfilt(sos, detrend(data[:, :, 2], axis=1))
        r_peaks = [nk.ecg_peaks(signal, SAMPLING_RATE)[1]['ECG_R_Peaks'] for signal in filtered]
        distance_r_peaks = np.array([[np.mean(np.ediff1d(x)) / SAMPLING_RATE] for x in r_peaks])
        if np.any(np.isnan(distance_r_peaks)):
            print(f"=====\nError processing {path}\nNAN peaks\n=====\n")
            return pd.DataFrame()
    except Exception as e:
        print(f"=====\nError processing {path}\n{e}\n=====\n")
        return pd.DataFrame()

    # Create labels
    label = np.zeros((len(data), 1))
    label[np.any(data[:, :, -5] == 5e6, axis=1), :] = 2  # rem-sleep non-event
    label[np.any(data[:, :, -3] > 0, axis=1), :] = 1  # respiratory event

    # Create dataframe
    features = np.concatenate([label, means, stdevs, distance_r_peaks], axis=1)
    feature_names = [columns[x] for x in stats_columns]
    feature_names = ['label'] + [f'mean_{x}' for x in feature_names] + \
                    [f'stdev_{x}' for x in feature_names] + ['distance_between_r_peaks']
    df = pd.DataFrame(features, columns=feature_names)

    filename = path[max(path.rfind('/'), path.rfind('\\')) + 1:path.find('.edf')]
    df['id'] = filename
    # if not os.path.exists(OUTPUT_PATH):
    #     os.makedirs(OUTPUT_PATH)
    # df.to_csv(f"{OUTPUT_PATH}/{filename}.csv", index=False)
    return df


def inspect_feature(arr: np.ndarray, columns: list[str], i: int, amount: int = 5000) -> None:
    print(columns[i])
    plt.clf()
    plt.plot(arr[:amount, i])
    plt.show()


if __name__ == '__main__':
    print(f"Started job at {datetime.now()}")

    paths = read_files_in_path(DATA_PATH)
    print(f"{len(paths)} files to process")

    if RUN_LOCALLY:
        dfs = [extract_features(path) for path in tqdm(paths)]
    else:
        cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
        pool = mp.Pool(processes=cpus)
        dfs = pool.map(extract_features, paths)
        pool.close()
        pool.join()

    df = pd.concat(dfs)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    df.to_csv(f"{OUTPUT_PATH}/anne-psg features.csv", index=False)

    print(f"Finished job at {datetime.now()}")
