# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:42:47 2022

@author: yryan
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks as fp

FILENAME = '/Users/alexstansfield/Desktop/Lab Images/04-09-22 First run Decreasing/Values/'
TITLE = 'BLEH'
SAVE_NAME = ''
X_VARIABLE = 'Distance (Pixels)'
Y_VARIABLE = 'Gray Value (Intensity)'

def read_data(file_name):

    number_of_files = 0
    for filename in glob.glob(file_name + '*.csv'):
        number_of_files += 1

    all_data = []

    for filename in glob.glob(file_name + '*.csv'):
        data = np.array([])
        valid_data = np.empty((0, 2))
        invalid_index = np.array([])
        try:
            data = np.genfromtxt(filename, dtype='float', delimiter=',',
                                skip_header=1)
            nan_index = np.where(np.isnan(data))[0]

            invalid_index = np.unique(np.append(invalid_index, nan_index))
            valid_data = np.delete(data, invalid_index.astype(int), 0)
            print(filename, " accepted")

        except IOError:
            print("Error: ", filename, " directory not found")
        except IndexError:
            print("Error: ", filename," is empty")
        
        all_data.append([valid_data])

    return all_data

def draw_plot(data):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(15, 6)

    axs[0].set_xlabel(X_VARIABLE, fontsize=14, fontfamily='times new roman')
    axs[0].set_ylabel(Y_VARIABLE, fontsize=14, fontfamily='times new roman')
    axs[0].set_title(TITLE, fontsize=18, fontfamily='times new roman')

    axs[0].plot(data[:, 0], data[:, 1], 'k')
    peaks, w = find_peaks(data)
    filtered_peaks = filter_peaks(peaks, w)
    axs[1].plot(data[:, 0], w, 'k')
    axs[1].scatter(filtered_peaks, w[filtered_peaks])

    peak_diff = np.diff(filtered_peaks)

    print(peak_diff)
    print(np.average(peak_diff))

    axs[0].grid()
    axs[1].grid()

    axs[0].set_xlim((np.min(data[:, 0]), np.max(data[:, 0])))
    axs[1].set_xlim((np.min(data[:, 0]), np.max(data[:, 0])))

    plt.tight_layout()
    plt.savefig(TITLE, dpi=300, transparent=False)
    plt.show()

def find_peaks(data):
    w = savgol_filter(data[:, 1], 41, 2)
    peaks, _ = fp(w, prominence=0.1)
    return peaks, w

def filter_peaks(peaks, values):
    return peaks[np.where(values[peaks] > 0.5)]

def main():
    all_data = read_data(FILENAME)

    for data in all_data:
        if len(data) > 0:
            draw_plot(data)
        else:
            print("No (valid) files provided, ending program")

main()