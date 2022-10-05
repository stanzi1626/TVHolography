# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:42:47 2022

@author: Ryand Yandoc and Alexander Stansfield
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks as fp

FILENAME = '/Users/alexstansfield/Desktop/Lab Images/04-09-22 First run Decreasing/Values/'
TITLE = 'BLEH2'
SAVE_NAME = ''
X_VARIABLE = 'Distance (Pixels)'
Y_VARIABLE = 'Gray Value (Intensity)'

SAVGOL_FILTER_PARAMETERS = {"15": 101,
                            "20": 81,
                            "25": 71,
                            "30": 61,
                            "35": 51,
                            "40": 41,
                            "50": 31,
                            "55": 21,
                            "60": 11,
                            "65": 11,
                            "70": 11,}

def read_data(file_name):

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
            # print(filename, " accepted")

        except IOError:
            print("Error: ", filename, " directory not found")
        except IndexError:
            print("Error: ", filename," is empty")
        
        all_data.append([filename[-15:-13], valid_data])

    return all_data

def draw_plot(title, data, savgol_parameter):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(15, 6)

    axs[0].set_xlabel(X_VARIABLE, fontsize=14, fontfamily='times new roman')
    axs[0].set_ylabel(Y_VARIABLE, fontsize=14, fontfamily='times new roman')
    axs[0].set_title(title + "V", fontsize=18, fontfamily='times new roman')
    axs[1].set_title(title + 'V filtered with peaks', fontsize=18, fontfamily='times new roman')

    axs[0].plot(data[:, 0], data[:, 1], 'k')
    peaks, w = find_peaks(data, savgol_parameter)
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
    plt.savefig(title, dpi=300, transparent=False)
    # plt.show()

    return np.array((int(title), np.average(peak_diff)))

def find_peaks(data, savgol_parameter):
    w = savgol_filter(data[:, 1], savgol_parameter, 2)
    peaks, _ = fp(w, prominence=0.1)
    return peaks, w

def filter_peaks(peaks, values):
    return peaks[np.where(values[peaks] > 0.5)]

def plot_averages(data):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(15, 6)

    axs.set_xlabel("Voltage", fontsize=14, fontfamily='times new roman')
    axs.set_ylabel("1 / Fringe seperation in pixels", fontsize=14, fontfamily='times new roman')
    axs.set_title("voltage against average fringe seperation", fontsize=18, fontfamily='times new roman')

    axs.scatter(data[:, 0], 1 / data[:, 1])

    plt.tight_layout()
    plt.savefig("voltage against average fringe seperation", dpi=300, transparent=False)
    return

def main():
    all_data = read_data(FILENAME)
    averages = np.empty((0, 2))

    for data in all_data:
        if len(data[1]) > 0:
            averages = np.vstack((averages, draw_plot(data[0], data[1], SAVGOL_FILTER_PARAMETERS[data[0]])))
        else:
            print("No (valid) files provided, ending program")
    
    plot_averages(averages)

main()