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

from Parameters import SAVGOL_FILTER_PARAMETERS_1, SAVGOL_FILTER_PARAMETERS_2

FILENAME_1 = "2022_10_04 Second Run/Rising/Data/"
FILENAME_2 = "2022_10_04 Second Run/Decreasing/Data/"
SAVE_FOLDER_1 = "2022_10_04 Second Run/Rising/Results/"
SAVE_FOLDER_2 = "2022_10_04 Second Run/Decreasing/Results/"
SAVE_FOLDER_AVERAGES = "2022_10_04 Second Run/Comparison/"
X_VARIABLE = "Voltage"
Y_VARIABLE = 'Gray Value (Intensity)'                          

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

def draw_plot(title, data, savgol_parameter, filename, save_folder, peak_prominence):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(15, 6)

    axs[0].set_xlabel("Distance (Pixels)", fontsize=14, fontfamily='times new roman')
    axs[0].set_ylabel(Y_VARIABLE, fontsize=14, fontfamily='times new roman')
    axs[0].set_title(filename[: -1] + "-" + title + "V", fontsize=18, fontfamily='times new roman')
    axs[1].set_title(title + 'V filtered with peaks', fontsize=18, fontfamily='times new roman')

    axs[0].plot(data[:, 0], data[:, 1], 'k')
    peaks, w = find_peaks(data, savgol_parameter, peak_prominence)
    filtered_peaks = filter_peaks(peaks, w)
    axs[1].plot(data[:, 0], w, 'k')
    axs[1].scatter(filtered_peaks, w[filtered_peaks])

    peak_diff = np.diff(filtered_peaks)

    # print(peak_diff)
    # print(np.average(peak_diff))
    # print(np.std(peak_diff) / np.sqrt(len(peak_diff)))

    axs[0].grid()
    axs[1].grid()

    axs[0].set_xlim((np.min(data[:, 0]), np.max(data[:, 0])))
    axs[1].set_xlim((np.min(data[:, 0]), np.max(data[:, 0])))

    plt.tight_layout()
    # plt.savefig(save_folder + title, dpi=300, transparent=False)

    return np.array((int(title), np.average(peak_diff), np.std(peak_diff) / np.sqrt(len(peak_diff))))

def find_peaks(data, savgol_parameter, peak_prominence):
    w = savgol_filter(data[:, 1], savgol_parameter, 2)
    peaks, _ = fp(w, prominence = peak_prominence) #0.02 for 2nd run decreasing, 0.1 for 2nd run increasing
    return peaks, w

def filter_peaks(peaks, values):
    return peaks[np.where(values[peaks] > 0.5)]

def plot_averages(data_1, data_2, save_folder):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(15, 6)

    axs.set_xlabel("Voltage", fontsize=14, fontfamily='times new roman')
    axs.set_ylabel("1 / Fringe seperation in pixels", fontsize=14, fontfamily='times new roman')
    axs.set_title(FILENAME_1[: -1] + " and " + FILENAME_2[: -1], fontsize=18, fontfamily='times new roman')

    axs.errorbar(data_1[:, 0], 1 / data_1[:, 1], yerr = (1 / data_1[:, 1]**2) * data_1[:, 2], fmt = 'kx')
    axs.errorbar(data_2[:, 0], 1 / data_2[:, 1], yerr = (1 / data_2[:, 1]**2) * data_2[:, 2], fmt = 'rx')

    axs.grid()

    plt.tight_layout()
    plt.savefig(save_folder + "Voltage against (average fringe seperation)^-1", dpi=300, transparent=False)
    return

def main():
    all_data_1 = read_data(FILENAME_1)
    all_data_2 = read_data(FILENAME_2)
    averages_1 = np.empty((0, 3))
    averages_2 = np.empty((0, 3))

    for data in all_data_1:
        if len(data[1]) > 0:
            averages_1 = np.vstack((averages_1, draw_plot(data[0], data[1], SAVGOL_FILTER_PARAMETERS_1[data[0]], FILENAME_1, SAVE_FOLDER_1, 0.1)))
        else:
            print("No (valid) files provided, ending program")
    
    for data in all_data_2:
        if len(data[1]) > 0:
            averages_2 = np.vstack((averages_2, draw_plot(data[0], data[1], SAVGOL_FILTER_PARAMETERS_2[data[0]], FILENAME_2, SAVE_FOLDER_2, 0.02)))
        else:
            print("No (valid) files provided, ending program")
    
    plot_averages(averages_1, averages_2, SAVE_FOLDER_AVERAGES)

main()