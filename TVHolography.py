# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:42:47 2022

@author: Ryand Yandoc and Alexander Stansfield
"""

import numpy as np
import matplotlib.pyplot as plt

from Parameters import SAVGOL_FILTER_PARAMETERS_1, SAVGOL_FILTER_PARAMETERS_2, PEAK_PROMINENCE
from Functions import read_data, find_peaks, filter_peaks, find_linear_parameters

FILENAME_1 = "2022_10_04 Second Run/Rising/Data/"
FILENAME_2 = "2022_10_04 Second Run/Decreasing/Data/"
SAVE_FOLDER_1 = "2022_10_04 Second Run/Rising/Results/"
SAVE_FOLDER_2 = "2022_10_04 Second Run/Decreasing/Results/"
SAVE_FOLDER_AVERAGES = "2022_10_04 Second Run/Comparison/"
X_VARIABLE = "Voltage"
Y_VARIABLE = 'Grey Value (Intensity)'                          

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
    plt.close()

    return np.array((int(title), 1 / np.average(peak_diff), (1 / (np.average(peak_diff) ** 2)) * np.std(peak_diff) / np.sqrt(len(peak_diff))))

def plot_averages(data_1, data_2, save_folder):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(15, 6)

    axs.set_xlabel("Voltage", fontsize=14, fontfamily='times new roman')
    axs.set_ylabel("1 / Fringe seperation in pixels", fontsize=14, fontfamily='times new roman')
    axs.set_title(FILENAME_1[: -1] + " and " + FILENAME_2[: -1], fontsize=18, fontfamily='times new roman')

    axs.errorbar(data_1[:, 0], data_1[:, 1], yerr = data_1[:, 2], fmt = 'kx')
    axs.errorbar(data_2[:, 0], data_2[:, 1], yerr = data_2[:, 2], fmt = 'rx')

    m_1, c_1, sigma_m_1, sigma_c_1 = find_linear_parameters(data_1)
    m_2, c_2, sigma_m_2, sigma_c_2 = find_linear_parameters(data_2)

    plt.plot(np.linspace(0, 55), m_1*np.linspace(0, 55) + c_1, color = 'black')
    plt.plot(np.linspace(0, 55), m_2*np.linspace(0, 55) + c_2, color = 'red')

    axs.grid()

    plt.tight_layout()
    plt.savefig(save_folder + "Voltage against (average fringe seperation)^-1", dpi=300, transparent=False)
    plt.close()
    return

def main():
    all_data_1 = read_data(FILENAME_1)
    all_data_2 = read_data(FILENAME_2)
    averages_1 = np.empty((0, 3))
    averages_2 = np.empty((0, 3))

    for data in all_data_1:
        if len(data[1]) > 0:
            averages_1 = np.vstack((averages_1, draw_plot(data[0], data[1], SAVGOL_FILTER_PARAMETERS_1[data[0]], FILENAME_1, SAVE_FOLDER_1, PEAK_PROMINENCE["Rising"])))
        else:
            print("No (valid) files provided, ending program")
    
    for data in all_data_2:
        if len(data[1]) > 0:
            averages_2 = np.vstack((averages_2, draw_plot(data[0], data[1], SAVGOL_FILTER_PARAMETERS_2[data[0]], FILENAME_2, SAVE_FOLDER_2, PEAK_PROMINENCE["Decreasing"])))
        else:
            print("No (valid) files provided, ending program")
    
    plot_averages(np.sort(averages_1, axis = 0)[2:-2], np.sort(averages_2, axis=0)[2:-2], SAVE_FOLDER_AVERAGES)

main()