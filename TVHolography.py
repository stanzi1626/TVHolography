# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:42:47 2022

@author: Ryand Yandoc and Alexander Stansfield
"""

import numpy as np
import matplotlib.pyplot as plt

from Parameters import SAVGOL_FILTER_PARAMETERS_1,\
        SAVGOL_FILTER_PARAMETERS_2, PEAK_PROMINENCE
from Functions import read_data, find_peaks,\
        filter_peaks, find_linear_parameters,\
        fit_gaussian
from scipy.optimize import curve_fit

FILENAME_1 = "2022_10_04 Second Run/Rising/Data/"
FILENAME_2 = "2022_10_04 Second Run/Decreasing/Data/"
SAVE_FOLDER_1 = "2022_10_04 Second Run/Rising/Results/"
SAVE_FOLDER_2 = "2022_10_04 Second Run/Decreasing/Results/"
SAVE_FOLDER_AVERAGES = "2022_10_04 Second Run/Comparison/"
X_VARIABLE = "Voltage"
Y_VARIABLE = 'Grey Value (Intensity)'


def red_chi_square(data, smoothed_y):
    chi_square_total = np.sum(((data[:, 1] - smoothed_y)**2) / data[:, 1])

    return chi_square_total / len(data)

def optimize_savgol(data, savgol_0, peak_prominence, axs):
    chi_lis = np.array([])
    for svg in range(11, savgol_0*10, 10):
        # print(svg)
        svg = 151
        peaks, w = find_peaks(data, svg, peak_prominence)
        chi_lis = np.append(chi_lis, red_chi_square(data, w))
        axs.plot(data[:, 0], w)


    print(chi_lis)
    band = 0.5
    best_chi_pos = np.where(np.logical_and(1-band < chi_lis,chi_lis < 1+band))
    best_chi = chi_lis[best_chi_pos]

    return best_chi

def draw_plot(title, data, savgol_parameter, filename,
              save_folder, peak_prominence):
    print("{0} V with default savgol param: {1}".format(title,
                                                        savgol_parameter))

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(15, 6)

    axs.set_xlabel("Distance (Pixels)", fontsize=14,
                      fontfamily='times new roman')
    axs.set_ylabel(Y_VARIABLE, fontsize=14, fontfamily='times new roman')
    axs.set_title(filename[: -1] + "-" + title + "V", fontsize=18,
                     fontfamily='times new roman')

    axs.plot(data[:, 0], data[:, 1], 'k')
    peaks, filtered_data = find_peaks(data, savgol_parameter, peak_prominence)
    flipped_data = data
    flipped_data[:, 1] = -data[:, 1]
    troughs, _ = find_peaks(flipped_data, savgol_parameter, peak_prominence)
    filtered_peaks = filter_peaks(peaks, filtered_data, 0.5)
    filtered_troughs = filter_peaks(troughs, filtered_data, 0.3)

    axs.plot(data[:, 0], filtered_data, 'r--')

    axs.scatter(filtered_peaks, filtered_data[filtered_peaks], c='b', s=50)
    axs.scatter(filtered_troughs, filtered_data[filtered_troughs], c='b', s=50)

    if len(filtered_peaks) > 3 or len(filtered_troughs) > 3:
        fit_gaussian(filtered_peaks, filtered_data[filtered_peaks], axs)
        fit_gaussian(filtered_troughs, filtered_data[filtered_troughs], axs)

    peak_diff = np.diff(filtered_peaks)

    print(optimize_savgol(data, savgol_parameter, peak_prominence, axs))

    # print(peak_diff)
    # print(np.average(peak_diff))
    # print(np.std(peak_diff) / np.sqrt(len(peak_diff)))

    axs.grid()

    axs.set_xlim((np.min(data[:, 0]), np.max(data[:, 0])))

    plt.tight_layout()
    plt.savefig(save_folder + title, dpi=300, transparent=False)
    plt.close()
    # plt.show()

    return np.array((int(title), 1 / np.average(peak_diff),
                    (1 / (np.average(peak_diff) ** 2))
                    * np.std(peak_diff) / np.sqrt(len(peak_diff))))

def plot_averages(data_1, data_2, save_folder):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(15, 6)

    axs.set_xlabel("Voltage", fontsize=14, fontfamily='times new roman')
    axs.set_ylabel("1 / Fringe seperation in pixels",
                   fontsize=14, fontfamily='times new roman')
    axs.set_title(FILENAME_1[: -1] + " and " + FILENAME_2[: -1],
                  fontsize=18, fontfamily='times new roman')

    # Non fitted data points
    axs.errorbar(np.hstack((data_1[:, 0][:2], data_1[:, 0][-2:])),
                 np.hstack((data_1[:, 1][:2], data_1[:, 1][-2:])),
                 yerr=np.hstack((data_1[:, 2][:2], data_1[:, 2][-2:])),
                 fmt='kx')
    axs.errorbar(np.hstack((data_2[:, 0][:2], data_2[:, 0][-2:])),
                 np.hstack((data_2[:, 1][:2], data_2[:, 1][-2:])),
                 yerr=np.hstack((data_2[:, 2][:2], data_2[:, 2][-2:])),
                 fmt='kx')
    # fitted data points [2:-2]
    axs.errorbar(data_1[:, 0][2:-2], data_1[:, 1][2:-2],
                 yerr=data_1[:, 2][2:-2], fmt='bx')
    axs.errorbar(data_2[:, 0][2:-2], data_2[:, 1][2:-2],
                 yerr=data_2[:, 2][2:-2], fmt='rx')

    m_1, c_1, sigma_m_1, sigma_c_1 = find_linear_parameters(data_1[2:-2])
    m_2, c_2, sigma_m_2, sigma_c_2 = find_linear_parameters(data_2[2:-2])

    plt.plot(np.linspace(0, 55), m_1*np.linspace(0, 55) + c_1, color='blue',
             label="Rising voltage: y =({0:.3g} $\pm$ {1:.3g})x"\
                   .format(m_1, sigma_m_1)
                   +" + {0:.1g} $\pm$ {1:.1g}"\
                   .format(c_1, sigma_c_1))
    plt.plot(np.linspace(0, 55), m_2*np.linspace(0, 55) + c_2, color='red',
             label="Decreasing voltage: y =({0:.3g} $\pm$ {1:.1g})x"\
                    .format(m_2, sigma_m_2)
                    +" + {0:.3g} $\pm$ {1:.1g}"\
                    .format(c_2, sigma_c_2))

    axs.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_folder + "Voltage against (average fringe seperation)^-1",
                dpi=300, transparent=False)
    plt.close()
    return


def main():
    all_data_1 = read_data(FILENAME_1)
    # all_data_2 = read_data(FILENAME_2)
    averages_1 = np.empty((0, 3))
    # averages_2 = np.empty((0, 3))

    for data in all_data_1[:1]:
        if len(data[1]) > 0:
            averages_1 = np.vstack((averages_1, draw_plot(data[0], data[1],
                                    SAVGOL_FILTER_PARAMETERS_1[data[0]],
                                    FILENAME_1,
                                    SAVE_FOLDER_1, PEAK_PROMINENCE["Rising"])))
        else:
            print("No (valid) files provided, ending program")

    # for data in all_data_2:
    #     if len(data[1]) > 0:
    #         averages_2 = np.vstack((averages_2, draw_plot(data[0], data[1],
    #                                 SAVGOL_FILTER_PARAMETERS_2[data[0]],
    #                                 FILENAME_2, SAVE_FOLDER_2,
    #                                 PEAK_PROMINENCE["Decreasing"])))
    #     else:
    #         print("No (valid) files provided, ending program")

    # plot_averages(np.sort(averages_1, axis=0),
    #               np.sort(averages_2, axis=0), SAVE_FOLDER_AVERAGES)


main()
