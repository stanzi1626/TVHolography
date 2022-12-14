# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:42:47 2022

@author: Ryand Yandoc and Alexander Stansfield
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

from Parameters import SAVGOL_FILTER_PARAMETERS_1,\
        SAVGOL_FILTER_PARAMETERS_2, PEAK_PROMINENCE
from Functions import extract_index, read_data, find_peaks,\
        filter_peaks, find_linear_parameters,\
        fit_gaussian, optimize_savgol, red_chi_square

FILENAME_1 = "2022_10_13 Third Run/Values-Filtered/Rising/Data/"
FILENAME_2 = "2022_10_13 Third Run/Values-Filtered/Decreasing/Data/"
SAVE_FOLDER_1 = "2022_10_13 Third Run/Values-Filtered/Rising/Results/"
SAVE_FOLDER_2 = "2022_10_13 Third Run/Values-Filtered/Decreasing/Results/"
SAVE_FOLDER_AVERAGES = "2022_10_13 Third Run/Values-Filtered/Comparison/"
X_VARIABLE = "Voltage"
Y_VARIABLE = 'Grey Value (Intensity)'

def draw_plot(title, data, savgol_parameter, filename,
              save_folder, peak_prominence, direction):
    print("{0} V with default savgol param: {1}".format(title + direction,
                                                        savgol_parameter))

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(15, 6)

    axs.set_xlabel("Distance (Pixels)", fontsize=14,
                      fontfamily='times new roman')
    axs.set_ylabel(Y_VARIABLE, fontsize=14, fontfamily='times new roman')
    axs.set_title(filename[: -1] + "-" + title + "V", fontsize=18,
                     fontfamily='times new roman')

    axs.plot(data[:, 0], data[:, 1], 'k')
    

    best_savgol_parameters = optimize_savgol(data[1000:2750], savgol_parameter,
                                             peak_prominence, axs)
    print(best_savgol_parameters)

    total_x_peak_data = []
    total_y_peak_data = []

    for sav_param in best_savgol_parameters:
        peaks, filtered_data = find_peaks(data, sav_param, peak_prominence)
        # flipped_data = copy.deepcopy(data)
        # flipped_data[:, 1] = -flipped_data[:, 1] 
        # troughs, _ = find_peaks(flipped_data, sav_param, peak_prominence)
        filtered_peaks = filter_peaks(peaks, filtered_data, 1)
        total_x_peak_data.append(filtered_peaks)
        total_y_peak_data.append(filtered_data[filtered_peaks])
        # filtered_troughs = filter_peaks(troughs, filtered_data, 0.3)
        axs.plot(filtered_peaks, filtered_data[filtered_peaks], '.')
    
    peak_x_averages = []
    peak_x_sigmas = []
    peak_x_range = []
    peak_y_averages = []
    peak_y_sigmas = []
    peak_y_range = []

    total_x_peak_data_filtered = np.zeros((0, len(best_savgol_parameters)))
    total_y_peak_data_filtered = np.zeros((0, len(best_savgol_parameters)))

    #filter out additional peaks found from different savgol parameters
    for i in range(len(total_x_peak_data[0])):
        while True:
            check = 0
            try:
                x_values = [item[i] for item in total_x_peak_data]
            except IndexError:
                break
            y_values = [item[i] for item in total_y_peak_data]
            mean = np.mean(x_values)
            std = np.std(x_values)
            for j, x in enumerate(x_values):
                if x > (mean + 3 * std) or x < (mean - 3 * std):
                    x_values.pop(j)
                    y_values.pop(j)
                    break
                else:
                    check += 1
            if check==len(best_savgol_parameters):
                total_x_peak_data_filtered = np.vstack((total_x_peak_data_filtered
                                                        ,x_values))
                total_y_peak_data_filtered = np.vstack((total_y_peak_data_filtered
                                                        ,y_values))
                break


    for peak_x in total_x_peak_data_filtered:
        peak_x_averages.append(np.mean(peak_x))
        peak_x_sigmas.append(np.std(peak_x))
        peak_x_range.append((np.max(peak_x) - np.min(peak_x)) / 2)
    for peak_y in total_y_peak_data_filtered:
        peak_y_averages.append(np.mean(peak_y))
        peak_y_sigmas.append(np.std(peak_y))
        peak_y_range.append((np.max(peak_y) - np.min(peak_y)) / 2)
    

    axs.errorbar(peak_x_averages, peak_y_averages, xerr = peak_x_range, yerr=peak_y_sigmas, fmt='bx')

    peak_diffs = np.diff(peak_x_averages)

    # axs.scatter(filtered_peaks, filtered_data[filtered_peaks], c='b', s=50)
    # axs.scatter(filtered_troughs, filtered_data[filtered_troughs], c='b', s=50)

    # if len(filtered_peaks) > 3 or len(filtered_troughs) > 3:
    #     fit_gaussian(filtered_peaks, filtered_data[filtered_peaks], axs)
    #     fit_gaussian(filtered_troughs, filtered_data[filtered_troughs], axs)

    axs.grid()

    axs.set_xlim((np.min(data[:, 0]), np.max(data[:, 0])))

    plt.tight_layout()
    plt.savefig(save_folder + title, dpi=300, transparent=False)
    plt.close()
    # plt.show()

    return np.array((int(title), 1 / np.average(peak_diffs),
                    (1 / (np.average(peak_diffs) ** 2))
                    * np.std(peak_diffs) / np.sqrt(len(peak_diffs))))


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

    rising_data = data_1[:, 1][2:-2]
    decreasing_data = data_2[:, 1][2:-2]
    reduced_chi_sqaure_rising = red_chi_square(rising_data,
                                               m_1*np.linspace(np.min(rising_data),
                                               np.max(rising_data),
                                               num=len(rising_data)) + c_1)
    reduced_chi_sqaure_decreasing = red_chi_square(decreasing_data,
                                               m_2*np.linspace(np.min(decreasing_data),
                                               np.max(decreasing_data),
                                               num=len(decreasing_data)) + c_2)
    print("The reduced chi-sqaure for rising: {0:.3g}".format(reduced_chi_sqaure_rising))
    print("The reduced chi-sqaure for decreasing: {0:.3g}".format(reduced_chi_sqaure_decreasing))

    axs.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_folder + "Voltage against (average fringe seperation)^-1",
                dpi=300, transparent=False)
    plt.close()
    return


def main():
    all_data_1 = read_data(FILENAME_1)
    all_data_2 = read_data(FILENAME_2)
    averages_1 = np.empty((0, 3))
    averages_2 = np.empty((0, 3))

    for data in all_data_1:
        if len(data[1]) > 0:
            averages_1 = np.vstack((averages_1, draw_plot(data[0], data[1],
                                    SAVGOL_FILTER_PARAMETERS_1[data[0]],
                                    FILENAME_1,
                                    SAVE_FOLDER_1, PEAK_PROMINENCE["Rising"],
                                    "Rising")))
        else:
            print("No (valid) files provided, ending program")

    for data in all_data_2:
        if len(data[1]) > 0:
            averages_2 = np.vstack((averages_2, draw_plot(data[0], data[1],
                                    SAVGOL_FILTER_PARAMETERS_2[data[0]],
                                    FILENAME_2, SAVE_FOLDER_2,
                                    PEAK_PROMINENCE["Decreasing"],
                                    "Decreasing")))
        else:
            print("No (valid) files provided, ending program")

    plot_averages(np.sort(averages_1, axis=0),
                  np.sort(averages_2, axis=0), SAVE_FOLDER_AVERAGES)


main()
