# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:42:47 2022

@author: Ryand Yandoc and Alexander Stansfield
"""

from Functions import gaussian_peak, read_data, find_peaks,\
    filter_peaks, find_linear_parameters,\
    fit_gaussian, weighted_arithmetic_mean,\
    distance_conversion, reduced_chi_square,\
    linear_function, find_residual
from Second_Run.Parameters import SAVGOL_FILTER_PARAMETERS_1,\
    SAVGOL_FILTER_PARAMETERS_2, PEAK_PROMINENCE
import numpy as np
import matplotlib.pyplot as plt
import sys

RUN = "Second_Run"


FILENAME_1 = "{0}/Rising/Data/".format(RUN)
FILENAME_2 = "{0}/Decreasing/Data/".format(RUN)
SAVE_FOLDER_1 = "{0}/Rising/Results/".format(RUN)
SAVE_FOLDER_2 = "{0}/Decreasing/Results/".format(RUN)
SAVE_FOLDER_AVERAGES = "{0}/Comparison/".format(RUN)
X_VARIABLE = "Voltage"
Y_VARIABLE = 'Grey Value (Intensity)'


def draw_plot(title, data, savgol_parameter, filename,
              save_folder, peak_prominence, direction):

    visibility = 0

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

    print(("{0}V {1} with savgol parameter of: {2}").format(title, direction,
                                                            savgol_parameter))

    if len(filtered_peaks) > 4 and len(filtered_troughs) > 4:
        I_max = fit_gaussian(
            filtered_peaks, filtered_data[filtered_peaks], axs)
        I_min = fit_gaussian(
            filtered_troughs, filtered_data[filtered_troughs], axs)

        try:
            visibility = (I_max - I_min) / (I_max + I_min)
        except:
            visibility = 0

   # flip the data back
    data[:, 1] = -data[:, 1]

    peak_diff = np.diff(filtered_peaks)
    fringe_spacing_guess = np.average(peak_diff)

    sigmas = []
    mean_peak = []

    # plot a gaussian at each peack using the average fringe spacing as the width
    for peak in filtered_peaks:
        mean, uncertainty = gaussian_peak(data[np.where((
            data[:, 0] < peak + fringe_spacing_guess / 2)
            & (data[:, 0] > peak - fringe_spacing_guess / 2))],
            peak, axs)
        sigmas.append(uncertainty)
        mean_peak.append(mean)

    fringe_spacing = np.diff(mean_peak)
    
    # fringe_spacing_uncertainty = []
    # for i in range(len(mean_peak) - 1):
    #     uncertainty = np.sqrt(sigmas[i]**2 + sigmas[i + 1]**2)
    #     fringe_spacing_uncertainty.append(uncertainty)

    # mean_fringe_spacing, mean_fringe_spacing_sigma = weighted_arithmetic_mean(
    #     fringe_spacing, fringe_spacing_uncertainty)

    # average spacing in pixels
    average_fringe_spacing = np.average(fringe_spacing)
    uncertainty_fringe_spacing = np.std(
        fringe_spacing) / np.sqrt(len(fringe_spacing))

    # pixels to metres
    pix_to_m = 465e2
    uncertainty_pix_to_m = 11e2

    metre_fringe_spacing = average_fringe_spacing / pix_to_m

    uncertainty_metre_fringe_spacing = np.sqrt((1/pix_to_m)**2 * uncertainty_fringe_spacing**2 +
                                               (average_fringe_spacing/(pix_to_m**2))**2 *
                                               uncertainty_pix_to_m**2)

    displacement, displ_err = distance_conversion(average_fringe_spacing,
                                                  uncertainty_fringe_spacing)

    axs.grid()
    axs.set_xlim((np.min(data[:, 0]), np.max(data[:, 0])))

    plt.tight_layout()
    plt.savefig(save_folder + title, dpi=300, transparent=False)
    plt.close()

    return (np.array((int(title), 1 / (metre_fringe_spacing),
                      (1 / (metre_fringe_spacing) ** 2)
                      * uncertainty_metre_fringe_spacing)),
            np.array((int(title), visibility)),
            np.array((int(title), displacement, displ_err)))


def plot_averages(data_1, data_2, y_ax_label, save_folder, save_title,
                  display_chi=True):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(15, 6)

    axs.set_xlabel("Voltage [V]", fontsize=14, fontfamily='times new roman')
    axs.set_ylabel(y_ax_label,
                   fontsize=14, fontfamily='times new roman')
    axs.set_title(FILENAME_1[: -1] + " and " + FILENAME_2[: -1],
                  fontsize=18, fontfamily='times new roman')

    # Non fitted data points
    axs.errorbar(np.hstack((data_1[:, 0][:1], data_1[:, 0][-1:])),
                 np.hstack((data_1[:, 1][:1], data_1[:, 1][-1:])),
                 yerr=np.hstack((data_1[:, 2][:1], data_1[:, 2][-1:])),
                 fmt='kx')
    axs.errorbar(np.hstack((data_2[:, 0][:1], data_2[:, 0][-1:])),
                 np.hstack((data_2[:, 1][:1], data_2[:, 1][-1:])),
                 yerr=np.hstack((data_2[:, 2][:1], data_2[:, 2][-1:])),
                 fmt='kx')
    # fitted data points [2:-2]
    axs.errorbar(data_1[:, 0][1:-1], data_1[:, 1][1:-1],
                 yerr=data_1[:, 2][1:-1], fmt='bx')
    axs.errorbar(data_2[:, 0][1:-1], data_2[:, 1][1:-1],
                 yerr=data_2[:, 2][1:-1], fmt='rx')

    m_1, c_1, sigma_m_1, sigma_c_1 = find_linear_parameters(data_1[1:-1])
    m_2, c_2, sigma_m_2, sigma_c_2 = find_linear_parameters(data_2[1:-1])

    if display_chi:
        chi_1 = reduced_chi_square(data_1[1:-1], (m_1, c_1))
        print("Rising reduced chi squared: ", chi_1)
        chi_2 = reduced_chi_square(data_2[1:-1], (m_2, c_2))
        print("Decreasing reduced chi squared: ", chi_2)

    plt.plot(np.linspace(0, np.max(data_1[:, 0])), m_1*np.linspace(0, np.max(data_1[:, 0])) + c_1, color='blue',
             label="Rising voltage: y =({0:.3g} $\pm$ {1:.3g})x"
                   .format(m_1, sigma_m_1)
                   + " + {0:.3g} $\pm$ {1:.1g}"
                   .format(c_1, sigma_c_1))
    plt.plot(np.linspace(0, np.max(data_1[:, 0])), m_2*np.linspace(0, np.max(data_1[:, 0])) + c_2, color='red',
             label="Decreasing voltage: y =({0:.3g} $\pm$ {1:.1g})x"
             .format(m_2, sigma_m_2)
             + " + {0:.3g} $\pm$ {1:.1g}"
             .format(c_2, sigma_c_2))

    axs.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_folder + save_title,
                dpi=300, transparent=False)
    plt.close()
    return


def residual_plot(data1, data2):
    data1 = data1[1:-1]
    data2 = data2[1:-1]

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(15, 6)
    fig.suptitle(FILENAME_1[: -1] + " and " + FILENAME_2[: -1] + "Residual plot",
                 fontsize=18, fontfamily='times new roman')

    rising_ax = axs[0]
    decreasing_ax = axs[1]
    rising_ax.set_title("Rising voltage",
                        fontsize=18, fontfamily='times new roman')
    decreasing_ax.set_title("Decreasing voltage",
                            fontsize=18, fontfamily='times new roman')

    rising_ax.set_xlabel("Voltage [V]", fontsize=14,
                         fontfamily='times new roman')
    rising_ax.set_ylabel(
        "Deformation [m]", fontsize=14, fontfamily='times new roman')
    decreasing_ax.set_xlabel(
        "Voltage [V]", fontsize=14, fontfamily='times new roman')
    decreasing_ax.set_ylabel(
        "Deformation [m]", fontsize=14, fontfamily='times new roman')

    m_1, c_1, _, _ = find_linear_parameters(data1)
    m_2, c_2, _, _ = find_linear_parameters(data2)

    fitted_y_data1 = find_residual(data1, (m_1, c_1))
    fitted_y_data2 = find_residual(data2, (m_2, c_2))

    rising_ax.errorbar(data1[:, 0], fitted_y_data1, data1[:, 2], fmt='bx')
    rising_ax.axhline(c='b')
    decreasing_ax.errorbar(data2[:, 0], fitted_y_data2, data2[:, 2], fmt='rx')
    decreasing_ax.axhline(c='r')

    rising_ax.grid()
    decreasing_ax.grid()

    plt.tight_layout()
    plt.savefig(SAVE_FOLDER_AVERAGES + "Residuals",
                dpi=300, transparent=False)
    plt.close()

    return


def plot_visibility(data_1, data_2, save_folder, save_title):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(15, 6)

    axs.set_xlabel("Voltage [V]", fontsize=14, fontfamily='times new roman')
    axs.set_ylabel("Visibility", fontsize=14, fontfamily='times new roman')
    axs.set_title(FILENAME_1[: -1] + " and " + FILENAME_2[: -1] + "Visibility",
                  fontsize=18, fontfamily='times new roman')

    filtered_data_1 = np.delete(data_1, np.where(data_1[:, 1] == 0), 0)
    filtered_data_2 = np.delete(data_2, np.where(data_2[:, 1] == 0), 0)

    axs.plot(filtered_data_1[:, 0], filtered_data_1[:, 1], 'b.')
    axs.plot(filtered_data_2[:, 0], filtered_data_2[:, 1], 'r.')

    axs.grid()

    plt.tight_layout()
    plt.savefig(save_folder + save_title,
                dpi=300, transparent=False)
    plt.close()

    return


def main():
    all_data_1 = read_data(FILENAME_1)
    all_data_2 = read_data(FILENAME_2)
    averages_1 = np.empty((0, 3))
    averages_2 = np.empty((0, 3))
    displacement_1 = np.empty((0, 3))
    displacement_2 = np.empty((0, 3))
    visibility_1 = np.empty((0, 2))
    visibility_2 = np.empty((0, 2))

    for data in all_data_1:
        if len(data[1]) > 0:
            avg_data_1, vis_1, displ_1 = draw_plot(data[0], data[1],
                                                   SAVGOL_FILTER_PARAMETERS_1[data[0]],
                                                   FILENAME_1,
                                                   SAVE_FOLDER_1, PEAK_PROMINENCE["Rising"],
                                                   "Rising")
            averages_1 = np.vstack((averages_1, avg_data_1))
            displacement_1 = np.vstack((displacement_1, displ_1))
            visibility_1 = np.vstack((visibility_1, vis_1))

        else:
            print("No (valid) files provided, ending program")

    for data in all_data_2:
        if len(data[1]) > 0:
            avg_data_2, vis_2, displ_2 = draw_plot(data[0], data[1],
                                                   SAVGOL_FILTER_PARAMETERS_2[data[0]],
                                                   FILENAME_2, SAVE_FOLDER_2,
                                                   PEAK_PROMINENCE["Decreasing"],
                                                   "Decreasing")
            averages_2 = np.vstack((averages_2, avg_data_2))
            displacement_2 = np.vstack((displacement_2, displ_2))
            visibility_2 = np.vstack((visibility_2, vis_2))
        else:
            print("No (valid) files provided, ending program")
    plot_averages(np.sort(averages_1, axis=0),
                  np.sort(averages_2, axis=0),
                  "1 / Fringe seperation [$m^{-1}$]",
                  SAVE_FOLDER_AVERAGES,
                  "(Average fringe seperation)^-1 against Voltage")

    plot_averages(np.sort(displacement_1, axis=0),
                  np.sort(displacement_2, axis=0),
                  "PZT Deformation [$m$]",
                  SAVE_FOLDER_AVERAGES,
                  "PZT displacement against Voltage",
                  False)

    residual_plot(np.sort(displacement_1, axis=0),
                  np.sort(displacement_2, axis=0))

    plot_visibility(visibility_1, visibility_2,
                    SAVE_FOLDER_AVERAGES,
                    "Visibility against Voltage")


main()
