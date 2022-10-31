import numpy as np
import glob
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import find_peaks as fp
from scipy.ndimage import gaussian_filter

FILTER_TYPE = 1  # 0 = gaussian convolution, 1 = savgol filter


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
            print(filename, " accepted")

        except IOError:
            print("Error: ", filename, " directory not found")
        except IndexError:
            print("Error: ", filename, " is empty")

        all_data.append([filename[-15:-13], valid_data])

    return all_data


def find_peaks(data, savgol_parameter, peak_prominence):
    if FILTER_TYPE == 0:
        filtered_data = gaussian_smooth_filter(data[:, 1], sigma=5)
    elif FILTER_TYPE == 1:
        filtered_data = savgol_filter(data[:, 1], savgol_parameter, 2)
    peaks, _ = fp(filtered_data, prominence=peak_prominence)
    return peaks, filtered_data


def filter_peaks(peaks, values, limit):
    return peaks[np.where(values[peaks] > limit)]


def linear_function(x, m, c):
    return m*x + c


def reduced_chi_square(data, param):
    chi_square_total = 0
    for datum in data:
        chi_square_total += (((linear_function(datum[0], *param) -
                               datum[1]) / datum[2]) ** 2)

    return chi_square_total / len(data)


def find_linear_parameters(data):
    try:
        expected, uncertainty = curve_fit(linear_function,
                                          data[:, 0], data[:, 1],
                                          sigma=data[:, 2])
    except RuntimeError:
        print('Scipy.optimize.curve_fit was not able to find the best'
              ' parameters, please run code again and input different'
              ' starting guesses')

    return expected[0], expected[1],\
        np.sqrt(uncertainty[0, 0]), np.sqrt(uncertainty[1, 1])


def gaussian_smooth_filter(y_data, sigma):
    return gaussian_filter(y_data, sigma)


def fit_gaussian(x_data, y_data, axis):
    mu_guess = np.average(x_data)
    std_guess = 100
    param, _ = curve_fit(gaussian_curve, x_data, y_data,
                         p0=[1.5, std_guess, mu_guess])
    # uncertainty = np.sqrt(np.diagonal(cov))
    fitted_data = gaussian_curve(np.linspace(
        np.min(x_data), np.max(x_data)), *param)
    axis.plot(np.linspace(np.min(x_data), np.max(x_data)),
              fitted_data, 'g--')
    return np.max(fitted_data)


def gaussian_curve(x_data, A, sigma, mu):
    exponent = - (x_data - mu) ** 2 / (2 * sigma ** 2)
    return A * np.exp(exponent)


def gaussian_peak(data, mean, axis):
    try:
        param, cov = curve_fit(gaussian_curve, data[:, 0], data[:, 1],
                               p0=[1.5, len(data[:, 0]), mean],
                               maxfev=1000000)
    except ValueError:
        return 0, 0
    axis.plot(np.linspace(np.min(data[:, 0]), np.max(data[:, 0])),
              gaussian_curve(np.linspace(np.min(data[:, 0]), np.max(data[:, 0])), *param), 'g--')
    return param[2], np.sqrt(cov[2, 2])


def distance_conversion(pixel_dist, pixel_err):
    pix_to_m = 465e2
    pix_to_m_err = 11e2

    scrn_len = 74.348e-3
    scrn_err = 3e-6

    wavelength = 632.8e-9

    displ = (wavelength/2) * (scrn_len * pix_to_m / pixel_dist)
    displ_err = (wavelength/2) * np.sqrt((pix_to_m * scrn_err / pixel_dist) ** 2 +
                                         (scrn_len * pix_to_m_err / pixel_dist) ** 2 +
                                         (scrn_len * pix_to_m * pixel_err / (pixel_dist ** 2)) ** 2)

    return displ, displ_err


def find_residual(data, param):
    x_data = data[:, 0]
    y_data = data[:, 1]
    fitted_y_data = linear_function(x_data, *param)

    y_diff = y_data - fitted_y_data

    return y_diff
