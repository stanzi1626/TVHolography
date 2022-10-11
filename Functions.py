import numpy as np
import glob
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import find_peaks as fp
from scipy.ndimage import gaussian_filter

FILTER_TYPE = 1 # 0 = gaussian convolution, 1 = savgol filter


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
            print("Error: ", filename, " is empty")

        all_data.append([filename[-15:-13], valid_data])

    return all_data


def find_peaks(data, savgol_parameter, peak_prominence):
    if FILTER_TYPE==0:
        filtered_data=gaussian_smooth_filter(data[:, 1], sigma=5)
    elif FILTER_TYPE==1:
        filtered_data = savgol_filter(data[:, 1], savgol_parameter, 2)
    peaks, _ = fp(filtered_data, prominence=peak_prominence)
    return peaks, filtered_data


def filter_peaks(peaks, values, limit):
    return peaks[np.where(values[peaks] > limit)]


def linear_function(x, m, c):
    return m*x + c


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

    axis.plot(np.linspace(np.min(x_data), np.max(x_data)),
              gaussian_curve(np.linspace(np.min(x_data), np.max(x_data)), *param), 'g--')
    return

def gaussian_curve(x_data, A, sigma, mu):
    exponent = - (x_data - mu) ** 2 / (2 * sigma ** 2)
    return A * np.exp(exponent)
