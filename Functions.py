import numpy as np
import glob
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import find_peaks as fp

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

def find_peaks(data, savgol_parameter, peak_prominence):
    w = savgol_filter(data[:, 1], savgol_parameter, 2)
    peaks, _ = fp(w, prominence = peak_prominence)
    return peaks, w

def filter_peaks(peaks, values):
    return peaks[np.where(values[peaks] > 0.5)]

def linear_function(x, m, c):
    return m*x + c

def find_linear_parameters(data):
    try:
        expected, uncertainty = curve_fit(linear_function,\
                                data[:, 0], data[:, 1], sigma = data[:, 2]) #
    except RuntimeError:
        print('Scipy.optimize.curve_fit was not able to find the best'
              ' parameters, please run code again and input different'
              ' starting guesses')

    return expected[0], expected[1], np.sqrt(uncertainty[0, 0]), np.sqrt(uncertainty[1, 1])