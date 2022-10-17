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

        all_data.append([filename[-8:-6], valid_data])

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



def red_chi_square(data, smoothed_y):
    chi_square_total = np.sum(((data - smoothed_y)**2) / data)

    return chi_square_total / len(data)


def find_closest(dataset, target, amount=1):
    diff_dict = {}
    diff = np.abs(target - dataset)
    min_val_pos = np.array([])
    
    for index, val in enumerate(diff):
        diff_dict[val] = index
        
    diff_sort = dict(sorted(diff_dict.items()))
    
    for item in diff_sort:
        min_val_pos = np.append(min_val_pos, diff_sort[item])
        
    return (np.take(dataset, min_val_pos[:amount].astype(int)), 
            min_val_pos[:amount])


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    

def optimize_savgol(data, savgol_0, peak_prominence, axs):
    chi_lis = np.array([])
    #abs_list = np.array([])
    range_values = range(5, 351, 2)
    for svg in range_values:
        peaks, w = find_peaks(data, svg, peak_prominence)
        chi_lis = np.append(chi_lis, red_chi_square(data[:, 1], w))
        # abs_list = np.append(abs_list, np.sum(np.abs(data[:, 1] - w)))
        # axs.plot(data[:, 0], w)


    # print(chi_lis)
    # print(abs_list)
    closest, closest_index = find_closest(chi_lis, chi_lis[np.where(range_values==find_nearest(range_values, savgol_0))], 5)
    best_savgol = np.take(range_values, closest_index.astype(int))

    for savgol_param in best_savgol:
        best_data = savgol_filter(data[:, 1], savgol_param, 2)
        axs.plot(data[:, 0], best_data)

    return best_savgol
