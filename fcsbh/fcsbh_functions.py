import glob
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from lmfit import Model
import re
matplotlib.style.use("seaborn-colorblind")


# def columns(filename):
#     b = np.genfromtxt(filename, delimiter="\t", skip_header=50, max_rows=186)
#     return b.shape[1]


def return_average_count(filename):
    """This function opens ascii files and reads the average count from the file data
    """

    b = open(filename, 'r')
    data = b.readlines()
    b.close()

    """This section is to find the average count as saved in the file
    """
    for k in data:
        if k.startswith('Mean countrate [kHz]'):
            for t in k.split():
                try:
                    mean_count = float(t)
                except ValueError:
                    pass
            break

    """This section is to slice the correlation and raw counts
    """
    counter = 0
    raw_start = 1000000
    for k in data:
        counter += 1
        if k.startswith('( Time[µs]  FCS_value )'):
            cor_start = counter
        if k.startswith('*BLOCK 3 Mcs  ( Time[s]  No_of_photons )'):
            cor_end = counter - 3
            raw_start = counter
        if k.startswith('*END') and counter > raw_start:
            break
        raw_end = counter
    # creating the correlation data var = correlation
    corr1 = data[cor_start:cor_end]
    corr2 = []
    for k in corr1:
        for m in k.split():
            corr2.append(m)
    corr3 = np.asarray(corr2)
    correlation = corr3.reshape(int(len(corr3)/2),2)

    raw1 = data[raw_start:raw_end]
    raw2 = []
    for k in raw1:
        for m in k.split():
            raw2.append(m)
    raw3 = np.asarray(raw2)
    raw_count = raw3.reshape(int(len(raw3)/2),2)

    print(correlation[:,0])
    print(correlation[:,1])
    # plt.plot(correlation[:,0],correlation[:,1])
    # plt.plot(raw_count[:,0],raw_count[:,1])
    # plt.show()
    return mean_count


# def fit_model(t, y):
#     gmodel = Model(correlation)
#     gmodel.set_param_hint('g0', value=0.1)
#     gmodel.set_param_hint('tauD', value=1e-4, min=1e-6, max=100)
#     gmodel.set_param_hint('sp', value=0.01, min=0.001, max=0.1)
#     gmodel.set_param_hint('bl', value=1e-6)
#     pars = gmodel.make_params()
#     return gmodel.fit(y, pars, t=t)


# def fit_model_error(t, y, wt=1):
#     gmodel = Model(correlation)
#     gmodel.set_param_hint('g0', value=0.1)
#     gmodel.set_param_hint('tauD', value=1e-4, min=1e-6, max=100)
#     gmodel.set_param_hint('sp', value=0.01, min=0.001, max=0.1)
#     gmodel.set_param_hint('bl', value=1e-6)
#     pars = gmodel.make_params()
#     return gmodel.fit(y, pars, t=t, weights=wt)


# def correlation(t, g0, tauD, sp, bl):
#     return g0 / ((1 + t / tauD) * (1 + sp * t / tauD)**(0.5)) + bl


# # def generate_report(result):
# #     print(result.fit_report())


# def analyse_data_single(filename, lowlimit=1e-6, highlimit=1):
#     corr_data = return_corr_function_data(filename=filename)

#     t = corr_data[0]

#     mask = (t > lowlimit)  # * (t < highlimit)

#     # Limit the time axis in the raw data
#     useful_t = corr_data[0][mask]

#     # Apply the same mask on all the y data
#     useful_y_list = [y[mask] for y in corr_data[1:]]
#     fit_param = fit_model(useful_t, useful_y_list)
#     plot_fits(corr_data, fit_param=fit_param, mask=mask)
#     return fit_param


# def analyse_data_multi(filename, lowlimit=1e-6, highlimit=1):
#     files_list = find_files()
#     b = np.genfromtxt(
#         files_list[0], delimiter="\t", usecols=(0), skip_header=50, max_rows=186).T
#     # print(b)
#     for fname in files_list:
#         b = np.column_stack((b, np.genfromtxt(
#             fname, delimiter="\t", usecols=(3), skip_header=50, max_rows=186) - 1))
#     # c contains only the Y values (for mean and std calculation)
#     c = b[::, 1:int(len(files_list)) + 1:]  # c has all Y data

#     b = np.column_stack((b, np.mean(c, axis=1), np.std(c, axis=1)))

#     t = b.T[0]

#     mask = (t > lowlimit) * (t < highlimit)

#     # Limit the time axis in the raw data
#     useful_t = b.T[0][mask]

#     # Apply the same mask on all the y data
#     useful_y_list = b.T[-2][mask]
#     useful_sd = b.T[-1][mask]
#     fit_param = fit_model_error(useful_t, useful_y_list, 1 / useful_sd)
#     plot_fits_multi(corr_data=b.T, fit_param=fit_param, mask=mask)
#     return fit_param


# def plot_fits(corr_data, fit_param, mask):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(corr_data[0][mask], corr_data[1][mask], 'o', mfc='none', label="Raw data")
#     ax.plot(corr_data[0][mask], fit_param.best_fit, '-r',
#             label="Fitted data")
#     ax.plot(corr_data[0][mask], fit_param.residual, '-g', label="Residuals")
#     ax.set_xscale('log')
#     ax.set_xlabel('Delay time (s)', fontsize=14)
#     ax.set_ylabel('Autocorrelation', fontsize=14)
#     ax.text(0.7, 0.6,
#             "tauD = " +
#             '%.1f' % (fit_param.best_values["tauD"] * 1e6) + ' $\mu s$' + '\n'
#             + "g0 = " + '%.2E' % (fit_param.best_values["g0"]) + '\n'
#             "sp = " + '%.2E' % (fit_param.best_values["sp"]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15)
#     ax.legend()
#     plt.show()


# def plot_fits_multi(corr_data, fit_param, mask):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.errorbar(corr_data[0], corr_data[-2], yerr=corr_data[-1],
#                 fmt='o', markevery=3, errorevery=3, label="Raw data")
#     ax.plot(corr_data[0][mask], fit_param.best_fit, '-r',
#             label="Fitted data")
#     ax.plot(corr_data[0][mask], fit_param.residual /
#             fit_param.weights, '-g', label="Residuals")
#     ax.set_xscale('log')
#     ax.set_xlabel('Delay time (s)', fontsize=14)
#     ax.set_ylabel('Autocorrelation', fontsize=14)
#     ax.text(0.7, 0.6,
#             "tauD = " +
#             '%.1f' % (fit_param.best_values["tauD"] * 1e6) + ' $\mu s$' + '\n'
#             + "g0 = " + '%.2E' % (fit_param.best_values["g0"]) + '\n'
#             "sp = " + '%.2E' % (fit_param.best_values["sp"]), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15)
#     ax.legend()
#     plt.show()


def find_files(keyword="./*.asc"):
    return sorted(glob.glob(keyword))
