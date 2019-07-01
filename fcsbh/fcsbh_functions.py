import glob
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from lmfit import Model
import re
from tempfile import TemporaryFile
import statistics
import math
import peakutils

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
    cor_end -= cor_start
    raw_end -= raw_start
    # creating the correlation data var = correlation and raw_count data

    correlation = np.genfromtxt(filename, usecols=None, skip_header=cor_start, max_rows=cor_end).T
    raw_count = np.genfromtxt(filename, usecols=None, skip_header=raw_start, max_rows=raw_end).T
    # Creating a filter/mask for appropriate time cut off for correlation
    t = correlation[0]
    mask = (t > 1) * (t < 100000)
    useful_corr_x = correlation[0][mask]
    useful_corr_y = correlation[1][mask]

    """ Statistical analysis of PCH dara from raw counts
    """
    z = raw_count[1]
    no_of_bins = int(1.9*mean_count)
    hist, bin_edges = np.histogram(z, bins=no_of_bins, range=(mean_count/10, 20.0*mean_count), weights=None, density=False)

    meanz = np.mean(z)
    stdd = np.std(z)

    """ plotting details if required
    """
    plt.plot(hist)
    # fig, ax = plt.subplots()
    # ax.semilogx(useful_corr_x,useful_corr_y)
    # ax.grid()
    # plt.autoscale(enable=False, tight=None)


    # just to show the plot
    plt.show()


    

def find_files(keyword="./*.asc"):
    return sorted(glob.glob(keyword))
