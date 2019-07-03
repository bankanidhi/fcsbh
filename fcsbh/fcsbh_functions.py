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

def return_data(filename,*args):
    """This function opens ascii files and processes the data to return one of the following:Average counts, correlation or raw counts, depending on the *args parameter
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
    edge_min = z.min()
    edge_max = z.max()
    no_of_bins = int(edge_max-edge_min)
    hist, bin_edges = np.histogram(z,bins=no_of_bins,range=(edge_min,edge_max))
    return z


def plot_raw_counts(filename):
    file_list = find_files()
    counter = 0
    for k in file_list:
        b = return_data(k)
        counter += 1
        plt.plot(b)
        if counter >= 10:
            break
    plt.xlabel("Time (ms)")
    plt.ylabel("Photons per ms")
    plt.ylim(0,16000)
    plt.show()




""" This function when called with an arguement return that file if present. Without ant argumrnts, it returns a list containing all .asc file
"""
def find_files(keyword="./*.asc"):
    return sorted(glob.glob(keyword))
