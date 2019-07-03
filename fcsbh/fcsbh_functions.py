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


def plot_raw_counts1(filename):
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
    plt.ylim(-25,16000)
    plt.show()

def plot_raw_counts(filename):
    file_list = find_files()
    counter = 0
    b = []
    spikes = []
    for k in file_list:
        temp = return_data(k)
        for m in temp:
            b.append(m)
        counter += 1
        if counter >= 10:
            break
    b = np.asarray(b)
    edge_min = b.min()
    edge_max = b.max()
    no_of_bins = int(edge_max-edge_min)
    hist, bin_edges = np.histogram(b,bins=no_of_bins,range=(edge_min,edge_max))
    
    for sp in b:
        if sp > 114:
            spikes.append(sp)
    spikes = np.asarray(spikes)
    spike_min = spikes.min()
    spike_max = spikes.max()
    spike_bin = int(spike_max - spike_min)

    hist1, bin_edges1 = np.histogram(spikes,bins=spike_bin,range=(spike_min,spike_max))
    print(len(spikes))

    # plt.plot(bin_edges[1:],hist)
    plt.bar(bin_edges1[1:],hist1)
    plt.show()


def gaus1(t, y0, a, b, w):
    return y0 + a*np.exp(((-4*np.log(2))*(t-b)**2)/w**2)   


def fit_model(t, y, jj, zjj, meanz, stdd):
    gmodel = Model(gaus1)
    gmodel.set_param_hint('a', value=zjj, min=0.8*zjj, max=1.2*zjj)
    gmodel.set_param_hint('b', value=jj, min=jj-2, max=jj+2)
    gmodel.set_param_hint('w', value=1.2, min=0.1, max=20)
    gmodel.set_param_hint('y0', value=meanz, min=meanz -
                          stdd/2, max=meanz + stdd/2)
    pars = gmodel.make_params()
    return gmodel.fit(y, pars, t=t)


def find_peaks(z,cutoff):
    indexes = peakutils.indexes(z, thres=cutoff, min_dist=20, thres_abs=True)
    return indexes

""" This function when called with an arguement return that file if present. Without ant argumrnts, it returns a list containing all .asc file
"""
def find_files(keyword="./*.asc"):
    return sorted(glob.glob(keyword))
