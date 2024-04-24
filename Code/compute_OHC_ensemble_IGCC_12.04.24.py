# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import csv
from datetime import date
import os.path


def read_JMA(datadir=os.path.expanduser('~/Data/JMA/'),
             filename='OHC_0-700.txt'):
    f = open(datadir + filename, "r")
    lines = f.readlines()
    keys = ['Year', 'HeatContentAnomaly', 'Uncertainty (1-Sigma)']
    f.close()
    ncols = len(lines[-1].split())  # Get number of columns from the last line..
    nrows = len(lines)  # Get number of data lines in text file
    data = np.zeros([nrows, ncols])  # Convention of rows then columns for Numpy arrays (?)
    for jj in range(nrows):
        for ii in range(ncols):
            data[jj, ii] = float(lines[jj].split()[ii])
    data_dict = dict.fromkeys(keys)
    for kk, key in enumerate(keys):
        data_dict[key] = data[:, kk]

    yrs = data_dict['Year'] + 0.5  # Express years as mid-points
    series = data_dict['HeatContentAnomaly']  # Data expressed as Zetta Joules (10^21)
    error = data_dict['Uncertainty (1-Sigma)']  # Data expressed as Zetta Joules Zetta Joules (10^21)

    return yrs, series, error


def read_NCEI(datadir=os.path.expanduser('~/Data/NCEI/'),
              filename='h22-w0-700m.dat',
              key1='WO',
              key2='WOse', skip=1):
    f = open(datadir + filename, "r")
    lines = f.readlines()
    header = lines[skip - 1].strip('\n')
    keys = header.split()
    f.close()
    ncols = len(lines[-1].split())  # Get number of columns from the last line..
    nrows = len(lines) - skip  # Get number of data lines in text file
    data = np.zeros([nrows, ncols])  # Convention of rows then columns for Numpy arrays (?)
    for jj in range(nrows):
        for ii in range(ncols):
            data[jj, ii] = float(lines[jj + skip].split()[ii])
    data_dict = dict.fromkeys(keys)
    for kk, key in enumerate(keys):
        data_dict[key] = data[:, kk]

    yrs = data_dict['YEAR']
    series = data_dict[key1] * 10.0 # Convert from 10^22 joules to Zetta Joules (10^21)
    error = data_dict[key2] * 10.0  # Convert from 10^22 joules to Zetta Joules (10^21)

    return yrs, series, error


def read_IAP(datadir=os.path.expanduser('~/Data/IAP/'),
               filename='OHC0_700m_annual_timeseries_1981_2010baseline_IAPv4.txt'):

    f = open(datadir + filename, "r")
    lines = f.readlines()
    f.close()
    ncols = len(lines[-1].split())  # Get number of columns from the last line..
    nrows = len(lines)  # Get number of data lines in text file
    data = np.zeros([nrows, ncols])  # Convention of rows then columns for Numpy arrays (?)
    for jj in range(nrows):
        for ii in range(ncols):
            data[jj, ii] = float(lines[jj].split()[ii])

    yrs = data[:, 0]
    series = data[:, 1]
    error = data[:, 2]

    ratio = 1.0 / 1.96  # Convert errors to +/- 1 standard deviation

    return yrs, series, error * ratio


def read_EN4(datadir=os.path.expanduser('~/Data/EN4/'),
             filename='OHC_series_1950_2023_MOHC_g10_19502023clim_0700m.txt',
             key1='HC_anomaly(Joules)',
             key2='+/-(1s.e.)', skip=1):
    f = open(datadir + filename, "r")
    lines = f.readlines()
    header = lines[skip - 1].strip('\n')
    keys = header.split()
    f.close()
    ncols = len(lines[-1].split())  # Get number of columns from the last line..
    nrows = len(lines) - skip  # Get number of data lines in text file
    data = np.zeros([nrows, ncols])  # Convention of rows then columns for Numpy arrays (?)
    for jj in range(nrows):
        for ii in range(ncols):
            data[jj, ii] = float(lines[jj + skip].split()[ii])
    data_dict = dict.fromkeys(keys)
    for kk, key in enumerate(keys):
        data_dict[key] = data[:, kk]

    yrs = data_dict['Year'] + 0.5 # Express years as mid-points
    series = data_dict[key1] / 1e21 # Convert from Joules to Zetta Joules
    error = data_dict[key2] / 1e21 # Convert from Joules to Zetta Joules

    return yrs, series, error


def read_Domingues(datadir=os.path.expanduser('~/Data/Domingues/'),
               filename='GOHC0700m_timeseries.txt'):

    f = open(datadir + filename, "r")
    lines = f.readlines()
    f.close()
    ncols = len(lines[-1].split())  # Get number of columns from the last line..
    nrows = len(lines)  # Get number of data lines in text file
    data = np.zeros([nrows, ncols])  # Convention of rows then columns for Numpy arrays (?)
    for jj in range(nrows):
        for ii in range(ncols):
            data[jj, ii] = float(lines[jj].split()[ii])

    yrs = data[:, 0]
    series = data[:, 1] * 10.0 # Convert to Zetta Joules
    error = data[:, 2] * 10.0 # Convert to Zetta Joules

    return yrs, series, error


def extract_period(yrs, series, fyr=1971.5, lyr=2023.5):
    """
    This function extracts a subsection of a timeseries on the basis of specified first and last years
    inputs:
        * yrs    = time coordinate in years
        * series = input timeseries
        * fyr    = first year of sub-period
        * lyr    = last year of sub-period
    outputs:
        * subyrs    = years corresponding to sub-section timeseries
        * subseries = sub-section of input timeseries
    """
    subi = np.where((yrs >= fyr) & (yrs <= lyr))[0]  # Index for sub-section
    subseries = series[subi].copy()
    subyrs = yrs[subi]
    return subyrs, subseries

def runningmean(yrs, series, window=3):
    """
    This function applies a running-mean smoothing to OHC timerseries in order to reduce the effect
    of sampling noise. The default is to use a 3-yr window, following Domingues et al (2008) and AR6
    Chapter 2.
    :param yrs: input time coordinate in years
    :param series: input timeseries
    :param window: the length of the window for running-mean
    :return: timeseries with running-mean applied, with same length as original timeseries
    """
    # Perform the runnning mean..
    weights = np.repeat(1.0, window)/window
    rmseries = np.convolve(series, weights, 'same') # Preserve original series length (end effects visible)
    rmseries[0] = series[0] # Make sure the start and end points are identical to original series (avoid edge effects)
    rmseries[-1] = series[-1]
    return rmseries

color_dict = {'IAP':'tab:purple',
           'Dom':'tab:blue',
           'EN4':'tab:red',
           'JMA':'tab:green',
           'NCEI':'tab:orange'}

# Define ensemble members for each vertical layer..
ohc_names = {'0-700m':['NCEI', 'JMA', 'EN4', 'IAP', 'Dom'],
            '700-2000m':['NCEI', 'JMA', 'IAP']}

# Build dictionaries from updated OHC timeseries read in directly from individual files..
ohc_dict_0to700m = {}
ohc_dict_0to700m_err = {}
ohc_dict_700to2000m = {}
ohc_dict_700to2000m_err = {}

# Select a range of years that can be accommodated by all products..
fyr = 1970.5 # The Domingues et al product is only available from 1970 onwards
lyr = 2023.5

# Read in the EN4 timeseries..
yrs, series, eseries = read_EN4()
subyrs, subseries = extract_period(yrs, series, fyr=fyr, lyr=lyr)
subyrs, subeseries = extract_period(yrs, eseries, fyr=fyr, lyr=lyr)
ohc_dict_0to700m['EN4'] = subseries
ohc_dict_0to700m_err['EN4'] = subeseries

# Read in the Domingues timeseries..
yrs, series, eseries = read_Domingues()
subyrs, subseries = extract_period(yrs, series, fyr=fyr, lyr=lyr)
subyrs, subeseries = extract_period(yrs, eseries, fyr=fyr, lyr=lyr)
ohc_dict_0to700m['Dom'] = subseries
ohc_dict_0to700m_err['Dom'] = subeseries

# Read in the NCEI timeseries..
yrs1, series1, eseries1 = read_NCEI()
yrs2, series2, eseries2 = read_NCEI(filename='h22-w0-2000m_merged.dat')
subyrs1, subseries1 = extract_period(yrs1, series1, fyr=fyr, lyr=lyr)
subyrs1, subeseries1 = extract_period(yrs1, eseries1, fyr=fyr, lyr=lyr)
subyrs2, subseries2 = extract_period(yrs2, series2, fyr=fyr, lyr=lyr)
subyrs2, subeseries2 = extract_period(yrs2, eseries2, fyr=fyr, lyr=lyr)
ohc_dict_0to700m['yrs'] = subyrs1
ohc_dict_0to700m['NCEI'] = subseries1
ohc_dict_0to700m_err['NCEI'] = subeseries1
ohc_dict_700to2000m['yrs'] = subyrs2
ohc_dict_700to2000m['NCEI'] = subseries2 - subseries1 # Estimate 700-2000 m by subtracting layers
ohc_dict_700to2000m_err['NCEI'] = subeseries2 * (1300./2000.) # Estimate based on scaled ratio of levels

# Read in the IAP timeseries..
yrs1, series1, eseries1 = read_IAP()
yrs2, series2, eseries2 = read_IAP(filename='OHC700_2000m_annual_timeseries_1981_2010baseline_IAPv4.txt')
subyrs1, subseries1 = extract_period(yrs1, series1, fyr=fyr, lyr=lyr)
subyrs1, subeseries1 = extract_period(yrs1, eseries1, fyr=fyr, lyr=lyr)
subyrs2, subseries2 = extract_period(yrs2, series2, fyr=fyr, lyr=lyr)
subyrs2, subeseries2 = extract_period(yrs2, eseries2, fyr=fyr, lyr=lyr)
ohc_dict_0to700m['IAP'] = subseries1
ohc_dict_0to700m_err['IAP'] = subeseries1
ohc_dict_700to2000m['IAP'] = subseries2
ohc_dict_700to2000m_err['IAP'] = subeseries2

# Read in the JMA timeseries..
yrs1, series1, eseries1 = read_JMA()
yrs2, series2, eseries2 = read_JMA(filename='OHC_700-2000.txt')
subyrs1, subseries1 = extract_period(yrs1, series1, fyr=fyr, lyr=lyr)
subyrs1, subeseries1 = extract_period(yrs1, eseries1, fyr=fyr, lyr=lyr)
subyrs2, subseries2 = extract_period(yrs2, series2, fyr=fyr, lyr=lyr)
subyrs2, subeseries2 = extract_period(yrs2, eseries2, fyr=fyr, lyr=lyr)
ohc_dict_0to700m['JMA'] = subseries1
ohc_dict_0to700m_err['JMA'] = subeseries1
ohc_dict_700to2000m['JMA'] = subseries2
ohc_dict_700to2000m_err['JMA'] = subeseries2


# Dictionaries for ensemble assessments of OHC change..
ensm_ohc_dict = {}
ensm_ohc_dict['units'] = 'ZJ'
ensm_ohc_dict['error units'] = 'ZJ (1-sigma)'
ensm_ohc_dict['baseline period'] = '1995-2014'

#-----------------------------
# Generate and plot the ensemble..
#-----------------------------

datestr = date.isoformat(date.today()) # Get string to represent processing date

# Generate array of years common to all ensemble members
core_yrs = np.arange(1971.5, 2023.5 + 1, 1.0) # Generate at timeseries of years from 1971.5 to 2023.5
fyr = core_yrs[0]
lyr = core_yrs[-1]

plotdir = os.path.expanduser('~/python/IGCC/src/notebooks/plots/')

byr1=1995.5 # Set basline period used in AR6 and extract indices
byr2=2014.5
bindex = np.where((yrs >= byr1) & (yrs <= byr2))[0]


for layer in ['0-700m', '700-2000m']:
    plotfile = 'plot_OHC_ensemble_structural_mapping_uncertainty_'+layer+'_IGCC_'+datestr+'.png'

    names = ohc_names[layer]
    nprod = len(names)

    dict = ohc_dict_0to700m
    dict_err = ohc_dict_0to700m_err

    if layer == '700-2000m':
        dict = ohc_dict_700to2000m
        dict_err = ohc_dict_700to2000m_err

    for nn, name in enumerate(names):
        series = dict[name].copy()
        eseries = dict_err[name].copy()
        yrs = dict['yrs']

        # Apply common baseline period..
        series -= series[bindex].mean()

        if name in ['NCEI', 'JMA', 'EN4', 'IAP']: # Domingues already has a 3-year smoothing applied
            print('Applying 3-year running mean to '+name+'...')
            series = runningmean(yrs, series)  # Apply 3-year running mean as per Domingues et al.
            eseries = runningmean(yrs, eseries)

        subyrs, subseries = extract_period(yrs, series, fyr=fyr, lyr=lyr)
        subyrs, subeseries = extract_period(yrs, eseries, fyr=fyr, lyr=lyr)

        if (layer == '0-700m') & (name == 'Dom'):
            ensm_ohc_dict['ocean_' + layer] = subseries - subseries[0] # Input Domingues as central estimate for 0-700m layer
            ensm_ohc_dict['yrs'] = subyrs

        if (layer == '700-2000m') & (name == 'JMA'):
            ensm_ohc_dict['ocean_' + layer] = subseries - subseries[0]  # Input JMA as central estimate for 700-2000m layer

        if nn == 0:  # Setup matrix for ensemble..
            nyrs = len(subseries)
            ensm = np.zeros((nprod, nyrs))
            ensm_error = np.zeros((nprod, nyrs))
            ensm[nn, :] = subseries
            ensm_error[nn, :] = subeseries

        else:
            ensm[nn, :] = subseries
            ensm_error[nn, :] = subeseries

    # Compute ensemble statistics for OHC:
    ensm_mean   = np.nanmean(ensm, axis=0).copy()
    struct_err  = np.nanstd(ensm, axis=0).copy()
    map_err     = np.nanmax(ensm_error, axis=0).copy()
    total_err   = np.sqrt(np.square(struct_err) + np.square(map_err)).copy()
    ensm_ohc_dict['ocean_' + layer + '_error'] = total_err

    plt.figure()
    f = plt.gcf()
    f.set_size_inches(12.0, 6.0)
    matplotlib.rcParams['font.size'] = 11

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for nn, name in enumerate(names):
        color = color_dict[name]
        series1 = ensm[nn, :]
        series2 = ensm_error[nn, :]
        ax1.plot(core_yrs, series1, color=color, label=name)
        ax2.plot(core_yrs, series2 * 1.645, color=color, label=name)
    ax1.plot(core_yrs, ensm_mean, 'k', linewidth=2.0, label='Ens_Mean')
    ax1.fill_between(core_yrs, ensm_mean - struct_err * 1.645, ensm_mean + struct_err * 1.645,
                     facecolor='gray', alpha=0.5, edgecolor='None', label='90% C.I.')
    ax1.set_title('OHC Structural Uncertainty ' + layer)
    ax1.set_ylabel('OHC Anomaly (ZJ)')
    ax1.legend(prop={'size': 10}, frameon=False, ncol=1)

    ax2.fill_between(core_yrs, np.zeros_like(map_err), map_err * 1.645,
                     facecolor='gray', alpha=0.5, edgecolor='None')
    ax2.plot(core_yrs, map_err * 1.645, 'k:', linewidth=2.0, label='Mapping Error')

    ax2.set_title('OHC Mapping Uncertainty ' + layer)
    ax2.set_ylabel('90% C.I. OHC Error (ZJ)')
    ax2.legend(prop={'size': 10}, frameon=False, ncol=1)

    print("Saving file: ", plotdir + plotfile)
    plt.savefig(plotdir + plotfile, dpi=300)
    plt.show()

#-----------------------------
#-----------------------------

#---------------------------------------------------------------------------
# Incorporate estimate of sub-2000m trends based on Purkey and Johson (2010)
# and Desbruyeres et al (2016)
#---------------------------------------------------------------------------

# Load from IPCC AR6 data file..
datadir = os.path.expanduser('~/python/ar6/src/notebooks/FGD/data/')
pfile   = 'AR6_energy_GMSL_timeseries_FGD_1971to2018_corrigendum.pickle'
data = pickle.load( open( datadir + pfile, 'rb' ) )
energy_dict = data['energy_dict']
yrs = energy_dict['yrs']
ohc = energy_dict['ocean_>2000m']
ohc_err = energy_dict['ocean_>2000m_error']

# Adopt a simple linear extrapolation of the trend + associated error
nzi = np.where(ohc != 0.0)[0]
ohc_fit=np.polyfit(yrs[nzi], ohc[nzi], 1) # Get linear fits for extrapolation
err_fit=np.polyfit(yrs[nzi], ohc_err[nzi], 1)
yi = np.where(yrs <= yrs[nzi[-1]])[0] # Isolate the years up to final year
new_yrs = np.array([2019.5, 2020.5, 2021.5, 2022.5, 2023.5])
extrp1 = np.poly1d(ohc_fit) # Create extrapolation function
new_ohc = extrp1(new_yrs)   # Get extrapolated values
extrp2 = np.poly1d(err_fit)
new_err = extrp2(new_yrs)

all_yrs = np.concatenate([yrs[yi], new_yrs]) # Concatenate to get whole timeseries
all_ohc = np.concatenate([ohc[yi], new_ohc])
all_err = np.concatenate([ohc_err[yi], new_err])

subyrs1, subseries1 = extract_period(all_yrs, all_ohc, fyr=fyr, lyr=lyr)
subyrs1, subeseries1 = extract_period(all_yrs, all_err, fyr=fyr, lyr=lyr)

ensm_ohc_dict['ocean_2000-6000m'] = subseries1
ensm_ohc_dict['ocean_2000-6000m_error'] = subeseries1

ensm_ohc_dict['ocean_Full-depth'] = ensm_ohc_dict['ocean_0-700m'] + ensm_ohc_dict['ocean_700-2000m'] + ensm_ohc_dict['ocean_2000-6000m']
ensm_ohc_dict['ocean_Full-depth_error'] = ensm_ohc_dict['ocean_0-700m_error'] + ensm_ohc_dict['ocean_700-2000m_error'] + ensm_ohc_dict['ocean_2000-6000m_error']


#-------------------------------------------------
# Write the OHC ensemble to CSV / *.pickle file..
#-------------------------------------------------

savedir  = os.path.expanduser('~/python/IGCC/src/notebooks/data/')
pickle_file = 'AR6_OHC_ensemble_IGCC_update_'+datestr+'.pickle'
csv_file = 'AR6_OHC_ensemble_IGCC_update_'+datestr+'.csv'

ohc_0to700m = ensm_ohc_dict['ocean_0-700m']
ohc_700to200m = ensm_ohc_dict['ocean_700-2000m']
ohc_below2000m = ensm_ohc_dict['ocean_2000-6000m']
ohc_total = ensm_ohc_dict['ocean_Full-depth']

ohc_err_0to700m = ensm_ohc_dict['ocean_0-700m_error']
ohc_err_700to200m = ensm_ohc_dict['ocean_700-2000m_error']
ohc_err_below2000m = ensm_ohc_dict['ocean_2000-6000m_error']
ohc_err_total = ensm_ohc_dict['ocean_Full-depth_error']

with open(savedir + csv_file, mode='w') as CSV_file:
    OHC_writer = csv.writer(CSV_file, delimiter=',',
                             quoting=csv.QUOTE_MINIMAL)
    OHC_writer.writerow(['Changes in global ocean heat content in Zettajoules (ZJ) ' +
                          'relative to the 1971 average.'])
    OHC_writer.writerow(['Year',
                          'Central Estimate 0-700m',
                          '0-700m Uncertainty (1-sigma)',
                          'Central Estimate 700-2000m',
                          '700-2000m Uncertainty (1-sigma)',
                          'Central Estimate >2000m',
                          '>2000m Uncertainty (1-sigma)',
                          'Central Estimate Full-depth',
                          'Full-depth Uncertainty (1-sigma)'])

    for yy, yr in enumerate(core_yrs):
        OHC_writer.writerow([yr,
        ohc_0to700m[yy],
        ohc_err_0to700m[yy],
        ohc_700to200m[yy],
        ohc_err_700to200m[yy],
        ohc_below2000m[yy],
        ohc_err_below2000m[yy],
        ohc_total[yy,],
        ohc_err_total[yy,]])

print("SAVING:"+savedir+csv_file)

ohc_dict = {'ensm_ohc_dict':ensm_ohc_dict}

print("SAVING:"+savedir+pickle_file)
pickle.dump( ohc_dict, open( savedir+pickle_file, "wb" ) )
#-------------------------------------------------
#-------------------------------------------------