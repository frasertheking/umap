#!/usr/bin/env python
"""pca_preprocess_all.py: Preprocess raw PIP and MET data into a csv for use in dimensionality reduction."""

import glob, os, math
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.rcParams.update({'font.size': 15})

def average_data_in_5min_intervals(dataset):
    start_of_day = pd.to_datetime(dataset['time'].values[0]).normalize()
    end_of_day = start_of_day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    time_intervals = pd.date_range(start=start_of_day, end=end_of_day, freq='5T')
    dataset['time'] = pd.to_datetime(dataset['time'].values)
    averaged_data = {}

    for var_name in ['temperature', 'relative_humidity', 'pressure', 'wind_speed', 'wind_direction']:
        resampled = dataset[var_name].resample(time='5T').mean()
        averaged_data[var_name] = resampled

    new_dataset = xr.Dataset(averaged_data)
    new_dataset = new_dataset.reindex(time=time_intervals, fill_value=np.nan)
    
    return new_dataset

def get_dew_point_c(t_air_c, rel_humidity):
    A = 17.27
    B = 237.7
    alpha = ((A * t_air_c) / (B + t_air_c)) + math.log(rel_humidity/100.0)
    return (B * alpha) / (A - alpha)

def calc_various_pca_inputs():
    # sites = ['MQT', 'FIN', 'HUR', 'HAUK', 'KIS', 'KO1', 'KO2', 'IMP', 'APX', 'NSA', 'YFB']
    # inst = ['006', '004', '008', '007', '007', '002', '003', '003', '007', '010', '003']
    sites = ['MQT', 'FIN', 'HUR', 'KIS', 'KO1', 'KO2', 'IMP', 'NSA', 'YFB']
    inst = ['006', '004', '008', '007', '002', '003', '003', '010', '003']

    # sites = ['HUR']
    # inst = ['008']

    ### Globals
    pip_path = '/Users/fraserking/Development/pip_processing/data/converted/'
    met_path = '/Users/fraserking/Development/pca/data/MET_OUT/'
        
    pip_dates = []
    for file in glob.glob(os.path.join(pip_path, '**', 'edensity_distributions', '*.nc'), recursive=True):
        pip_dates.append(file[-15:-7])

    site_array = []
    N_0_array = []
    lambda_array = []
    total_particle_array = []
    avg_ed_array = []
    avg_rho_array = []
    avg_sr_array = []
    avg_rr_array = []
    avg_vvd_array = []
    mwd_array = []
    avg_t_array = []
    avg_rh_array = []
    avg_dp_array = []
    avg_p_array = []
    avg_ws_array = []
    avg_wd_array = []
    times = []

    for w,site in enumerate(sites):
        print("Working on site " + site)
        number_of_files = 0
        for date in pip_dates:
            # print("Working on day", date)
            
            year = int(date[:4])
            month = int(date[4:6])
            day = int(date[-2:])

            # Load PIP data
            try:
                ds_edensity_lwe_rate = xr.open_dataset(pip_path + str(year) + '_' + site + '/netCDF/adjusted_edensity_lwe_rate/' + inst[w] + date + '_min.nc')
                ds_edensity_distributions = xr.open_dataset(pip_path + str(year) + '_' + site + '/netCDF/edensity_distributions/' + inst[w] + date + '_rho.nc')
                ds_velocity_distributions = xr.open_dataset(pip_path + str(year) + '_' + site + '/netCDF/velocity_distributions/' + inst[w] + date + '_vvd.nc')
                ds_particle_size_distributions = xr.open_dataset(pip_path + str(year) + '_' + site + '/netCDF/particle_size_distributions/' + inst[w] + date + '_psd.nc')
            except FileNotFoundError:
                # print("Could not open PIP file")
                continue

            # Load MET data
            try:
                if 'KO' in site:
                    site = 'ICP'
                met_observations = xr.open_dataset(met_path + site + '/' + date + '_met.nc')
            except FileNotFoundError:
                # print("Could not open MET file")
                continue
            
            dsd_values = ds_particle_size_distributions['psd'].values
            edd_values = ds_edensity_distributions['rho'].values
            vvd_values = ds_velocity_distributions['vvd'].values
            sr_values = ds_edensity_lwe_rate['nrr_adj'].values
            rr_values = ds_edensity_lwe_rate['rr_adj'].values
            ed_values = ds_edensity_lwe_rate['ed_adj'].values
            bin_centers = ds_particle_size_distributions['bin_centers'].values

            resampled_met = average_data_in_5min_intervals(met_observations)
            avg_t_array.append(resampled_met['temperature'].values)
            avg_rh_array.append(resampled_met['relative_humidity'].values)
            avg_dp_array.append(list(map(get_dew_point_c, resampled_met['temperature'].values, resampled_met['relative_humidity'].values)))
            avg_p_array.append(resampled_met['pressure'].values)
            avg_ws_array.append(resampled_met['wind_speed'].values)
            avg_wd_array.append(resampled_met['wind_direction'].values)

            if len(ds_particle_size_distributions.time) != 1440:
                # print("PIP data record too short for day, skipping!")
                continue

            ########## PIP CALCULATIONS 
            func = lambda t, a, b: a * np.exp(-b*t)

            # Initialize the datetime object at the start of the day
            current_time = datetime(year, month, day, 0, 0)

            # Loop over each 5-minute block
            count = 0
            for i in range(0, dsd_values.shape[0], 5):
                if i >= 1440:
                    continue

                count += 1
                block_avg = np.mean(dsd_values[i:i+5, :], axis=0)
                valid_indices = ~np.isnan(block_avg)
                block_avg = block_avg[valid_indices]
                valid_bin_centers = bin_centers[valid_indices]

                times.append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
                current_time += timedelta(minutes=5)

                if block_avg.size == 0:
                    N_0_array.append(np.nan)
                    lambda_array.append(np.nan)
                    avg_vvd_array.append(np.nan)
                    avg_ed_array.append(np.nan)
                    avg_rho_array.append(np.nan)
                    avg_sr_array.append(np.nan)
                    avg_rr_array.append(np.nan)
                    total_particle_array.append(0)
                    mwd_array.append(np.nan)
                    site_array.append(np.nan)
                    continue

                # Calculate average fallspeed over the 5-minute interval
                vvd_slice = vvd_values[i:i+5, :]
                avg_vvd_array.append(vvd_slice[vvd_slice != 0].mean())

                # Calculate the average eDensity of the 5-minute interval
                avg_ed_array.append(np.nanmean(ed_values[i:i+5]))

                # Calculate the average eDensity of the 5-minute interval
                rho_slice = edd_values[i:i+5]
                avg_rho_array.append(rho_slice[rho_slice != 0].mean())

                # Calculate the average snowfall rate over the 5-minute interval
                avg_sr_array.append(np.nanmean(sr_values[i:i+5]))

                # Calculate the average rainfall rate over the 5-minute interval
                avg_rr_array.append(np.nanmean(rr_values[i:i+5]))

                # Calculate total number of particles over the 5-minute interval
                total_particle_array.append(np.nansum(dsd_values[i:i+5, :], axis=(0, 1)))

                # Calculate mean mass diameter over the 5-minute interval
                if edd_values[i:i+5, valid_indices].shape == dsd_values[i:i+5, valid_indices].shape:
                    mass_dist = edd_values[i:i+5, valid_indices] * dsd_values[i:i+5, valid_indices] * (4/3) * np.pi * (valid_bin_centers/2)**3
                    mass_weighted_diameter = np.sum(mass_dist * valid_bin_centers) / np.sum(mass_dist)
                    mwd_array.append(mass_weighted_diameter)
                else:
                    mwd_array.append(np.nan)

                # Calculate N0 and Lambda
                try:
                    popt, pcov = curve_fit(func, valid_bin_centers, block_avg, p0 = [1e4, 2], maxfev=600)
                    if popt[0] > 0 and popt[0] < 10**7 and popt[1] > 0 and popt[1] < 10:
                        N_0_array.append(popt[0])
                        lambda_array.append(popt[1])
                    else:
                        N_0_array.append(np.nan)
                        lambda_array.append(np.nan)

                except RuntimeError:
                    N_0_array.append(np.nan)
                    lambda_array.append(np.nan)

                site_array.append(site)

            number_of_files += 1


    avg_t_array = [j for sub in avg_t_array for j in sub]
    avg_rh_array = [j for sub in avg_rh_array for j in sub]
    avg_dp_array = [j for sub in avg_dp_array for j in sub]
    avg_p_array = [j for sub in avg_p_array for j in sub]
    avg_ws_array = [j for sub in avg_ws_array for j in sub]
    avg_wd_array = [j for sub in avg_wd_array for j in sub]

    df = pd.DataFrame(data={'site': site_array, 'time': times, 'n0': N_0_array, 'D0': mwd_array, 'Nt': total_particle_array, \
                            'Fs': avg_vvd_array, 'Sr': avg_sr_array, 'Rr': avg_rr_array,  'Ed': avg_ed_array, \
                            'Rho': avg_rho_array, 'lambda': lambda_array, 't': avg_t_array, 'rh': avg_rh_array, 'dp': avg_dp_array, 'p': avg_p_array, 'ws': avg_ws_array, 'wd': avg_wd_array})
    df.dropna(inplace=True)
    
    df.to_csv('/Users/fraserking/Development/pca/data/pca_inputs/all_sites_pip_met2.csv')

def plot_corr(df, size=12):
    corr_df = df.drop(columns=['type'])
    print(corr_df)
    corr = corr_df.corr()
    
    corr_sum = corr.sum().sort_values(ascending=False)
    corr = corr.loc[corr_sum.index, corr_sum.index]

    fig, ax = plt.subplots(figsize=(size, size))
    plt.title("PSD Variable Correlation Matrix")
    h = ax.matshow(corr, cmap='bwr', vmin=-1, vmax=1)
    fig.colorbar(h, ax=ax, label='Correlation')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    plt.savefig('/data2/fking/s03/images/corr.png')
    sns_plot = sns.pairplot(df, kind="hist", diag_kind="kde", hue='type', height=5, palette=['blue', 'red'], corner=True)
    sns_plot.map_lower(sns.kdeplot, levels=4, color=".2")
    sns_plot.savefig('/data2/fking/s03/images/output_kde.png')


def plot_timeseries(site):
    df = pd.read_csv('/data2/fking/s03/data/processed/pca_inputs/' + site + '_pip.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna() 
    df = df[(df['Ed'] >= 0)]
    df = df[(df['Ed'] <= 1)]
    df = df[(df['lambda'] <= 2)]
    df['time'] = pd.to_datetime(df['time'])

    df.set_index('time', inplace=True)
    cols = ['Nt', 'n0', 'lambda', 'Ed', 'D0', 'Sr', 'Fs', 'Rho']
    units = ['#', 'm-3 mm-1', 'mm-1', 'g cm-3', 'mm', 'mm hr-1', 'm s-1', 'g cm-3']
    df_rolling = df[cols].rolling(window=1000).mean()

    fig, axs = plt.subplots(4, 2, figsize=(20, 10), sharex=True)

    for i, ax in enumerate(axs.flatten()):
        col = cols[i]
        df_rolling[col].plot(ax=ax, color='black', linewidth=2)
        ax.set_title(col)
        ax.set_ylabel(col + ' (' + units[i] + ')')
        ax.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig('/data2/fking/s03/images/timeseries.png')


def load_and_plot_pca_for_site(site):
    df = pd.read_csv('/data2/fking/s03/data/processed/pca_inputs/' + site + '_pip.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna()
    df = df[(df['Ed'] >= 0)]
    df = df[(df['Ed'] <= 1)]
    df = df[(df['lambda'] <= 2)]
    df['type'] = df['Rho'].apply(lambda x: 'snow' if x < 0.4 else 'rain')

    df['Log10_n0'] = df['n0'].apply(np.log10)
    df['Log10_lambda'] = df['lambda'].apply(np.log10)
    df['Log10_Ed'] = df['Ed'].apply(np.log10)
    df['Log10_Fs'] = df['Fs'].apply(np.log10)
    df['Log10_Rho'] = df['Rho'].apply(np.log10)
    df['Log10_D0'] = df['D0'].apply(np.log10)
    df['Log10_Sr'] = df['Sr'].apply(np.log10)
    df['Log10_Nt'] = df['Nt'].apply(np.log10)
    df.drop(columns=['n0', 'lambda', 'Nt', 'Ed', 'Fs', 'Rho', 'D0', 'Sr'], inplace=True)
    print(df.describe())
    plot_corr(df)
    print(df)

def load_raw_values_and_save_standardized_version(site):
    df = pd.read_csv('/data2/fking/s03/data/processed/pca_inputs/' + site + '_pip.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna()
    df = df[(df['Ed'] >= 0)]
    df = df[(df['Ed'] <= 1)]
    df = df[(df['lambda'] <= 2)]
    
    df['Log10_n0'] = df['n0'].apply(np.log10)
    df['Log10_lambda'] = df['lambda'].apply(np.log10)
    df['Log10_Nt'] = df['Nt'].apply(np.log10)
        
    df.drop(columns=['n0', 'lambda', 'Nt'], inplace=True)
    
    for col in ['Log10_n0', 'Log10_lambda', 'Ed', 'Fs', 'Rho', 'D0', 'Sr', 'Log10_Nt']:
        df['std_'+col] = (df[col] - df[col].mean()) / df[col].std()

    df.to_csv('/data2/fking/s03/data/processed/pca_inputs/final_' + site + '_pip.csv', index=False)


if __name__ == '__main__':
    calc_various_pca_inputs()
    # plot_timeseries('MQT')
    # load_and_plot_pca_for_site('MQT')
    # load_raw_values_and_save_standardized_version('MQT')

