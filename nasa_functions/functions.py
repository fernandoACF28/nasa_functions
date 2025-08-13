# HERE WE HAVE SOME IMPORTANTS LIBRARIES FOR IMPORT

import os
from tqdm import tqdm
import numpy as np
import pandas as pd 
import earthaccess
import rioxarray as rxr
import xarray as xr
import gc
from haversine import haversine, Unit
from glob import glob as gb
from datetime import datetime,timedelta
from warnings import filterwarnings
from concurrent.futures import ThreadPoolExecutor
filterwarnings('ignore')


########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
# This function is useful for create a temporal series for do download.
def date_list_range(start_date,
                    end_date,
                    start_time,
                    end_time):
    dataframes_list = []
    date_range = pd.date_range(start=start_date,
                               end=end_date)
    for date in date_range:
        df = pd.DataFrame({
            'temporal_start': [f'{date.date()}T{start_time}'],
            'temporal_end': [f'{date.date()}T{end_time}']
        })
        dataframes_list.append(df)
    return dataframes_list
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
#  This fuction is useful for download of products from NASA. ##########################################
def downloads_files_nasa(start_date: str, 
                         end_date: str,
                         product_name: str, 
                         coordinates: tuple[int, ...] | list[tuple,...],
                         start_time: str = '00:00:00',
                         end_time: str = '23:59:59',
                         threads: int = 8):
    '''
    start_date: insert your date in 'YYYY-MM-DD'

    end_date: insert your date in 'YYYY-MM-DD'

    start_time: insert your time in 'HH:MM:SS'

    end_time: insert your time in 'HH:MM:SS'

    product_name: name of product ex: 'MYD04_L2'

    coordinates:  west,south,east,north.

    threads: parallel number of threads to download files.
    '''

    data_ok,data_not_ok = [],[]
    dataframes_list = date_list_range(start_date, end_date, start_time, end_time)

    def download_single_date(df):
        """Auxiliar function to download files in a interval of time"""
        start_time = df['temporal_start'].values[0]
        end_time = df['temporal_end'].values[0]

        try:
            results = earthaccess.search_data(
                short_name=product_name, # the product of interest
                cloud_hosted=True, 
                bounding_box=(coordinates[0], coordinates[1], coordinates[2], coordinates[3]), # west,south,east,north. 
                temporal=(start_time, end_time), # the data and time
                count=-1 # registration number obtained count = -1 = all files. ''exchangeable''
            )
            
            create_path = f"{product_name}_{start_time[:4]}_to_{end_time[:4]}" # name a folder for saving your files
            os.makedirs(create_path, exist_ok=True)  # Create a folder for saving your files

            # parallel downloads
            files = earthaccess.download(results, create_path, threads=threads, pqdm_kwargs={"disable": True})

            if files:
                return start_time, True
            else:
                return start_time, False
        except:
            return start_time, False

    # parallezing the downloads with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(tqdm(executor.map(download_single_date, dataframes_list), 
                            total=len(dataframes_list), 
                            desc='Downloading files', 
                            leave=False))

    # Creating a log
    for date, success in results:
        if success:
            data_ok.append(date)
        else: data_not_ok.append(date)

    # save logs
    df1,df2 = pd.DataFrame({'data_ok': data_ok}),pd.DataFrame({'data_not_ok':data_not_ok})
    if df2.empty:
        df2.loc[len(df2)] = 'All the requested files have been downloaded'
        df2 = df2.rename(columns={'data_not_ok': f'###### the logger of {product_name} download from {start_date} until {end_date} is #######'})
        df2.to_csv(f'log_download_{product_name}_{start_date}.txt', index=False)
    else: 
        df1 = df1.rename(columns={'data_ok': f'###### the logger of {product_name} download from {start_date} until {end_date} is #######'})
        df1['status'] = 'OK'
        df1.to_csv(f'log_download_{product_name}_{start_date}.txt', index=False)

    

    print(f'Your download for {start_date} until {end_date} is complete!')
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
def extract_time(data_list:str):
    year,julian_day,hour,minute = int(data_list[:4]), int(data_list[4:7]), int(data_list[7:9]), int(data_list[9:])
    date = datetime(year, 1, 1) + timedelta(days=julian_day - 1)
    month,day = date.month, date.day
    string_year = f'{year}-{month}-{day} {hour}:{minute}:00' # '2002-02-04 15:32:00'
    dt = datetime.strptime(string_year,"%Y-%m-%d %H:%M:%S") # create datetime
    return str(dt) 

########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
################# This functions is useful METRIC GET ACCURRACY OF a product of RETRIEVAL  #############
def expected_error_AOD(aod_station, 
                       aod_estimated,
                       val_percent):
    """
    aod_station: insert aod_reference_station
    aod_estimed: insert aod_estimated
    val_percent: the percent of error
    Checking the envelope of AOD estimated between the expected error (EE),
    about this criteria: AOD - EE <= AOD_modelo <= AOD + EE
    where EE = 0.05 + 0.1 * AOD. https://darktarget.gsfc.nasa.gov/content/what-estimated-error-aod-dark-target-product
    """
    aod_station,aod_estimated = aod_station.dropna(),aod_estimated.dropna()
    expected_error = 0.05 + val_percent * aod_station
    lower_bound = aod_station - expected_error
    upper_bound = aod_station + expected_error
    within_envelope = (aod_estimated >= lower_bound) & (aod_estimated <= upper_bound)
    proportion_within_envelope = within_envelope.sum() / len(within_envelope)
    return proportion_within_envelope*100
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
################# This functions is useful Matching csv file in the Time (time window)  ################
def Organize_time(df_aeronet, 
                           df_satellite,
                           x_col,
                           aeronet_aod_col='AOD_550nm',
                           window_minutes='15min',
                           merge_tolerance='15min',
                           min_samples=2,
                           kind_direction='nearest'):
    """
    This function do the centered mean for each line of a dataframe in a ±window_minutes.  
    Parâmetros:
    - df_aeronet: DataFrame with the data
    - df_satellite: DataFrame with the data
    - x_col: nome of column time
    - aeronet_aod_col: name of var to do mean
    - window_minutes:  the window of interesting each side ex ±15min 
    - merge_tolerance: the tolerance to matching ±15min 
    - min_samples: min of samples to do a mean
    - direction_kind: "backward","forward" and "nearest"
    these function return a dataframe with --> [time_col, value_col+'_mean', 'n_samples']
    """
    df_aero,df_sat = df_aeronet.copy(),df_satellite.copy()
    df_aero[x_col],df_sat[x_col] = pd.to_datetime(df_aero[x_col]), pd.to_datetime(df_sat[x_col])
    df_aero,df_sat = df_aero.sort_values(x_col),df_sat.sort_values(x_col)
    stats = []
    for _, sat_row in df_sat.iterrows():
        t_center = sat_row[x_col]
        t_start = t_center - pd.Timedelta(window_minutes)
        t_end = t_center + pd.Timedelta(window_minutes)
        mask = (df_aero[x_col] >= t_start) & (df_aero[x_col] <= t_end)
        subset = df_aero.loc[mask]
        if len(subset) >= min_samples:
            stats.append({
                'time': t_center,
                'AOD_AERONET_mean': subset[aeronet_aod_col].mean(),
                'AOD_AERONET_std': subset[aeronet_aod_col].std(),
                '440-675_Angstrom_Exponent':subset['440-675_Angstrom_Exponent'].mean(),
                'n_samples': len(subset),
                'window_start': t_start,
                'window_end': t_end
            })
    df_stats = pd.DataFrame(stats)
    df_merged = pd.merge_asof(
        df_sat,
        df_stats,
        on=x_col,
        direction=kind_direction, #type: ignore
        tolerance=pd.Timedelta(merge_tolerance))

    return df_merged
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
################# This functions is useful for interpolate the AOD and Organize AERONET data ###########
def treatments_aeronet(df):
    ''' 
    df : dataframe from aeronet
    '''
    df['Date(dd:mm:yyyy)'] = pd.to_datetime(df['Date(dd:mm:yyyy)'], format='%d:%m:%Y')
    df['Time(hh:mm:ss)'] = pd.to_timedelta(df['Time(hh:mm:ss)'])
    df['AOD_500nm'] = df['AOD_500nm'].replace(-999.0, np.nan)
    df['440-870_Angstrom_Exponent'] = df['440-870_Angstrom_Exponent'].replace(-999.0,np.nan)
    df['AOD_550nm'] = df['AOD_500nm'] * ((550 / 500) ** (-1 * df['440-870_Angstrom_Exponent']))
    df['time'] = df['Date(dd:mm:yyyy)'] + df['Time(hh:mm:ss)']
    df_new = df[['time', 'AOD_550nm']].drop_duplicates()
    return df_new
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
############################## RMSE METRIC #############################################################
def rmse_dataframe(df,var1,var2):
    ''' 
    df: dataframe
    var1: variable 1 - real
    var2: variable 2 - predicted
    '''
    rmse = rmse = (np.mean((df[var1] - df[var2]) ** 2)) ** 0.5
    return rmse
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################