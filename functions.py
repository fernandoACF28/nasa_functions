import os
from tqdm import tqdm
import numpy as np
import pandas as pd 
import earthaccess
import rioxarray as rxr
import xarray as xr
from haversine import haversine, Unit
from glob import glob as gb
from datetime import datetime,timedelta
from warnings import filterwarnings
from concurrent.futures import ThreadPoolExecutor
filterwarnings('ignore')





# This function is util for create a temporal series for do download.

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

#  This fuction is util for download of products from NASA.

def downloads_files_nasa(start_date: str, 
                         end_date: str,
                         product_name: str, 
                         coordinates: tuple[int, ...],
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
        except Exception as e:
            print(f"Error in downloading {start_time}: {e}")
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


def extract_csv(lat,lon,path_csv):
    df = ds.sel(lat=lat,lon=lon,method='nearest').to_dataframe().drop(columns={'lat','lon','spatial_ref'}).reset_index() # Here, I set the lat and lon closer the station.
    df.to_csv(path_csv,index=False) # Here, I'm saving all csv files


def extract_csv_files_from_HDF(path:str,
                               index: int,
                               lat: float,
                               lon: float,
                               station_name: str,
                               name_folder: str,
                               year_data: str,
                               var_select: str,
                               radius: int = None,
                               path_finaly_csv: str = None):
    var = var_select
    d1 = rxr.open_rasterio(path)[0]
    list = d1.attrs.get('Orbit_time_stamp').split(' ')
    list_datas = [k[:-1] for k in list if len(k) > 0]
    listas_datas = [extract_time(n) for n in list_datas]
    d2 = d1.assign_coords(band=("band", listas_datas))
    d2 = d2.rio.reproject("EPSG:4326")
    d2 = d2.rename({'x':'lon','y':'lat'})
    d2 = d2[var]
    d3 = d2.where(d2 != -28672 , float('nan'))*0.001
    if radius != None:
        lat_values,lon_values = d3.lat.values,d3.lon.values 
        mask = np.zeros((len(lat_values),len(lon_values)),dtype=bool)
        station_coords = (lat,lon)
        for i,lat in enumerate(lat_values):
            for j, lon in enumerate(lon_values):
                distance = haversine(station_coords,(lat,lon),unit=Unit.KILOMETERS)
                if distance <= radius:
                    mask[i,j] = True     
        mask_da = xr.DataArray(mask,coords={'lat':lat_values,'lon':lon_values},dims=['lat','lon'])
        d4 = d3.where(mask_da,drop=True)
        mean_each_time = d4.mean(dim=("lat", "lon"))
        df = mean_each_time.drop_vars(['spatial_ref']).to_dataframe().reset_index()
        df_final = df.rename(columns={'band':'time',var : f'{var}_{str(radius)}km'})
    else:
        d4 = d3.sel(x=lon,y=lat,method='nearest')
        d4 = d4.Optical_Depth_055
        df = d4.drop_vars(['x','y','spatial_ref']).to_dataframe().reset_index()
        df_final = df.rename(columns={'band':'time'})
    os.makedirs(name_folder, exist_ok=True)
    if path_finaly_csv == None: df_final.to_csv(f'{name_folder}/MCD19A2_{year_data}_{station_name}_{index}.csv',index=False)
    else: df_final.to_csv(f'{path_finaly_csv}/{name_folder}/MCD19A2_{year_data}_{station_name}_{index}.csv',index=False)
    del d1,list,list_datas,d2,d3,d4,df,df_final


def filter_by_radius(ds, station_coords, radius):
        distances = xr.apply_ufunc(
            lambda lat, lon: haversine(station_coords, (lat, lon), unit=Unit.KILOMETERS),
            ds.lat,
            ds.lon,
            vectorize=True
        )
        return ds.where(distances <= radius).mean(dim=("lat", "lon"))

def extract_csv_from_NetCDF(path: str,
                         index: str,
                         station_lat: str,
                         station_lon: str,
                         year_data:str,
                         station_name:str,
                         radius_km: int,
                         folder_csv: str):
    ''' 
    path: the directory of your file xarray
    index: number of your file
    station_lat: the latitude of interesting
    station_lon: the longitutde of interesting
    year_data: the year of file for organize 
    station_name: name of your station interesting
    radius_km: the radius in kilometer for make the mean
    folder_csv: name of folder to save your files  
    '''
    try:
        ds = xr.open_dataset(path)
        mean_within_radius = filter_by_radius(ds, (station_lat, station_lon), radius_km)
        df = mean_within_radius.to_dataframe().reset_index()
    except: pass
    folder_csv = folder_csv
    os.makedirs(folder_csv, exist_ok=True)
    try: 
        df.to_csv(f'{folder_csv}/{year_data}_{station_name}_{index}.csv',index=False)
    except: pass
    del ds,df

def extract_time(data_list:str):
    year,julian_day,hour,minute = int(data_list[:4]), int(data_list[4:7]), int(data_list[7:9]), int(data_list[9:])
    date = datetime(year, 1, 1) + timedelta(days=julian_day - 1)
    month,day = date.month, date.day
    string_year = f'{year}-{month}-{day} {hour}:{minute}:00' # '2002-02-04 15:32:00'
    dt = datetime.strptime(string_year,"%Y-%m-%d %H:%M:%S") # create datetime
    return str(dt) 



def expected_error_AOD(aod_station, 
                       aod_estimated):
    """
    Calculate the error and the diference between expected AOD and estimate
    with respect AERONET Station.
    """
    expected_aod = aod_station * 0.1 + 0.05 # https://doi.org/10.1016/j.atmosenv.2021.118659
    error_diff = aod_estimated - expected_aod
    relative_error_percentage = (abs(error_diff) / expected_aod) * 100
    mean_error_percentage = relative_error_percentage.mean() if hasattr(relative_error_percentage, 'mean') else relative_error_percentage
    return mean_error_percentage

def mergin_csv_in_one(df1,
                      df2):
    '''
    df1 : the first dataframe with you want to merge
    df2: the second dataframe with you want to merge   
    '''
    df1["time"], df2["time"] = pd.to_datetime(df1["time"]),pd.to_datetime(df2["time"])
    df1, df2 = df1.drop_duplicates(subset=["time"]), df2.drop_duplicates(subset=["time"])
    df1, df2 = df1.sort_values("time"),df2.sort_values("time")
    df_merged = pd.merge_asof(
        df1, df2, 
        on="time", 
        direction="backward",  # Ou "forward" ou "nearest"
        tolerance=pd.Timedelta("10min")  # tolerance
    )
    return df_merged.dropna().drop(columns='time') # reset index optional

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


def rmse_dataframe(df,var1,var2):
    ''' 
    df: dataframe
    var1: variable 1 - real
    var2: variable 2 - predicted
    '''
    rmse = rmse = (np.mean((df[var1] - df[var2]) ** 2)) ** 0.5
    return rmse



def extract_csv_files_from_HDF_STD(path: str,
                               index: int,
                               lat: float,
                               lon: float,
                               station_name: str,
                               name_folder: str,
                               year_data: str,
                               var_select: str,
                               radius: int = None,
                               path_finaly_csv: str = None):
    var = var_select
    d1 = rxr.open_rasterio(path)[0]
    list = d1.attrs.get('Orbit_time_stamp').split(' ')
    list_datas = [k[:-1] for k in list if len(k) > 0]
    listas_datas = [extract_time(n) for n in list_datas]
    
    d2 = d1.assign_coords(band=("band", listas_datas))
    d2 = d2.rio.reproject("EPSG:4326")
    d2 = d2.rename({'x':'lon', 'y':'lat'})
    d2 = d2[var]
    d3 = d2.where(d2 != -28672, float('nan')) * 0.001

    if radius is not None:
        lat_values, lon_values = d3.lat.values, d3.lon.values 
        mask = np.zeros((len(lat_values), len(lon_values)), dtype=bool)
        station_coords = (lat, lon)

        for i, lat_val in enumerate(lat_values):
            for j, lon_val in enumerate(lon_values):
                distance = haversine(station_coords, (lat_val, lon_val), unit=Unit.KILOMETERS)
                if distance <= radius:
                    mask[i, j] = True     
        
        mask_da = xr.DataArray(mask, coords={'lat': lat_values, 'lon': lon_values}, dims=['lat', 'lon'])
        d4 = d3.where(mask_da, drop=True)

        # Média e desvio padrão espacial
        mean_each_time = d4.mean(dim=("lat", "lon"))
        std_each_time = d4.std(dim=("lat", "lon"))

        # DataFrame
        df_mean = mean_each_time.drop_vars(['spatial_ref']).to_dataframe().reset_index()
        df_std = std_each_time.to_dataframe().reset_index()

        df_final = df_mean.merge(df_std, on='band')
        df_final = df_final.rename(columns={
            'band': 'time',
            var: f'{var}_{str(radius)}km',
            f'{var}_std': f'{var}_{str(radius)}km_std'
        })

    else:
        # Valor pontual
        d4 = d3.sel(lon=lon, lat=lat, method='nearest')
        df = d4.drop_vars(['lon', 'lat', 'spatial_ref']).to_dataframe().reset_index()
        df_final = df.rename(columns={'band': 'time', var: f'{var}_point'})


    # Salvar CSV
    os.makedirs(name_folder, exist_ok=True)
    output_path = f'{name_folder}/MCD19A2_{year_data}_{station_name}_{index}.csv' \
        if path_finaly_csv is None else f'{path_finaly_csv}/{name_folder}/MCD19A2_{year_data}_{station_name}_{index}.csv'
    
    df_final.to_csv(output_path, index=False)

    # Limpeza
    del d1, list, list_datas, d2, d3, d4, df_final