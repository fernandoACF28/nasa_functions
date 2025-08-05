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
############### util class to extract from MAIAC parquet datas; ########################################
import rioxarray as rxr
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from pyproj import CRS, Transformer
import os
import pandas as pd


class Maiac:
    def __init__(self, path_HDF : str,
                 var_ds: str,
                 hdf_index: str,
                 coordinate_time: str,
                 dict_stations: dict,
                 window_sizes: list[int],
                 year: str,
                 index_iteraction:str,
                 product_name: str,
                 _FillValue:int ,
                 scale_factor:int):
        self.path_HDF = path_HDF
        self.var_ds = var_ds
        self.hdf_index = hdf_index
        self.coordinate_time = coordinate_time
        self.dict_stations = dict_stations 
        self.window_sizes = window_sizes
        self.year = year
        self.index_iteraction = index_iteraction 
        self.product_name = product_name 
        self._FillValue = _FillValue
        self.scale_factor = scale_factor

    @staticmethod
    def select_pixels(ds,
                    station: tuple[float, float],
                    window_size:int):
        lat,lon = station[1],station[0]
        lat_idx = np.abs(ds.lat.values - lat).argmin()
        lon_idx = np.abs(ds.lon.values - lon).argmin()
        lat_start = max(0, lat_idx - window_size)
        lat_end = min(len(ds.lat), lat_idx + window_size + 1)  
        lon_start = max(0, lon_idx - window_size)
        lon_end = min(len(ds.lon), lon_idx + window_size + 1)
        DatArray = ds.isel(
                        lat=slice(lat_start, lat_end),
                        lon=slice(lon_start, lon_end))
        return DatArray
    @staticmethod
    def get_mean_and_STD(ds_filled,time_index):
        """
        ds_filled -->> data spatial filtered around the station.
        var -->> name of var of interesting
        time_index -->> it's important, because you have diferents arrays in time,
        you need to specify the index of time.
        """
        val_mean = ds_filled.isel(time=time_index).mean().values.item()
        std_val = ds_filled.isel(time=time_index).std().values.item()
        return val_mean,std_val
    @staticmethod
    def valid_windows(number):
        """
        number --->> win_size
        if my window size is equal to 1, i need to check the pixels around
        the station, for that, its only add plus 2 and take square, after that i have 
        the number of values in matrix (win_size+2)**2 = 9 pixels with center value.
        But we need only half of samples avaible, for that, we take minus one pixel (central_value) and divided by/2.
        """
        number += 2  
        return int((number**2-1)/2)
    @staticmethod
    def filter_data(path_hdf,var,fillValue,scale_factor,index_hdf):
        # open_file
        ds = rxr.open_rasterio(path_hdf)[index_hdf]
        # until masks
        qa = ds['AOD_QA']
        qa_aod = ((qa >> 8) & 0b1111) == 0b0000
        cloud_clear = (qa & 0b00000111) == 0b001
        no_glint = ((qa >> 12) & 0b1) == 0
        on_land = ((qa >> 3) & 0b11) == 0b00
        mask_ideal = qa_aod & cloud_clear & no_glint & on_land
        # aplying the mask 
        new_ds = ds[var].where(ds[var] != fillValue)*scale_factor
        new_ds = new_ds.where(mask_ideal).rename({'x':'lon','y':'lat'})
        # get time
        lista = ds.attrs.get('Orbit_time_stamp').split(' ')
        list_datas = [k[:-1] for k in lista if len(k) > 0]
        @staticmethod
        def extract_time(data_list:str):
                year,julian_day,hour,minute = int(data_list[:4]), int(data_list[4:7]), int(data_list[7:9]), int(data_list[9:])
                date = datetime(year, 1, 1) + timedelta(days=julian_day - 1)
                month,day = date.month, date.day
                string_year = f'{year}-{month}-{day} {hour}:{minute}:00' 
                dt = datetime.strptime(string_year,"%Y-%m-%d %H:%M:%S") 
                return str(dt)
        lista_datas = [extract_time(n) for n in list_datas]
        new_ds = new_ds.assign_coords(band=("band", lista_datas))
        new_ds = new_ds.rename({'band':'time'})
        return new_ds
    @staticmethod
    def lat_to_Sinusoidal(coor: tuple[float, float]):
            """
            Converte coordenadas geográficas (lat, lon) para projeção sinusoidal (y, x em metros)
            """
            wkt = """PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",
            DATUM["Not specified (based on custom spheroid)",
            SPHEROID["Custom spheroid",6371007.181,0]],
            PRIMEM["Greenwich",0],
            UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],
            PROJECTION["Sinusoidal"],
            PARAMETER["longitude_of_center",0],
            PARAMETER["false_easting",0],
            PARAMETER["false_northing",0],
            UNIT["Meter",1],
            AXIS["Easting",EAST],
            AXIS["Northing",NORTH]]"""
            
            crs_sinu = CRS.from_wkt(wkt)
            crs_geo = CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")
            transformer = Transformer.from_crs(crs_geo, crs_sinu, always_xy=True)
            
            lon, lat = coor[1], coor[0]
            lon, lat = transformer.transform(lon, lat)
            return lat,lon
    
    def Getting_dataframe_from_netCDF_sequential(self,classe):
        """
        path_hdf -> path of your hdf file
        hdf_index -> index of your hdf_file
        coordinate_time -> name of your time coordinate 
        dict_stations -> dicionary of your stations
        window_sizes -> number of sizes of pixels around station
        year - > year for diferenciate the outpufiles
        index_iteraction -> number of file for diferenciate the output files
        """ 
        ds = classe.filter_data(path_hdf=self.path_HDF,
                         var=self.var_ds,
                         fillValue=self._FillValue,
                         scale_factor=self.scale_factor,
                         index_hdf=self.hdf_index)
        dict_stations = self.dict_stations
        data_list,size_time = [], len(ds.time)
        for station in self.dict_stations:
            coords_station, station_name = classe.lat_to_Sinusoidal(dict_stations[station]), station # lat_to_Sinusoidal(dict_stations[station])
            for i_time in range(size_time):
                dictionary_station = {
                    'station': station_name,
                    f'{self.coordinate_time}': ds[f'{self.coordinate_time}'].values[i_time]
                }
                sequence_broken = False 
                for win in sorted(self.window_sizes): 
                    
                    if sequence_broken:
                        dictionary_station[f'mean_px_{win}x{win}'] = np.nan
                        dictionary_station[f'std_px_{win}x{win}'] = np.nan
                        continue
                    
                    ds_filled = classe.select_pixels(ds=ds,
                                            station=coords_station,
                                            window_size=win)
                    
                    arr = ds_filled.isel(time=i_time).values
                    valid_pixels = int(np.sum(~np.isnan(arr)))
                    valid_win = classe.valid_windows(win)
                    dictionary_station[f'inst_aod'] = arr[win][win]
            
                    if isinstance(valid_win, int) and valid_pixels >= valid_win:
                        val_mean, val_std = get_mean_and_STD(ds_filled, i_time)
                        dictionary_station[f'mean_px_{win}x{win}'] = val_mean
                        dictionary_station[f'std_px_{win}x{win}'] = val_std
                    else:
                        dictionary_station[f'mean_px_{win}x{win}'] = np.nan
                        dictionary_station[f'std_px_{win}x{win}'] = np.nan
                        sequence_broken = True 
                data_list.append(dictionary_station)
        
        df = pd.DataFrame(data_list)
        os.makedirs('Parquet_datas', exist_ok=True)     
        df.to_parquet(f'Parquet_datas/{self.product_name}_{self.year}_{self.index_iteraction}.parquet',index=False)