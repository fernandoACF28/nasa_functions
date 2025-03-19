import os
from tqdm import tqdm
import pandas as np
import pandas as pd 
import earthaccess
import rioxarray as rxr
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
                               year_data: str):
    d1 = rxr.open_rasterio(path)[0]
    list = d1.attrs.get('Orbit_time_stamp').split(' ')
    list_datas = [k[:-1] for k in list if len(k) > 0]
    listas_datas = [extract_time(n) for n in list_datas]
    d2 = d1.assign_coords(band=("band", listas_datas))
    d2 = d2.rio.reproject("EPSG:4326")
    d3 = d2.where(d2 != -28672 , float('nan'))*0.001
    d4 = d3.sel(x=lon,y=lat,method='nearest')
    d5 = d4.Optical_Depth_055
    df = d5.drop_vars(['x','y','spatial_ref']).to_dataframe().reset_index()
    df_final = df.rename(columns={'band':'time'})
    os.makedirs(name_folder, exist_ok=True)
    df_final.to_csv(f'd:/MAIAC/scripts MAIAC/{name_folder}/MCD19A2_{year_data}_{station_name}_{index}.csv',index=False)
    del d1,list,list_datas,d2,d3,d4,d5,df,df_final
    
def extract_time(data_list:str):
    year,julian_day,hour,minute = int(data_list[:4]), int(data_list[4:7]), int(data_list[7:9]), int(data_list[9:])
    date = datetime(year, 1, 1) + timedelta(days=julian_day - 1)
    month,day = date.month, date.day
    string_year = f'{year}-{month}-{day} {hour}:{minute}:00' # '2002-02-04 15:32:00'
    dt = datetime.strptime(string_year,"%Y-%m-%d %H:%M:%S") # create datetime
    return str(dt) 