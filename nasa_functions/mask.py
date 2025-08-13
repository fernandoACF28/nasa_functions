import os
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import rioxarray as rxr
from scipy.stats import t
from glob import glob as gb
import nasa_functions as nasa
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.lines import Line2D
from pyproj import CRS, Transformer
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta




BITMASKS = {
    "Cloud Mask": {
        "bit_offset": 0,
        "bit_length": 3,
        "values": {
            "undefined": 0b000,
            "clear": 0b001,
            "possibly_cloudy": 0b010,
            "cloudy": 0b011,
            "cloud_shadow": 0b101,
            "hot_spot": 0b110,
            "water_sediments": 0b111,
        },
    },
    "Land/Water Mask": {
        "bit_offset": 3,
        "bit_length": 2,
        "values": {
            "land": 0b00,
            "water": 0b01,
            "snow": 0b10,
            "ice": 0b11,
        },
    },
    "Glint Mask": {
        "bit_offset": 12,
        "bit_length": 1,
        "values": {
            "no_glint": 0b0,
            "glint": 0b1,
        },
    },
    "QA AOD": {
        "bit_offset": 8,
        "bit_length": 4,
        "values": {
            "best_quality": 0b0000,
            "Water_sediments":0b0001,
            "neighbor_cloud_one":0b0011,
            "neighbor_cloud_taller_one":0b0100,
            "no_retrieval_cloudy":0b0101,
            "aod_retrieved_CM_cloudy": 0b1011
        },
    },
    }

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

def calcular_aod_550(df, colunas_aod, comprimentos_onda_nm):
    """
    Interpola a AOD para 550 nm usando o modelo log-log quadrático de Lyapustin.
    
    Parâmetros:
    - df: DataFrame com colunas de AOD
    - colunas_aod: lista com os nomes das colunas de AOD (ex: ['AOD_440', 'AOD_500', 'AOD_675'])
    - comprimentos_onda_nm: lista com os comprimentos de onda correspondentes às colunas (ex: [440, 500, 675])
    
    Retorna:
    - Uma cópia do DataFrame com uma nova coluna: 'AOD_550_est'
    """

    ln_lambda = np.log(comprimentos_onda_nm)
    ln_target = np.log(550)

    def estima_linha(row):
        try:
            aod_vals = np.array([row[col] for col in colunas_aod])
            if np.any(aod_vals <= 0):  # para evitar log de zero ou valores negativos
                return np.nan

            ln_aod = np.log(aod_vals)
            coef = np.polyfit(ln_lambda, ln_aod, 2)  # [beta_2, beta_1, beta_0]
            ln_aod_550 = coef[0]*ln_target**2 + coef[1]*ln_target + coef[2]
            return np.exp(ln_aod_550)
        except Exception as e:
            return np.nan  # retorna NaN se houver erro
    df['Date(dd:mm:yyyy)'] = pd.to_datetime(df['Date(dd:mm:yyyy)'], format='%d:%m:%Y')
    df['Time(hh:mm:ss)'] = pd.to_timedelta(df['Time(hh:mm:ss)'])
    df['time'] = df['Date(dd:mm:yyyy)'] + df['Time(hh:mm:ss)']
    df = df.copy()
    df['AOD_550nm_est'] = df.apply(estima_linha, axis=1)
    df['440-675_Angstrom_Exponent'] = df['440-675_Angstrom_Exponent'].replace(-999.0,np.nan)
    df_new = df[['time','AOD_550nm_est','440-675_Angstrom_Exponent']]
    return df_new.dropna()


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
        lat,lon = station[0],station[1]
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
                    center_value = arr[win][win]
                    if isinstance(center_value, (int, float, np.integer, np.floating)) and not np.isnan(center_value):
                        dictionary_station[f'inst_aod'] = center_value
                        if isinstance(valid_win, int) and valid_pixels >= valid_win:
                            val_mean, val_std = classe.get_mean_and_STD(ds_filled, i_time)
                            dictionary_station[f'mean_px_{win}x{win}'] = val_mean
                            dictionary_station[f'std_px_{win}x{win}'] = val_std
                        else:
                            dictionary_station[f'mean_px_{win}x{win}'] = np.nan
                            dictionary_station[f'std_px_{win}x{win}'] = np.nan
                            sequence_broken = True
                    else: pass
                data_list.append(dictionary_station)
        
        df = pd.DataFrame(data_list)
        os.makedirs('Parquet_datas', exist_ok=True)     
        df.to_parquet(f'Parquet_datas/{self.product_name}_{self.year}_{self.index_iteraction}.parquet',index=False) # aod_estimated

def expected_error_AOD(aod_station, aod_estimated):  # aod_estimated
            """
            aod_station: insert aod_reference_station
            aod_estimed: insert aod_estimated



            Verifica a proporção de estimativas AOD dentro do intervalo de erro esperado (EE),
            conforme o critério: AOD - EE <= AOD_modelo <= AOD + EE
            onde EE = 0.05 + 0.1 * AOD.
            """
            aod_station,aod_estimated = aod_station.dropna(),aod_estimated.dropna()
            expected_error = 0.05 + 0.15 * aod_station
            lower_bound = aod_station - expected_error
            upper_bound = aod_station + expected_error
                
            within_envelope = (aod_estimated >= lower_bound) & (aod_estimated <= upper_bound)
            proportion_within_envelope = within_envelope.sum() / len(within_envelope)
            return proportion_within_envelope*100

class AeroStations:
    def __init__(self,data,x_col,y_col,std_val_x,std_val_y,x_label,y_label,title,axis,despine=True):
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.std_val_x = std_val_x
        self.std_val_y = std_val_y
        self.title = title 
        self.x_label = x_label 
        self.y_label = y_label
        self.axis = axis
        self.despine = despine

    def Plots_aero_vs_MCD(self,density_min,density_max):
        # params from error bar 
        errorbar_kwargs = {
            'xerr': self.data[self.std_val_x] if self.std_val_x in self.data else None,
            'yerr': self.data[self.std_val_y] if self.std_val_y in self.data else None,
            'fmt': 'o',
            'color': 'none',
            'markersize': 3,
            'ecolor': 'gray',
            'alpha': 0.3,
            'capsize': 2 }
        if errorbar_kwargs['xerr'] is None: del errorbar_kwargs['xerr']
        if errorbar_kwargs['yerr'] is None: del errorbar_kwargs['yerr']
        self.axis.errorbar(self.data[self.x_col],
            self.data[self.y_col],**errorbar_kwargs)
        # retirate line from plot left and bootom 
    
        if self.despine == True:
             sns.despine(left=False, bottom=False)
        else: pass 
        # Useful metrics
        ee = expected_error_AOD(self.data[self.x_col],self.data[self.y_col])
        rmse = nasa.rmse_dataframe(self.data,self.x_col,self.y_col)
        n_samples = len(self.data)
        mask = ~np.isnan(self.data[self.x_col]) & ~np.isnan(self.data[self.y_col])
        x, y = self.data[self.x_col][mask], self.data[self.y_col][mask]
        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)
        density_min = 0

        # density_min,density_max = density.min(),density.max()
        density = (density - density_min) / (density_max - density_min)
        min_val = min(self.data[self.x_col].min(),self.data[self.y_col].min())
        max_val = max(self.data[self.x_col].max(),self.data[self.y_col].max())
        # Importants plots first (regplot,scatter(with density),line 1:1)
        sns.regplot(self.data,x=self.x_col,y=self.y_col,ci=95,color='darkred',
                    scatter=False,
                    ax=self.axis,
                    line_kws={'linewidth':0.7},label='Linear Fit')
        sc = self.axis.scatter(x, y, c=density, cmap='hot', s=4, edgecolors='none',zorder=2,vmin=0,vmax=1)
        line1to1 = self.axis.plot([min_val,max_val],[min_val,max_val],c='lightsteelblue',linestyle='--',label='1x1 line',lw=0.7)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        alpha = 0.05
        df = len(x) - 2
        t_crit = t.ppf(1 - alpha/2, df)
        slope_ci = t_crit * std_err
        if intercept >=0:
            equation = f'Y = ({slope:.3f} ± {slope_ci:.3f})X + {intercept:.3f}'
        else:
            equation = f'Y = ({slope:.3f} ± {slope_ci:.3f})X {intercept:.3f}'
        self.axis.text(0.05, 0.95,
            f'R: {r_value:.3}'r' $p_{\mathrm{val}}$: 'f'{p_value:.2f}\nn: {n_samples}\nEE: {ee:.2f}% \nR$^{2}$: {r_value**2:.2f}\nRMSE: {rmse:.2f}\nstd_err: {std_err:.2}\n{equation}',
            transform=self.axis.transAxes, fontsize=10,fontdict={'fontfamily':'serif'},
            verticalalignment='top')
        
        error_caps = Line2D([0], [0],
                            color='gray',
                            linestyle='none',
                            marker='_',  # horizontal cap
                            markersize=10,
                            markeredgewidth=1.5,
                            alpha=0.5,
                            label='STD_Value'
                        )
        handles, labels = self.axis.get_legend_handles_labels()
        handles.extend([line1to1, error_caps])
        labels.extend(['1:1 Line', 'STD_Value'])
        self.axis.legend(handles=handles, labels=labels)

        # Title and Legends
        self.axis.set_title(self.title,fontname='serif',fontsize=12)
        self.axis.set_xlabel(self.x_label,fontname='serif',fontsize=12)
        self.axis.set_ylabel(self.y_label,fontname='serif',fontsize=12)
        return sc 

    def Plots_aero_vs_MCD_single_plot(self):
        # params from error bar 
        errorbar_kwargs = {
            'xerr': self.data[self.std_val_x] if self.std_val_x in self.data else None,
            'yerr': self.data[self.std_val_y] if self.std_val_y in self.data else None,
            'fmt': 'o',
            'color': 'none',
            'markersize': 3,
            'ecolor': 'gray',
            'alpha': 0.3,
            'capsize': 2 }
        if errorbar_kwargs['xerr'] is None: del errorbar_kwargs['xerr']
        if errorbar_kwargs['yerr'] is None: del errorbar_kwargs['yerr']
        self.axis.errorbar(self.data[self.x_col],
            self.data[self.y_col],**errorbar_kwargs)
        # retirate line from plot left and bootom 
        #sns.despine(left=False, bottom=False)
        # Useful metrics
        ee = expected_error_AOD(self.data[self.x_col],self.data[self.y_col])
        rmse = nasa.rmse_dataframe(self.data,self.x_col,self.y_col)
        n_samples = len(self.data)
        mask = ~np.isnan(self.data[self.x_col]) & ~np.isnan(self.data[self.y_col])
        x, y = self.data[self.x_col][mask], self.data[self.y_col][mask]
        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)
        density_min = 0
        density_min,density_max = density.min(),density.max()
        density = (density - density_min) / (density_max - density_min)
        min_val = min(self.data[self.x_col].min(),self.data[self.y_col].min())
        max_val = max(self.data[self.x_col].max(),self.data[self.y_col].max())
        # Importants plots first (regplot,scatter(with density),line 1:1)
        sns.regplot(self.data,x=self.x_col,y=self.y_col,ci=95,color='darkred',
                    scatter=False,
                    ax=self.axis,
                    line_kws={'linewidth':0.7},label='Linear Fit')
        sc = self.axis.scatter(x, y, c=density, cmap='hot', s=4, edgecolors='none',zorder=2,vmin=0,vmax=1)
        line1to1 = self.axis.plot([min_val,max_val],[min_val,max_val],c='lightsteelblue',linestyle='--',label='1x1 line',lw=0.7)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        equation = f'Y = {slope:.3f}X + {intercept:.3f}'
        self.axis.text(0.05, 0.95,
            f'R: {r_value:.3}'r' $p_{\mathrm{val}}$: 'f'{p_value:.2f}\nn: {n_samples}\nEE: {ee:.2f}% \nR$^{2}$: {r_value**2:.2f}\nRMSE: {rmse:.2f}\nstd_err: {std_err:.2}\n{equation}',
            transform=self.axis.transAxes, fontsize=10,fontdict={'fontfamily':'serif'},
            verticalalignment='top')
        
        error_caps = Line2D([0], [0],
                            color='gray',
                            linestyle='none',
                            marker='_',  # horizontal cap
                            markersize=10,
                            markeredgewidth=1.5,
                            alpha=0.5,
                            label='STD_Value'
                        )
        handles, labels = self.axis.get_legend_handles_labels()
        handles.extend([line1to1, error_caps])
        labels.extend(['1:1 Line', 'STD_Value'])
        self.axis.legend(handles=handles, labels=labels)

        # Title and Legends
        self.axis.set_title(self.title,fontname='serif',fontsize=12)
        self.axis.set_xlabel(self.x_label,fontname='serif',fontsize=12)
        self.axis.set_ylabel(self.y_label,fontname='serif',fontsize=12)
        return sc
    def PlotAeroxMCD(self,AE,cmap='hot',size=4):
        # params from error bar 
        
        errorbar_kwargs = {
            'xerr': self.data[self.std_val_x] if self.std_val_x in self.data else None,
            'yerr': self.data[self.std_val_y] if self.std_val_y in self.data else None,
            'fmt': 'o',
            'color': 'none',
            'markersize': 3,
            'ecolor': 'gray',
            'alpha': 0.3,
            'capsize': 2 }
        if errorbar_kwargs['xerr'] is None: del errorbar_kwargs['xerr']
        if errorbar_kwargs['yerr'] is None: del errorbar_kwargs['yerr']
        self.axis.errorbar(self.data[self.x_col],
            self.data[self.y_col],**errorbar_kwargs)
        # retirate line from plot left and bootom 
    
        if self.despine == True:
             sns.despine(left=False, bottom=False)
        else: pass 
        # Useful metrics
        ee = expected_error_AOD(self.data[self.x_col],self.data[self.y_col])
        rmse = nasa.rmse_dataframe(self.data,self.x_col,self.y_col)
        n_samples = len(self.data)
        mask = ~np.isnan(self.data[self.x_col]) & ~np.isnan(self.data[self.y_col])
        x, y = self.data[self.x_col][mask], self.data[self.y_col][mask]
    
        # Importants plots first (regplot,scatter(with density),line 1:1)
        sns.regplot(self.data,x=self.x_col,y=self.y_col,ci=95,color='darkred',
                    scatter=False,
                    ax=self.axis,
                    line_kws={'linewidth':0.7},label='Linear Fit')
        sc = self.axis.scatter(x, y, c=AE, cmap=cmap, s=size, edgecolors='none',zorder=2)
        min_val = min(self.data[self.x_col].min(),self.data[self.y_col].min())
        max_val = max(self.data[self.x_col].max(),self.data[self.y_col].max())
        line1to1 = self.axis.plot([min_val,max_val],[min_val,max_val],c='lightsteelblue',linestyle='--',label='1x1 line',lw=0.7)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        alpha = 0.05
        df = len(x) - 2
        t_crit = t.ppf(1 - alpha/2, df)
        slope_ci = t_crit * std_err
        if intercept >=0:
            equation = f'Y = ({slope:.3f} ± {slope_ci:.3f})X + {intercept:.3f}'
        else:
            equation = f'Y = ({slope:.3f} ± {slope_ci:.3f})X {intercept:.3f}'
        self.axis.text(0.05, 0.95,
            f'R: {r_value:.3}'r' $p_{\mathrm{val}}$: 'f'{p_value:.2f}\nn: {n_samples}\nEE: {ee:.2f}% \nR$^{2}$: {r_value**2:.2f}\nRMSE: {rmse:.2f}\nstd_err: {std_err:.2}\n{equation}',
            transform=self.axis.transAxes, fontsize=10,fontdict={'fontfamily':'serif'},
            verticalalignment='top')
        
        error_caps = Line2D([0], [0],
                            color='gray',
                            linestyle='none',
                            marker='_',  # horizontal cap
                            markersize=10,
                            markeredgewidth=1.5,
                            alpha=0.5,
                            label='STD_Value'
                        )
        handles, labels = self.axis.get_legend_handles_labels()
        handles.extend([line1to1, error_caps])
        labels.extend(['1:1 Line', 'STD_Value'])
        self.axis.legend(handles=handles, labels=labels)

        # Title and Legends
        self.axis.set_title(self.title,fontname='serif',fontsize=12)
        self.axis.set_xlabel(self.x_label,fontname='serif',fontsize=12)
        self.axis.set_ylabel(self.y_label,fontname='serif',fontsize=12)
        return sc 