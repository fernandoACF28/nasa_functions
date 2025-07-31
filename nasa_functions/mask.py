import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from glob import glob as gb
import nasa_functions as nasa
from pyproj import CRS, Transformer
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde




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
    def __init__(self,path,var,scale_factor,fillValue,index_hdf_aod,index_hdf_angle,mask,coordinates:tuple[float,float],window_size:int,number_int:int):
        self.path = path
        self.var = var
        self.index_hdf_aod = index_hdf_aod
        self.index_hdf_angle = index_hdf_angle
        self.scale_factor = scale_factor 
        self.fillValue = fillValue
        self.coordinates = coordinates
        self.mask = mask
        self.window_size = window_size
        self.number_int = number_int


    @staticmethod
    def extract_time(data_list:str):
        year,julian_day,hour,minute = int(data_list[:4]), int(data_list[4:7]), int(data_list[7:9]), int(data_list[9:])
        date = datetime(year, 1, 1) + timedelta(days=julian_day - 1)
        month,day = date.month, date.day
        string_year = f'{year}-{month}-{day} {hour}:{minute}:00'
        dt = datetime.strptime(string_year,"%Y-%m-%d %H:%M:%S") 
        return str(dt)
    
    @staticmethod
    def apply_mask(qa_array, mask_name, value_names):
        """
        Applying multiple maks.
        """
        mask_info = BITMASKS[mask_name]
        offset = mask_info["bit_offset"]
        length = mask_info["bit_length"]

        if not isinstance(value_names, list):
            value_names = [value_names]

        mask = xr.zeros_like(qa_array, dtype=bool)
        for value_name in value_names:
            value = mask_info["values"][value_name]
            condition = ((qa_array >> offset) & ((1 << length) - 1)) == value
            mask |= condition
        return mask
    
    @staticmethod
    def sinusoidal_to_latlon(coor: tuple[float, float]):
        """
        convert geographic coordinate (degrees) (lat,lon)
        in sinusoidal coordinates (meters) ----> returning (lon,lat)
        """
        crs_sinu = CRS.from_wkt(wkt)
        crs_geo = CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")
        transformer = Transformer.from_crs(crs_geo, crs_sinu, always_xy=True)
        lon, lat = coor[1], coor[0] 
        x, y = transformer.transform(lon, lat)
        return x, y # -----> (lon, lat)
    
    @staticmethod
    def select_pixels(ds,station: tuple[float, float],window_size):
        lon,lat = Maiac.sinusoidal_to_latlon((station[0],station[1]))
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
    
    def applying_mask(self):
        ds = rxr.open_rasterio(self.path)[self.index_hdf_aod]
        qa = ds['AOD_QA']
        mask_ideal = xr.ones_like(qa, dtype=bool)
        for mask_name, value_name in self.mask.items():
            condition = Maiac.apply_mask(qa, mask_name, value_name)
            mask_ideal &= condition
        new_ds = ds.where(mask_ideal)
        return new_ds
    
    
    def filter_data(self):
        """
        Organizing your file, according to your filter
        """   
        ds = rxr.open_rasterio(self.path)[self.index_hdf_aod]
        qa = ds['AOD_QA']
        mask_ideal = xr.ones_like(qa, dtype=bool)

        for mask_name, value_name in self.mask.items():
            condition = Maiac.apply_mask(qa, mask_name, value_name)
            mask_ideal &= condition
        new_ds = ds.where(mask_ideal)
        ds_new = new_ds.where(mask_ideal).rename({'x': 'lon', 'y': 'lat'})
        ds_new = ds_new[self.var].where(ds_new[self.var] != self.fillValue) * self.scale_factor
        lista = ds.attrs.get('Orbit_time_stamp').split(' ')
        list_datas = [k[:-1] for k in lista if k]
        lista_datas = [Maiac.extract_time(n) for n in list_datas]
        ds_new = ds_new.assign_coords(band=("band", lista_datas))
        ds_new = ds_new.rename({'band': 'time'})
        return ds_new
    @staticmethod
    def extract_csv_from_hdf(ds,
                    coords: tuple[float,float],
                    central_pixel: bool = True,
                    window_size: int = 1):
        '''
        ds -> DataArray
        coords -> tuple (lat,lon)
        central_pixels -> get data around stations only if have data station
        '''
        ds = Maiac.select_pixels(ds,(coords[0],coords[1]),window_size)
        central_value = ds.isel(lat=window_size, lon=window_size)
        is_valid = central_value.notnull()
            
        # create a dataframe for each time
        results = []
        for i in range(len(ds.time)):
            time_val = ds.time[i].values
            current_valid = is_valid[i].item() if len(is_valid.shape) > 0 else is_valid.item()
            result_dict = {
                    'time': time_val,
                    'aod_inst': np.nan,
                    f'mean_aod_{window_size}x{window_size}_px': np.nan,
                    'std_aod': np.nan,
                    'samples': 0
                }
            if current_valid or not central_pixel:
                current_ds = ds.isel(time=i)
                valid_counts = current_ds.count().item()
                result_dict.update({
                        'aod_inst': float(central_value[i].item()),
                        f'mean_aod_{window_size}x{window_size}_px': float(current_ds.mean().item()),
                        'std_aod': float(current_ds.std().item()),
                        'samples': int(valid_counts),
                    })
                
            results.append(result_dict)
            
        return pd.DataFrame(results)
    
    def extract_csv(self):
        """
        Organizing your file, according to your filter
        """  
        ds = self.filter_data()  # Corrigido: usando self em vez de maiac
        data_frame = self.extract_csv_from_hdf(ds, self.coordinates, window_size = self.window_size)
        year_file = self.path[8:-54]
        os.makedirs(f'csv_file/{year_file}_{self.window_size}',exist_ok=True)
        data_frame.to_csv(f'csv_file/{year_file}_{self.window_size}/{year_file}_{self.number_int}.csv')

        return data_frame  # Adicionei um return para que o resultado possa ser armazenado

    

def expected_error_AOD(aod_station, aod_estimated):
    """
    aod_station: insert aod_reference_station
    aod_estimed: insert aod_estimated



    Verifica a proporção de estimativas AOD dentro do intervalo de erro esperado (EE),
    conforme o critério: AOD - EE <= AOD_modelo <= AOD + EE
    onde EE = 0.05 + 0.1 * AOD.
    """
    aod_station,aod_estimated = aod_station.dropna(),aod_estimated.dropna()
    expected_error = 0.05 + 0.15 * aod_estimated
    lower_bound = aod_station - expected_error
    upper_bound = aod_station + expected_error
        
    within_envelope = (aod_estimated >= lower_bound) & (aod_estimated <= upper_bound)
    proportion_within_envelope = within_envelope.sum() / len(within_envelope)
    return proportion_within_envelope*100

class AeroStations:
    def __init__(self,data,x_col,y_col,std_val_x,std_val_y,x_label,y_label,title,axis):
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.std_val_x = std_val_x
        self.std_val_y = std_val_y
        self.title = title 
        self.x_label = x_label 
        self.y_label = y_label
        self.axis = axis

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