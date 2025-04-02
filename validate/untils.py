import pandas as pd 
import numpy as np


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