# nasa_functions

This package provides utilities for downloading and extracting temporal series from HDF files, specifically for NASA data.

## Installation

To install this package, you can use `pip`:

```bash
pip install git+https://github.com/fernandoACF28/nasa_functions.git

```

## Here an example how to get temporal series from hdf files
Here, we have 2 stations with lat and lon. We're open all files and get your respective temporal series.

```bash
from glob import glob as gb
from functions  import *
files_Hdf = gb(r'g:/MAIAC/MCD19A2_2002_to_2002/*.hdf')

for file,i in tqdm(zip(files_Hdf,range(1,len(files_Hdf),1))):
    extract_csv_files_from_HDF(file,i,
                               -23.6444,
                               -46.4964,
                               '2002',
                               '2002')
```
# Upgrade functions

``` bash 
pip install --upgrade --force-reinstall git+https://github.com/fernandoACF28/nasa_functions.git
```
# For using the process_hdf_FILES().

Install in your virtual enviroment (use python version 3.10)
``` bash 
conda create -n name_your_enviroment python=3.10

conda install -c conda-forge rasterio pyproj rioxarray geopandas gdal xarray libgdal-hdf4 tqdm ipywidgets haversine
```
