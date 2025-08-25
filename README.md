# nasa_functions

This package provides utilities for downloading and extracting temporal series from HDF files, specifically for NASA data.

First it's necessary prepare your environment!!!!

Install in your virtual enviroment (use python version 3.10)
``` bash 
conda create -n name_your_enviroment python=3.10
```

First install the package manager ultra violet (uv) in your virtual environment.

``` bash 
pip install uv
```

For all the libraries:

``` bash 
uv pip install pyproj rioxarray geopandas xarray tqdm haversine earthaccess seaborn pyarrow fastparquet
```
for the libraries gdal and libgdal-hdf4 you need install using conda.
``` bash 
conda install -c conda-forge gdal=3.10 rasterio libgdal-hdf4
```

## Installation

To install this package, you can use `pip`:

```bash
uv pip install git+https://github.com/fernandoACF28/nasa_functions.git

```
# For using the process_hdf_FILES().
## Here an example how to get temporal series from hdf files
Here, we have one station with lat and lon. We're open all files and get your respective temporal series.
``` bash 
maiac = Maiac(path=path,
                        var='Optical_Depth_055',
                        scale_factor=0.001,
                        fillValue=-28672,
                        index_hdf_aod=0,
                        index_hdf_angle=1,
                        mask = mask_dict,
                        coordinates=(-23.561500, -46.734983),
                        window_size=window_size,
                        number_int=1)
maiac.extract_csv()
```
    

# Upgrade functions

``` bash 
uv pip install --upgrade --force-reinstall git+https://github.com/fernandoACF28/nasa_functions.git
```



