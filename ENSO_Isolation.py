import numpy as np
import xarray as xr
import sacpy as scp
from eofs.standard import Eof
from scipy.stats import linregress

# Define a function to compute Empirical Orthogonal Functions (EOFs)
def EOF(dataa, eofscalingg):
    """
    Computes Empirical Orthogonal Functions (EOFs) and returns related results.

    Parameters:
        dataa: xarray DataArray containing the input data, 
               should have 'lat' and 'lon' coordinates.
        eofscalingg: integer, controls EOF scaling. 
                    0 means no scaling, 1 means scaling EOFs, 2 means scaling PCs.

    Returns:
        tuple: containing the EOF solver, EOF patterns, 
              principal components (PCs), and variance fractions.
    """

    lat = dataa.lat
    coslat = np.cos(np.deg2rad(lat.values)) # Compute cosine of latitudes (for weighting)
    wgts = np.sqrt(coslat)[..., np.newaxis] # Compute weights using square root of cosine values
    solver = Eof(dataa.values, weights=wgts) # Initialize EOF solver with weights

    len_lon, len_lat = len(dataa.lon), len(dataa.lat) # Get dimensions of longitude and latitude

    if eofscalingg == 0: # No scaling
        eof = solver.eofs(neofs=4, eofscaling=eofscalingg) * np.sqrt(len_lat * len_lon)
        pc = solver.pcs(npcs=4, pcscaling=eofscalingg) / np.sqrt(len_lat * len_lon)
    else: # Scale EOFs, PCs use inverse scaling
        eof = solver.eofs(neofs=4, eofscaling=eofscalingg)
        pc = solver.pcs(npcs=4, pcscaling=3 - eofscalingg)
    eof = xr.DataArray(eof, dims=['eof', 'lat', 'lon'], # Wrap 4 EOFs in xarray DataArray with coordinates
                       coords={'eof': [1, 2, 3, 4], 'lat': dataa.lat, 'lon': dataa.lon}) 
    varfrac = solver.varianceFraction(neigs=4)

    return solver, eof, pc, varfrac

# Define a function for ENSO-related analysis
def ENSO_Isolation(dataa):
    """
    Analyzes ENSO-related variability and separates ENSO-related and ENSO-unrelated components.

    Parameters:
        dataa: xarray DataArray containing the input data, 
               should have 'lat', 'lon', and 'time' coordinates.

    Returns:
        dict: containing ENSO-related and ENSO-unrelated components.
    """
    # Define regions of interest for identifying ENSO patterns (El Niño-like/La Niña-like)
    lat0, lat1 = -5, 5
    lone0, lone1 = 180, 250
    lonw0, lonw1 = 90, 160

    
    dataa_de = scp.get_anom(dataa, method=1) # Compute anomalies; Remove seasonal cycle; Detrend
    solver, eof, pc, varf = EOF(dataa_de, 2) # Conduct EOF analysis with scaling parameter 2
    eof0 = eof[0].values
    pattern = np.nan_to_num(eof0.flatten() - np.nanmean(eof0.flatten())) # Get EOF pattern and reshape pattern for projection
  
    dataa_ano = np.nan_to_num(scp.get_anom(dataa, method=0).values.reshape(dataa.shape[0], -1)) # Reshape data with seasonal cycle removed for projection 
    proj = np.dot(dataa_ano, pattern) / solver.eigenvalues(neigs=1)  # Project data onto the EOF pattern
    
    data_re = np.outer(proj, eof0.flatten()).reshape(dataa.shape[0], dataa.shape[1], dataa.shape[2]) # Reconstruct ENSO-related data
    data_re_xr = xr.DataArray(data_re, dims=['time', 'lat', 'lon'], 
                              coords={'time': dataa.time, 'lat': dataa.lat, 'lon': dataa.lon}) # Wrap reconstructed data in xarray DataArray

    
    eof0_xr = xr.DataArray(eof0, dims=['lat', 'lon'], 
                           coords={'lat': dataa.lat, 'lon': dataa.lon}) # Wrap the first EOF pattern in xarray DataArray

    # Adjust PC and EOF signs based on the west-east difference to make El Niño-like pattern  
    west_east = eof0_xr.sel(lon=slice(lonw0, lonw1), lat=slice(lat0, lat1)).mean(dim=['lat', 'lon'], skipna=True) - \
                eof0_xr.sel(lon=slice(lone0, lone1), lat=slice(lat0, lat1)).mean(dim=['lat', 'lon'], skipna=True) 
    if west_east <= 0:
        d_pc1 = proj
        d_eof = eof0_xr
    else:
        d_pc1 = -proj
        d_eof = -eof0_xr


    resultt = {
        'ENSO pattern': d_eof,
        'ENSO time series': d_pc1,
        'ENSO-related': data_re_xr,
        'ENSO-unrelated': dataa - data_re
    }
    return resultt