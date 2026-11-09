import numpy as np
import xarray as xr

file_path = "data/weatherdata.nc"

ds = xr.open_dataset(file_path, engine="netcdf4")
wind = ds[["u", "v"]].isel(pressure_level=0)

# Building the matrix
u = wind["u"].transpose("valid_time", "latitude", "longitude")
v = wind["v"].transpose("valid_time", "latitude", "longitude")

nt = u.sizes["valid_time"]
U = u.values.reshape(nt, -1).T.astype(np.float32)   
V = v.values.reshape(nt, -1).T.astype(np.float32)   

X = np.vstack([U, V])                                

lat = ds["latitude"].values
lon = ds["longitude"].values
time = ds["valid_time"].values
Ng = U.shape[0]

np.savez_compressed("data/X_snapshots.npz", X=X, lat=lat, lon=lon, time=time)
print(X.shape, X.dtype)

# Sanity Check
print("NaNs:", np.isnan(X).sum())
print("u range:", U.min(), U.max())
print("v range:", V.min(), V.max())
print("time step (h):", np.unique(np.diff(time)).astype("timedelta64[h]"))