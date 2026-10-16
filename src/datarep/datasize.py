
import xarray as xr

file_path = "data/weatherdata.nc"

with xr.open_dataset(file_path, engine="netcdf4") as ds:
    # Select the single pressure level
    wind = ds[["u", "v"]].isel(pressure_level=0)

    # Flatten each spatial wind field and combine u and v
    u = wind["u"].stack(space=("latitude", "longitude"))
    v = wind["v"].stack(space=("latitude", "longitude"))

    X_shape = (
        u.sizes["space"] + v.sizes["space"],
        ds.sizes["valid_time"],
    )

    print("=== ROM Snapshot Matrix ===")
    print(f"Matrix shape: {X_shape}")
    print(f"State variables: {X_shape[0]:,}")
    print(f"Snapshots: {X_shape[1]:,}")
    print(f"Total values: {X_shape[0] * X_shape[1]:,}")
    print(f"Float32 memory: "
          f"{X_shape[0] * X_shape[1] * 4 / (1024**2):.2f} MiB")
