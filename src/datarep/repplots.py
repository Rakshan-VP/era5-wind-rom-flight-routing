
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Settings
DATA_FILE = "data/weatherdata.nc"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
ds = xr.open_dataset(DATA_FILE, engine="netcdf4")

try:
    # Select 250 hPa
    level_index = int(
        np.argmin(np.abs(ds["pressure_level"].values - 250.0))
    )

    u = ds["u"].isel(pressure_level=level_index)
    v = ds["v"].isel(pressure_level=level_index)

    # Crop to Chennai-Singapore region
    lat = ds["latitude"].values
    lat_slice = (
        slice(18, 0) if lat[0] > lat[-1]
        else slice(0, 18)
    )

    u = u.sel(
        latitude=lat_slice,
        longitude=slice(78, 108)
    )
    v = v.sel(
        latitude=lat_slice,
        longitude=slice(78, 108)
    )

    lon = u["longitude"].values
    lat = u["latitude"].values
    times = ds["valid_time"].values
    samples = np.arange(1, len(times) + 1)

    # =====================================================
    # Plot 1: Spatial wind-speed magnitude at snapshot 1
    # =====================================================
    speed = np.hypot(u.isel(valid_time=0), v.isel(valid_time=0))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    mesh = ax.pcolormesh(
        lon, lat, speed.values,
        shading="auto",
        cmap="Purples"
    )
    fig.colorbar(mesh, ax=ax, label="Wind speed magnitude (m/s)")

    # 0.25-degree grid
    ax.set_xticks(np.arange(78, 108.01, 0.25), minor=True)
    ax.set_yticks(np.arange(0, 18.01, 0.25), minor=True)
    ax.grid(which="minor", color="black", alpha=0.2, linewidth=0.25)

    # Chennai and Singapore airports (longitude, latitude)
    chennai = (80.170, 12.994)
    singapore = (103.991, 1.364)

    ax.scatter(*chennai, color="black", s=50, zorder=5)
    ax.scatter(*singapore, color="black", s=50, zorder=5)

    ax.annotate("A: Chennai Airport", chennai,
                xytext=(5, 6), textcoords="offset points", fontsize=8)
    ax.annotate("B: Singapore Changi", singapore,
                xytext=(-100, 8), textcoords="offset points", fontsize=8)

    ax.plot(
        [chennai[0], singapore[0]],
        [chennai[1], singapore[1]],
        "k--", linewidth=1.2, label="Direct reference line"
    )

    ax.set_title(
        "Spatial wind-speed magnitude at 250 hPa\n"
        + str(times[0])[:16]
    )
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fom_plot1.png", dpi=300)
    plt.close(fig)

    # =====================================================
    # Plot 2: Temporal variation of u at grid point 1
    # =====================================================
    u1 = u.isel(latitude=0, longitude=0).values

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(samples, u1, color="black", linewidth=0.8)
    ax.set_title("Temporal variation of zonal wind $u_1$")
    ax.set_xlabel("Time sample (hourly index)")
    ax.set_ylabel("Zonal wind u (m/s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fom_plot2.png", dpi=300)
    plt.close(fig)

    # =====================================================
    # Plot 3: Temporal variation of v at grid point 1
    # =====================================================
    v1 = v.isel(latitude=0, longitude=0).values

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(samples, v1, color="purple", linewidth=0.8)
    ax.set_title("Temporal variation of meridional wind $v_1$")
    ax.set_xlabel("Time sample (hourly index)")
    ax.set_ylabel("Meridional wind v (m/s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fom_plot3.png", dpi=300)
    plt.close(fig)

    print("Three plots saved in:", RESULTS_DIR.resolve())
    print("1. fom_plot1.png — spatial wind-speed magnitude")
    print("2. fom_plot2.png — temporal u at grid point 1")
    print("3. fom_plot3.png — temporal v at grid point 1")

finally:
    ds.close()
