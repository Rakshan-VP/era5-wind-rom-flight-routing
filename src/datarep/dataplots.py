
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

ds = xr.open_dataset("data/weatherdata.nc", engine="netcdf4")

u = ds["u"].isel(pressure_level=0)
v = ds["v"].isel(pressure_level=0)

speed = np.sqrt(u**2 + v**2)

total_frames = ds.sizes["valid_time"]

vmin = 0
vmax = float(speed.max())

lon = ds["longitude"].values
lat = ds["latitude"].values

fig, ax = plt.subplots(figsize=(10, 6))

mesh = ax.pcolormesh(
    lon,
    lat,
    speed.isel(valid_time=0).values,
    shading="auto",
    cmap="turbo",
    vmin=vmin,
    vmax=vmax
)

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Wind speed magnitude (m/s)")

ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
title = ax.set_title("")

def update(frame):
    current_speed = speed.isel(valid_time=frame).values
    mesh.set_array(current_speed.ravel())

    timestamp = str(ds["valid_time"].values[frame])[:16]
    title.set_text(
        f"ERA5 Wind Speed Magnitude at 250 hPa\n"
        f"{timestamp} | Frame {frame + 1}/{total_frames}"
    )

    return mesh, title

animation = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=100,
    blit=False
)

# Save all hourly snapshots to a video
animation.save(
    "results/wind_speed_all_hours.mp4",
    writer=FFMpegWriter(fps=15, bitrate=2500)
)

plt.close(fig)
ds.close()

print(f"Video saved: wind_speed_all_hours.mp4")
print(f"Total frames: {total_frames}")
