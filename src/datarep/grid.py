
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

ds = xr.open_dataset("data/weatherdata.nc", engine="netcdf4")

south, north = 0, 18
west, east = 78, 108

if (float(ds.latitude.min()) > south or
    float(ds.latitude.max()) < north or
    float(ds.longitude.min()) > west or
    float(ds.longitude.max()) < east):
    ds.close()
    raise ValueError(
        "Dataset does not cover 0-18 N, 78-108 E. "
        "Download ERA5 again using these coordinates."
    )

u = ds["u"].isel(valid_time=0, pressure_level=0)
v = ds["v"].isel(valid_time=0, pressure_level=0)

speed = np.sqrt(u**2 + v**2)
chennai = (80.170, 12.994)
singapore = (103.991, 1.364)

fig, ax = plt.subplots(figsize=(12, 7))
mesh = ax.pcolormesh(
    ds.longitude, ds.latitude, speed,
    shading="auto", cmap="turbo"
)
fig.colorbar(mesh, ax=ax, label="Wind speed magnitude (m/s)")

ax.set_xticks(np.arange(west, east + 0.25, 0.25), minor=True)
ax.set_yticks(np.arange(south, north + 0.25, 0.25), minor=True)
ax.grid(which="minor", color="black", alpha=0.18, linewidth=0.25)

ax.set_xticks(np.arange(west, east + 1, 2))
ax.set_yticks(np.arange(south, north + 1, 2))
ax.grid(which="major", color="black", alpha=0.45, linewidth=0.6)
ax.scatter(*chennai, color="white", edgecolor="black",
           s=100, marker="o", zorder=5)
ax.annotate("A: Chennai Airport", chennai,
            xytext=(7, 8), textcoords="offset points",
            color="white", fontsize=9, weight="bold")

ax.scatter(*singapore, color="white", edgecolor="black",
           s=100, marker="o", zorder=5)
ax.annotate("B: Singapore Changi", singapore,
            xytext=(-125, 10), textcoords="offset points",
            color="white", fontsize=9, weight="bold")

ax.plot(
    [chennai[0], singapore[0]],
    [chennai[1], singapore[1]],
    "w--", linewidth=1.5, label="Airport-to-airport line"
)


# Illustrative curved route
t = np.linspace(0, 1, 100)

route_lon = chennai[0] + t * (singapore[0] - chennai[0])
route_lat = (
    chennai[1]
    + t * (singapore[1] - chennai[1])
    + 1.5 * np.sin(np.pi * t)
)

ax.plot(
    route_lon,
    route_lat,
    color="black",
    linewidth=3,
    label="Illustrative route",
    zorder=6
)

ax.legend(loc="upper right")

timestamp = str(ds.valid_time.values[0])[:16]
ax.set_title(f"ERA5 Wind-Speed Magnitude at 250 hPa\n{timestamp}")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_xlim(west, east)
ax.set_ylim(south, north)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("chennai_singapore_wind_grid.png", dpi=300)
plt.show()

ds.close()
