import sys
import numpy as np, xarray as xr, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
SRC  = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.append(str(SRC))
from style import W1, W2, C, save

OUT = Path("results/datarepresentation"); OUT.mkdir(parents=True, exist_ok=True)

# ---- load only u, v ----
ds = xr.open_dataset("data/jan2022_2026.nc")
tn  = "valid_time" if "valid_time" in ds.dims else "time"
lat = "latitude"   if "latitude"  in ds.dims else "lat"
lon = "longitude"  if "longitude" in ds.dims else "lon"
ds  = ds[["u", "v"]].squeeze(drop=True).sortby(tn)     # drops singleton level dim

t = ds[tn].values
u = ds["u"].transpose(tn, lat, lon).values.astype(np.float32)
v = ds["v"].transpose(tn, lat, lon).values.astype(np.float32)
m, Nphi, Nlam = u.shape
Ng, n = Nphi * Nlam, 2 * Nphi * Nlam

# ---- snapshot matrix X in R^{n x m}: [u; v] stacked, columns = time ----
X = np.concatenate([u.reshape(m, -1), v.reshape(m, -1)], axis=1).T
years = t.astype("datetime64[Y]").astype(int) + 1970
train, test = years < years.max(), years == years.max()

info = (f"File          : jan2022_2026.nc\n"
        f"Variables     : u, v\n"
        f"Grid          : {Nphi} x {Nlam} (lat x lon), Ng = {Ng}\n"
        f"State dim n   : {n}\n"
        f"Snapshots m   : {m}  ({t[0]} -> {t[-1]})\n"
        f"X shape       : {X.shape}, {X.dtype}, {X.nbytes/2**20:.1f} MiB\n"
        f"Train years   : {sorted(set(years[train]))} -> {train.sum()} snapshots\n"
        f"Test year     : {years.max()} -> {test.sum()} snapshots\n"
        f"Lat range     : {float(ds[lat].min())}-{float(ds[lat].max())}\n"
        f"Lon range     : {float(ds[lon].min())}-{float(ds[lon].max())}\n")
print(info); (OUT / "data_info.txt").write_text(info)

# ---- video: 5 subplots (one per January), wind speed vs time ----
spd = np.sqrt(u**2 + v**2)
yrs = sorted(set(years))
S = [spd[years == y] for y in yrs]
nt = min(len(s) for s in S)
vmax = np.percentile(spd, 99.5)
ext = [float(ds[lon].min()), float(ds[lon].max()), float(ds[lat].min()), float(ds[lat].max())]

fig, axs = plt.subplots(1, 5, figsize=(W2, 1.9), sharey=True)
ims = []
for ax, s, y in zip(axs, S, yrs):
    ims.append(ax.imshow(s[0], origin="lower" if float(ds[lat][0]) < float(ds[lat][-1]) else "upper",
                         extent=ext, vmin=0, vmax=vmax, cmap="viridis", aspect="auto"))
    ax.set_title(str(y)); ax.set_xlabel("Lon [°E]"); ax.grid(False)
axs[0].set_ylabel("Lat [°N]")
fig.colorbar(ims[0], ax=axs, shrink=0.9, label="Wind speed [m/s]")
txt = fig.suptitle("", fontsize=8)

def upd(k):
    for im, s in zip(ims, S): im.set_data(s[k])
    txt.set_text(f"250 hPa, Jan day {k//24 + 1}, {k%24:02d}:00 UTC")
    return ims

ani = FuncAnimation(fig, upd, frames=range(0, nt, 3), blit=False)
try:
    ani.save(OUT / "wind_5years.mp4", writer=FFMpegWriter(fps=15, bitrate=2000))
except Exception:
    ani.save(OUT / "wind_5years.gif", writer=PillowWriter(fps=15))
plt.close(fig)
print("saved to", OUT)