import sys
import numpy as np, xarray as xr, matplotlib.pyplot as plt
from pathlib import Path
SRC  = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.append(str(SRC))
from style import W1, W2, C, save

OUT = Path("results/datarepresentation"); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data"); DATA.mkdir(exist_ok=True)

# ---- load u, v ----
ds  = xr.open_dataset("data/jan2022_2026.nc")
tn  = "valid_time" if "valid_time" in ds.dims else "time"
lat = "latitude"   if "latitude"  in ds.dims else "lat"
lon = "longitude"  if "longitude" in ds.dims else "lon"
ds  = ds[["u", "v"]].squeeze(drop=True).sortby(tn)

t = ds[tn].values
u = ds["u"].transpose(tn, lat, lon).values.astype(np.float32)
v = ds["v"].transpose(tn, lat, lon).values.astype(np.float32)
m = u.shape[0]

# ---- X = [u1..uNg ; v1..vNg] x time  (n x m) ----
X = np.vstack([u.reshape(m, -1).T, v.reshape(m, -1).T])
n = X.shape[0]; Ng = n // 2

# ---- split: 2022-25 train, 2026 test ----
years = t.astype("datetime64[Y]").astype(int) + 1970
tr, te = years < years.max(), years == years.max()
Xtr, Xte = X[:, tr], X[:, te]

np.savez(DATA / "snapshot_matrix.npz", X=X, Xtr=Xtr, Xte=Xte, t=t, years=years,
         lat=ds[lat].values, lon=ds[lon].values)

print(f"X    : {X.shape}  ({X.nbytes/2**20:.1f} MiB)  rows 1..{Ng}=u, {Ng+1}..{n}=v, cols=time")
print(f"Xtr  : {Xtr.shape}   Xte : {Xte.shape}")

# ---- quick look at X ----
fig, ax = plt.subplots(figsize=(W2, 2.2))
im = ax.imshow(X, aspect="auto", cmap="RdBu_r", vmin=-40, vmax=40, interpolation="none")
ax.axhline(Ng, c="k", lw=0.6)
for b in np.where(np.diff(years) != 0)[0] + 1: ax.axvline(b, c="k", lw=0.6, ls="--")
ax.set_xlabel("Snapshot index (dashed = year boundary)")
ax.set_ylabel("State index ($u_1..u_{N_g}, v_1..v_{N_g}$)"); ax.grid(False)
fig.colorbar(im, ax=ax, label="Wind [m/s]")
save(fig, OUT, "snapshot_matrix")