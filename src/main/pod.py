import sys
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
SRC  = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.append(str(SRC))
from style import W1, W2, C, save

OUT = Path("results/pod"); OUT.mkdir(parents=True, exist_ok=True)

# ---- load + split (train = 2022-25, test = 2026) ----
d = np.load("data/snapshot_matrix.npz")
Xtr, Xte, lat, lon = d["Xtr"].astype(np.float64), d["Xte"].astype(np.float64), d["lat"], d["lon"]
n = Xtr.shape[0]; Ng = n // 2; Nphi, Nlam = len(lat), len(lon)
org = "lower" if lat[0] < lat[-1] else "upper"
ext = [lon.min(), lon.max(), lat.min(), lat.max()]

# ---- POD on training data only ----
xbar = Xtr.mean(axis=1, keepdims=True)
U, S, _ = np.linalg.svd(Xtr - xbar, full_matrices=False)
np.savez("data/pod.npz", U=U, S=S, xbar=xbar)
E = np.cumsum(S**2) / np.sum(S**2)
for p in (0.90, 0.95, 0.99):
    print(f"r for {p:.0%} energy: {np.searchsorted(E, p) + 1}")

# ---- reconstruction error vs r (train and test) ----
rs = np.unique(np.r_[1:10, 10:101:5, 150, 200, 300]); rs = rs[rs <= len(S)]
def err(X, r):
    Xc = X - xbar; Ur = U[:, :r]
    return np.mean(np.linalg.norm(Xc - Ur @ (Ur.T @ Xc), axis=0) / np.linalg.norm(X, axis=0))
etr = [err(Xtr, r) for r in rs]; ete = [err(Xte, r) for r in rs]

# ---- plot 1: singular values + residual energy (zoomed) ----
k = np.arange(1, len(S) + 1)
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.1))
ax[0].loglog(k, S / S[0], c=C["blue"])
ax[0].set(xlabel="Mode $i$", ylabel="$\\sigma_i/\\sigma_1$")
ax[1].semilogy(k[:400], 1 - E[:400], c=C["orange"])
for p, ls in zip((0.90, 0.95, 0.99), (":", "--", "-.")):
    r_p = np.searchsorted(E, p) + 1
    ax[1].axhline(1 - p, c="k", ls=ls, lw=0.5)
    ax[1].plot(r_p, 1 - p, "ko", ms=3)
    ax[1].annotate(f"{p:.0%}: $r$={r_p}", (r_p, 1 - p), xytext=(4, 3),
                   textcoords="offset points", fontsize=6)
ax[1].set(xlabel="$r$", ylabel="Residual energy $1-E(r)$")
save(fig, OUT, "singular_values")

# ---- plot 2: reconstruction error vs r ----
fig, ax = plt.subplots(figsize=(W1, 2.2))
ax.semilogy(rs, etr, "o-", c=C["blue"], label="Train (2022-25)")
ax.semilogy(rs, ete, "s-", c=C["orange"], label="Test (2026)")
ax.set(xlabel="$r$", ylabel="Mean relative error"); ax.legend()
save(fig, OUT, "recon_error_vs_r")

# ---- plot 3: first 6 spatial modes (u-part), single column ----
fig, axs = plt.subplots(3, 2, figsize=(W1, 4.6), sharex=True, sharey=True)
for i, ax in enumerate(axs.flat):
    f = U[:Ng, i].reshape(Nphi, Nlam); a = np.abs(f).max()
    im = ax.imshow(f, origin=org, extent=ext, cmap="RdBu_r", vmin=-a, vmax=a, aspect="auto")
    ax.set_title(f"Mode {i+1} ({S[i]**2/np.sum(S**2):.1%})", fontsize=7); ax.grid(False)
for ax in axs[-1]: ax.set_xlabel("Lon [°E]")
for ax in axs[:, 0]: ax.set_ylabel("Lat [°N]")
fig.colorbar(im, ax=axs, shrink=0.8, label="$u$-component (normalised)")
save(fig, OUT, "pod_modes")

# ---- plot 4: temporal coefficients of the test year, single column ----
A = U[:, :4].T @ (Xte - xbar)
fig, ax = plt.subplots(figsize=(W1, 2.2))
for i in range(4): ax.plot(A[i], label=f"$a_{i+1}$", lw=0.6)
ax.set(xlabel="Test snapshot (hourly)", ylabel="Coefficient"); ax.legend(ncol=2, fontsize=6)
save(fig, OUT, "pod_coeffs_test")

# ---- video: r chosen from 90/95/99% energy; top = speed, bottom = |error| ----
P = [0.90, 0.95, 0.99]
R = [int(np.searchsorted(E, p)) + 1 for p in P]
print("r for 90/95/99% energy:", R)
frames = range(0, min(240, Xte.shape[1]), 2)
spd = lambda x: np.hypot(x[:Ng], x[Ng:]).reshape(Nphi, Nlam)
recon = {r: xbar + U[:, :r] @ (U[:, :r].T @ (Xte - xbar)) for r in R}
sp_all = np.hypot(Xte[:Ng], Xte[Ng:])
vmin, vmax = np.percentile(sp_all, [5, 95])     
emax = np.percentile(np.abs(np.hypot(recon[R[0]][:Ng], recon[R[0]][Ng:])
                            - np.hypot(Xte[:Ng], Xte[Ng:])), 99.5)

fig, axs = plt.subplots(2, 4, figsize=(9, 4.2), sharex=True, sharey=True)
kw = dict(origin=org, extent=ext, aspect="auto")
top = [axs[0, 0].imshow(spd(Xte[:, 0]), vmin=0, vmax=vmax, cmap="viridis", **kw)]
axs[0, 0].set_title("ERA5 (original)")
axs[1, 0].imshow(spd(xbar[:, 0]), vmin=0, vmax=vmax, cmap="viridis", **kw)
axs[1, 0].set_title("Training mean")
bot = []
for j, (p, r) in enumerate(zip(P, R), start=1):
    top.append(axs[0, j].imshow(spd(Xte[:, 0]), vmin=0, vmax=vmax, cmap="viridis", **kw))
    axs[0, j].set_title(f"{p:.0%} energy, $r={r}$")
    bot.append(axs[1, j].imshow(np.zeros((Nphi, Nlam)), vmin=0, vmax=emax, cmap="magma", **kw))
    axs[1, j].set_title(f"Error, $r={r}$")
for ax in axs.flat: ax.grid(False)
for ax in axs[1]: ax.set_xlabel("Lon [°E]")
for ax in axs[:, 0]: ax.set_ylabel("Lat [°N]")
fig.colorbar(top[0], ax=axs[0], shrink=0.9, pad=0.01, label="Speed [m/s]")
fig.colorbar(bot[0], ax=axs[1], shrink=0.9, pad=0.01, label="|Speed error| [m/s]")
txt = fig.suptitle("", fontsize=9)

def upd(k):
    true = spd(Xte[:, k]); top[0].set_data(true)
    for j, r in enumerate(R):
        rec = spd(recon[r][:, k])
        top[j + 1].set_data(rec); bot[j].set_data(np.abs(rec - true))
    txt.set_text(f"Test year, hour {k}")
    return top + bot

ani = FuncAnimation(fig, upd, frames=frames, blit=False)
try: ani.save(OUT / "pod_reconstruction.mp4", writer=FFMpegWriter(fps=12, bitrate=3000))
except Exception: ani.save(OUT / "pod_reconstruction.gif", writer=PillowWriter(fps=12))
plt.close(fig)
print("saved to", OUT)