# src/main/pod.py -- Phase 1: POD basis from TRAINING data only
# Figures -> results/pod/*.pdf + *.png | Basis -> data/pod_basis.npz
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SRC  = Path(__file__).resolve().parent.parent      
ROOT = SRC.parent                                  
sys.path.append(str(SRC))
from style import W1, W2, C, save                  

DATA = ROOT / "data"
OUT  = ROOT / "results" / "pod"
OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(exist_ok=True)
N_TR, N_VAL, R_MAX = 984, 216, 200

# load + chronological split
d = np.load(DATA / "X_snapshots.npz")
X, lat, lon = d["X"].astype(np.float64), d["lat"], d["lon"]
nlat, nlon = len(lat), len(lon); Ng = nlat * nlon
assert X.shape[0] == 2 * Ng
splits = {"train": X[:, :N_TR], "val": X[:, N_TR:N_TR + N_VAL], "test": X[:, N_TR + N_VAL:]}

# POD (mean + SVD from train only) 
x_mean = splits["train"].mean(axis=1, keepdims=True)
Uf, S, _ = np.linalg.svd(splits["train"] - x_mean, full_matrices=False)
energy = np.cumsum(S**2) / np.sum(S**2)
r_cand = {p: int(np.searchsorted(energy, p) + 1) for p in (0.90, 0.95, 0.99)}

# projection error vs r 
def proj_err(Xs, r):
    Ur = Uf[:, :r]
    Xr = x_mean + Ur @ (Ur.T @ (Xs - x_mean))
    rel = lambda s: (np.linalg.norm(Xr[s] - Xs[s], axis=0) / np.linalg.norm(Xs[s], axis=0)).mean()
    return rel(slice(0, Ng)), rel(slice(Ng, None))      # eps_u, eps_v

r_list = np.unique(np.r_[np.arange(1, 21), np.arange(25, R_MAX + 1, 5), list(r_cand.values())])
err = {k: np.array([proj_err(Xs, r) for r in r_list]) for k, Xs in splits.items()}

# save numbers 
np.savetxt(OUT / "projection_error.csv", np.column_stack([r_list] + [err[k] for k in splits]),
           delimiter=",", fmt="%.6g", comments="",
           header="r," + ",".join(f"{k}_eps_u,{k}_eps_v" for k in splits))

A_all = Uf[:, :R_MAX].T @ (X - x_mean)                   # (R_MAX, 1416)
np.savez(DATA / "pod_basis.npz", Ur=Uf[:, :R_MAX], S=S, x_mean=x_mean, A_all=A_all,
         n_tr=N_TR, n_val=N_VAL, Ng=Ng, lat=lat, lon=lon)

summary = [f"r for {int(p*100)}% energy: {r}" for p, r in r_cand.items()]
summary += ["", "   r | val eps_u eps_v | test eps_u eps_v"]
for r in r_cand.values():
    i = np.where(r_list == r)[0][0]
    summary.append(f"{r:4d} | {err['val'][i,0]:.4f} {err['val'][i,1]:.4f} | "
                   f"{err['test'][i,0]:.4f} {err['test'][i,1]:.4f}")
(OUT / "summary.txt").write_text("\n".join(summary)); print("\n".join(summary))

fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
ax[0].semilogy(np.arange(1, len(S) + 1), S, color=C["black"], marker=".", ms=2, lw=0.6)
ax[0].set(xlabel="Mode index $i$", ylabel=r"Singular value $\sigma_i$")
ax[1].plot(np.arange(1, len(S) + 1), energy, color=C["blue"])
for p, r in r_cand.items():
    ax[1].axhline(p, c="gray", ls="--", lw=0.6)
    ax[1].axvline(r, c=C["orange"], ls=":", lw=0.8)
    ax[1].text(r + 4, p - 0.07, f"$r={r}$", color=C["orange"], fontsize=7)
ax[1].set(xlabel="Number of modes $r$", ylabel="Cumulative energy $E(r)$", xlim=(0, 300))
for a, s in zip(ax, "ab"):
    a.text(0.03, 0.92, f"({s})", transform=a.transAxes, fontsize=8, fontweight="bold")
save(fig, OUT, "spectrum")

lon2d, lat2d = np.meshgrid(lon, lat)
fig, axs = plt.subplots(2, 4, figsize=(W2, 3.3), sharex=True, sharey=True)
for k in range(4):
    for row, (sl, lbl) in enumerate(((slice(0, Ng), "u"), (slice(Ng, None), "v"))):
        m = Uf[sl, k].reshape(nlat, nlon); v = np.abs(m).max()
        im = axs[row, k].pcolormesh(lon2d, lat2d, m, cmap="RdBu_r", vmin=-v, vmax=v,
                                    shading="auto", rasterized=True)
        axs[row, k].grid(False)
        axs[row, k].text(0.04, 0.90, f"${lbl}$, mode {k+1} ({100*S[k]**2/np.sum(S**2):.1f}%)",
                         transform=axs[row, k].transAxes, fontsize=6,
                         bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
        fig.colorbar(im, ax=axs[row, k], fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
for a in axs[1]: a.set_xlabel("Longitude (°E)")
for a in axs[:, 0]: a.set_ylabel("Latitude (°N)")
save(fig, OUT, "modes")

fig, axs = plt.subplots(4, 1, figsize=(W1, 4.2), sharex=True)
for k, a in enumerate(axs):
    a.plot(A_all[k], color=C["blue"], lw=0.6); a.set_ylabel(f"$a_{k+1}$")
    a.axvline(N_TR, c=C["orange"], ls="--", lw=0.8)
    a.axvline(N_TR + N_VAL, c=C["green"], ls="-.", lw=0.8)
axs[-1].set_xlabel("Time (h)")
top = axs[0].get_ylim()[1]
for x, t in ((N_TR / 2, "train"), (N_TR + N_VAL / 2, "val"), (N_TR + 1.5 * N_VAL, "test")):
    axs[0].text(x, top, t, ha="center", va="bottom", fontsize=6)
save(fig, OUT, "coeffs")

fig, ax = plt.subplots(figsize=(W1, 2.5))
for (k, e), ls, mk, c in zip(err.items(), ("--", "-", "-"), (None, "o", "s"),
                             (C["black"], C["blue"], C["orange"])):
    ax.semilogy(r_list, e.mean(1), ls=ls, marker=mk, color=c, markevery=3, label=k)
for r in r_cand.values(): ax.axvline(r, c="gray", ls=":", lw=0.6)
ax.set(xlabel="Number of modes $r$", ylabel=r"Mean $(\varepsilon_u+\varepsilon_v)/2$")
ax.legend()
save(fig, OUT, "projection_error")