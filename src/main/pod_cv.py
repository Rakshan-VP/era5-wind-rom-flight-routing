# src/main/pod_cv.py -- Basis 2: blocked time-series CV, one POD basis per fold
# Each fold: hold out one contiguous block; drop GAP hours on both sides from training;
# mean + SVD use the remaining training snapshots only.
# Out: data/pod_cv.npz, results/pod/cv_*.{csv,txt,pdf,png}
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SRC  = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.append(str(SRC))
from style import W1, C, save

DATA = ROOT / "data"
OUT  = ROOT / "results" / "pod_cv"
OUT.mkdir(parents=True, exist_ok=True)
K_FOLDS, GAP, R_CV = 5, 48, 160          # folds, gap (h) each side, modes stored per fold

# ---- load -------------------------------------------------------------------
d = np.load(DATA / "X_snapshots.npz")
X, lat, lon = d["X"].astype(np.float64), d["lat"], d["lon"]
Ng = len(lat) * len(lon); nt = X.shape[1]
assert X.shape[0] == 2 * Ng

def pod(Xtr):
    mu = Xtr.mean(axis=1, keepdims=True)
    U, S, _ = np.linalg.svd(Xtr - mu, full_matrices=False)
    return mu, U, S

def proj_err(Xs, mu, U, r):
    Ur = U[:, :r]
    Xr = mu + Ur @ (Ur.T @ (Xs - mu))
    rel = lambda s: (np.linalg.norm(Xr[s] - Xs[s], axis=0) / np.linalg.norm(Xs[s], axis=0)).mean()
    return rel(slice(0, Ng)), rel(slice(Ng, None))          # eps_u, eps_v

def r_at(S, ps=(0.90, 0.95, 0.99)):
    e = np.cumsum(S**2) / np.sum(S**2)
    return {p: int(np.searchsorted(e, p) + 1) for p in ps}

r_list = np.unique(np.r_[np.arange(1, 21), np.arange(25, 201, 5), 51])

# ---- one basis per fold -----------------------------------------------------
edges = np.linspace(0, nt, K_FOLDS + 1).astype(int)
cv_err = np.zeros((K_FOLDS, len(r_list), 2))
cv_r, store = [], {}
for f in range(K_FOLDS):
    s, e = edges[f], edges[f + 1]
    te = np.arange(s, e)
    tr = np.r_[np.arange(0, max(s - GAP, 0)), np.arange(min(e + GAP, nt), nt)]
    mu, U, S = pod(X[:, tr])
    rc = r_at(S); cv_r.append(rc)
    cv_err[f] = [proj_err(X[:, te], mu, U, r) for r in r_list]
    store.update({f"Ur_{f}": U[:, :R_CV].astype(np.float32), f"mean_{f}": mu,
                  f"S_{f}": S, f"A_{f}": (U[:, :R_CV].T @ (X - mu)).astype(np.float32),
                  f"train_{f}": tr, f"test_{f}": te})
    print(f"fold {f+1}/{K_FOLDS}: test h {s}-{e-1}, train {len(tr)} snaps, "
          f"r90/95/99 = {rc[0.90]}/{rc[0.95]}/{rc[0.99]}")

np.savez_compressed(DATA / "pod_cv.npz", edges=edges, gap=GAP, Ng=Ng, lat=lat, lon=lon, **store)

# ---- numbers ----------------------------------------------------------------
cv_m = cv_err.mean(2)                                        # (K, nr): (eps_u+eps_v)/2
mean, std = cv_m.mean(0), cv_m.std(0, ddof=1)
np.savetxt(OUT / "cv_projection_error.csv",
           np.column_stack([r_list, cv_err[:, :, 0].mean(0), cv_err[:, :, 1].mean(0), mean, std, cv_m.T]),
           delimiter=",", fmt="%.6g", comments="",
           header="r,eps_u_mean,eps_v_mean,eps_mean,eps_std," + ",".join(f"fold{f+1}" for f in range(K_FOLDS)))

L = [f"Blocked CV basis: {K_FOLDS} folds, gap {GAP} h", "fold | test hours   | r90 r95 r99"]
L += [f"{f+1:4d} | {edges[f]:4d}-{edges[f+1]-1:4d}    | {c[0.90]:3d} {c[0.95]:3d} {c[0.99]:3d}"
      for f, c in enumerate(cv_r)]
L += ["", "   r | eps_u  eps_v | (u+v)/2 mean +- std over folds"]
for r in (10, 25, 51, 100, 155):
    i = np.where(r_list == r)[0][0]
    L.append(f"{r:4d} | {cv_err[:, i, 0].mean():.3f}  {cv_err[:, i, 1].mean():.3f} | {mean[i]:.3f} +- {std[i]:.3f}")
(OUT / "cv_summary.txt").write_text("\n".join(L)); print("\n" + "\n".join(L))

# ---- figure: CV projection error vs r (overlay chronological test if available)
fig, ax = plt.subplots(figsize=(W1, 2.6))
for f in range(K_FOLDS):
    ax.semilogy(r_list, cv_m[f], color="gray", lw=0.5, alpha=0.6, label="single fold" if f == 0 else None)
ax.semilogy(r_list, mean, color=C["blue"], lw=1.2, label="CV mean")
ax.fill_between(r_list, mean - std, mean + std, color=C["blue"], alpha=0.2, lw=0, label=r"$\pm1$ std")
csv = OUT / "projection_error.csv"
if csv.exists():
    t = np.loadtxt(csv, delimiter=",", skiprows=1)           # r, train(u,v), val(u,v), test(u,v)
    ax.semilogy(t[:, 0], t[:, 5:7].mean(1), "--", color=C["orange"], label="chronological test")
ax.set(xlabel="Number of modes $r$", ylabel=r"Mean $(\varepsilon_u+\varepsilon_v)/2$")
ax.legend(fontsize=6)
save(fig, OUT, "cv_projection_error")