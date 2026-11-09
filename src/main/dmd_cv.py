# src/main/dmd_cv.py -- POD-DMD on the blocked-CV bases (one basis, one model per fold)
# Needs: data/pod_cv.npz (pod_cv.py), data/X_snapshots.npz
# Out:   results/dmd_cv/*.pdf|png|csv|txt
import sys, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SRC  = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.append(str(SRC))
from style import W1, W2, C, save

DATA = ROOT / "data"
OUT  = ROOT / "results" / "dmd_cv"; OUT.mkdir(parents=True, exist_ok=True)

R_LIST = [5, 10, 15, 25, 35, 51, 100]
R_PLOT = 25
LAMS   = [0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3]
H, STRIDE = 48, 6
INNER, GAP_IN = 168, 48          # inner validation: last 168 h of training; 48 h buffer before it

# ---- load -------------------------------------------------------------------
cv = np.load(DATA / "pod_cv.npz")
X = np.load(DATA / "X_snapshots.npz")["X"].astype(np.float64)
Ng, K = int(cv["Ng"]), len(cv["edges"]) - 1

# ---- helpers ----------------------------------------------------------------
def pairs(idx):
    """(a_k, a_k+1) index pairs inside contiguous segments only."""
    segs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return np.concatenate([s[:-1] for s in segs]), np.concatenate([s[1:] for s in segs])

def fit_dmd(A1, A2, lam):
    if lam == 0: return A2 @ np.linalg.pinv(A1, rcond=1e-10)
    G = A1 @ A1.T
    return A2 @ A1.T @ np.linalg.inv(G + lam * np.trace(G) / len(G) * np.eye(len(G)))

def stabilize(A):
    w, V = np.linalg.eig(A); m = np.abs(w) > 1
    if not m.any(): return A
    w[m] /= np.abs(w[m]); return np.real((V * w) @ np.linalg.inv(V))

def rollout(A, a0):
    a = np.empty((len(a0), H + 1)); a[:, 0] = a0
    for k in range(1, H + 1): a[:, k] = A @ a[:, k - 1]
    return a

def curve(pred, restarts):
    """Mean over restarts of (eps_u+eps_v)/2 at each lead time."""
    out = []
    for k0 in restarts:
        Xh, Xt = pred(k0), X[:, k0:k0 + H + 1]
        f = lambda s: np.linalg.norm(Xh[s] - Xt[s], axis=0) / np.linalg.norm(Xt[s], axis=0)
        out.append((f(slice(0, Ng)) + f(slice(Ng, None))) / 2)
    return np.mean(out, axis=0)

# ---- run folds --------------------------------------------------------------
res = {}                                   # (r, method) -> list over folds of lead-time curves
lam_chosen = {r: [] for r in R_LIST}
with np.errstate(all="ignore"):
    for f in range(K):
        Uf, mu = cv[f"Ur_{f}"].astype(np.float64), cv[f"mean_{f}"]
        a, tr, te = cv[f"A_{f}"].astype(np.float64), cv[f"train_{f}"], cv[f"test_{f}"]
        test_rs = range(te[0], te[-1] + 1 - H, STRIDE)

        vs = tr[-INNER]; fit_idx = tr[tr < vs - GAP_IN]           # inner split of the training data
        inner_rs = range(vs, tr[-1] + 1 - H, STRIDE)
        i1f, i2f = pairs(fit_idx); i1, i2 = pairs(tr)

        res.setdefault(("all", "persist"), []).append(curve(lambda k0: np.repeat(X[:, [k0]], H + 1, 1), test_rs))
        res.setdefault(("all", "clim"), []).append(curve(lambda k0: np.repeat(mu, H + 1, 1), test_rs))
        for r in R_LIST:
            Ur, ar = Uf[:, :r], a[:r]
            pred = lambda A: (lambda k0: mu + Ur @ rollout(A, ar[:, k0]))
            sc = [curve(pred(stabilize(fit_dmd(ar[:, i1f], ar[:, i2f], l))), inner_rs).mean() for l in LAMS]
            lam = LAMS[int(np.nanargmin(sc))]; lam_chosen[r].append(lam)   # chosen on training data only
            A = stabilize(fit_dmd(ar[:, i1], ar[:, i2], lam))              # refit on all training
            res.setdefault((r, "dmd"), []).append(curve(pred(A), test_rs))
            res.setdefault((r, "persist_r"), []).append(
                curve(lambda k0: mu + Ur @ np.repeat(ar[:, [k0]], H + 1, 1), test_rs))
        print(f"fold {f+1}/{K} done ({len(test_rs)} restarts)")

# ---- numbers ----------------------------------------------------------------
fm = lambda key: np.array([c.mean() for c in res[key]])           # per-fold 48 h mean
fp, fc = fm(("all", "persist")), fm(("all", "clim"))
L = [f"POD-DMD, blocked CV ({K} folds): 48 h restarted error, mean (eps_u+eps_v)/2 over lead time",
     f"Full-space persistence {fp.mean():.3f} +- {fp.std(ddof=1):.3f} | climatology {fc.mean():.3f} +- {fc.std(ddof=1):.3f}", "",
     "  r | DMD mean +- std | persist_r mean +- std | folds DMD<persist_r | diff (DMD - persist_r) | ridge per fold"]
rows = []
for r in R_LIST:
    d, p = fm((r, "dmd")), fm((r, "persist_r"))
    rows.append([r, d.mean(), d.std(ddof=1), p.mean(), p.std(ddof=1), int((d < p).sum()), (d - p).mean()])
    L.append(f"{r:4d} | {d.mean():.3f} +- {d.std(ddof=1):.3f} | {p.mean():.3f} +- {p.std(ddof=1):.3f} | "
             f"{int((d < p).sum())}/{K} | {(d - p).mean():+.4f} | {lam_chosen[r]}")
(OUT / "summary.txt").write_text("\n".join(L)); print("\n".join(L))
with open(OUT / "dmd_cv_results.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["r", "dmd_mean", "dmd_std", "persist_r_mean", "persist_r_std", "folds_better", "diff"])
    w.writerows(rows)
np.savez(DATA / "dmd_cv_curves.npz", **{f"{k[0]}_{k[1]}": np.array(v) for k, v in res.items()})

# ---- Fig 1: error vs r (mean +- std over folds) -----------------------------
R = np.array(R_LIST); rw = np.array(rows)
fig, ax = plt.subplots(figsize=(W1, 2.6))
ax.errorbar(R, rw[:, 1], rw[:, 2], color=C["orange"], marker="^", ls="--", capsize=2, label="POD-DMD")
ax.errorbar(R + 0.8, rw[:, 3], rw[:, 4], color="gray", marker="+", ls="-.", capsize=2, label="Persistence (rank $r$)")
ax.axhline(fp.mean(), color="k", ls="-.", lw=0.6, label="Persistence")
ax.axhline(fc.mean(), color="k", ls=":", lw=0.8, label="Climatology")
ax.set(xlabel="Number of modes $r$", ylabel=r"48 h mean $(\varepsilon_u+\varepsilon_v)/2$")
ax.legend(fontsize=6); save(fig, OUT, "cv_error_vs_r")

# ---- Fig 2: error vs lead time at r = R_PLOT (fold mean, +- std band) -------
fig, ax = plt.subplots(figsize=(W1, 2.6))
for key, col, ls, lab in (((R_PLOT, "dmd"), C["orange"], "--", "POD-DMD"),
                          ((R_PLOT, "persist_r"), "gray", "-.", "Persistence (rank $r$)"),
                          (("all", "persist"), "k", "-.", "Persistence"),
                          (("all", "clim"), "k", ":", "Climatology")):
    cc = np.array(res[key]); m, s = cc.mean(0), cc.std(0, ddof=1)
    ax.plot(m, color=col, ls=ls, label=lab); ax.fill_between(range(H + 1), m - s, m + s, color=col, alpha=0.12, lw=0)
ax.set(xlabel="Lead time (h)", ylabel=r"$(\varepsilon_u+\varepsilon_v)/2$", xlim=(0, H))
ax.legend(fontsize=6); save(fig, OUT, "cv_error_vs_lead")