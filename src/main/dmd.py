# src/main/dmd.py -- Phase 2: POD-DMD (linear reduced model in POD coefficients)
# Needs: data/pod_basis.npz (pod.py), data/X_snapshots.npz
# Out:   results/dmd/*.pdf|png|csv|txt, data/dmd_models.npz
import sys, time, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SRC  = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.append(str(SRC))
from style import W1, W2, C, save

DATA = ROOT / "data"
OUT  = ROOT / "results" / "dmd"; OUT.mkdir(parents=True, exist_ok=True)

R_LIST = [5, 10, 15, 25, 35, 51, 100, 155]     # all r tested
R_MAIN = [10, 25, 51]                          # r drawn in lead-time plots
LAMS   = [0, 1e-4, 1e-3, 1e-2, 1e-1]           # ridge strengths (0 = plain pseudoinverse)
H, STRIDE = 48, 6                              # short-horizon test: 48 h, restart every 6 h
STABILIZE = True                               # rescale |lambda|>1 onto the unit circle

# ---- load basis + truth -----------------------------------------------------
b = np.load(DATA / "pod_basis.npz")
Ur_all, x_mean, A_all = b["Ur"], b["x_mean"], b["A_all"]
n_tr, n_val, Ng = int(b["n_tr"]), int(b["n_val"]), int(b["Ng"])
X = np.load(DATA / "X_snapshots.npz")["X"].astype(np.float64)
win = {"val": (n_tr, n_tr + n_val), "test": (n_tr + n_val, X.shape[1])}

# ---- helpers ----------------------------------------------------------------
def rel_err(Xh, Xt):
    f = lambda s: np.linalg.norm(Xh[s] - Xt[s], axis=0) / np.linalg.norm(Xt[s], axis=0)
    return f(slice(0, Ng)), f(slice(Ng, None))                 # eps_u(k), eps_v(k)

def fit_dmd(a, lam):
    A1, A2 = a[:, :-1], a[:, 1:]
    if lam == 0:
        return A2 @ np.linalg.pinv(A1, rcond=1e-10)
    G = A1 @ A1.T
    return A2 @ A1.T @ np.linalg.inv(G + lam * np.trace(G) / len(G) * np.eye(len(G)))

def stabilize(A):
    w, V = np.linalg.eig(A)
    m = np.abs(w) > 1
    if not (STABILIZE and m.any()): return A
    w[m] /= np.abs(w[m])
    return np.real((V * w) @ np.linalg.inv(V))

def rollout(A, a0, K):
    a = np.empty((len(a0), K)); a[:, 0] = a0
    for k in range(1, K): a[:, k] = A @ a[:, k - 1]
    return a

def short_curve(pred, s, e):
    """Mean (eps_u+eps_v)/2 vs lead time; restart from ERA5 every STRIDE h."""
    out = []
    for k0 in range(s, e - H, STRIDE):
        eu, ev = rel_err(pred(k0), X[:, k0:k0 + H + 1]); out.append((eu + ev) / 2)
    return np.mean(out, axis=0)

dmd_pred = lambda A, r: (lambda k0: x_mean + Ur_all[:, :r] @ rollout(A, A_all[:r, k0], H + 1))
persist  = lambda k0: np.repeat(X[:, [k0]], H + 1, axis=1)     # hold the starting field
clim     = lambda k0: np.repeat(x_mean, H + 1, axis=1)         # training mean

# ---- baselines (independent of r) -------------------------------------------
base_short = {(n, k): short_curve(f, *win[n]) for n in win for k, f in (("persist", persist), ("clim", clim))}
base_free = {}
for n, (s, e) in win.items():
    for k, Xh in (("persist", np.repeat(X[:, [s]], e - s, axis=1)), ("clim", np.repeat(x_mean, e - s, axis=1))):
        eu, ev = rel_err(Xh, X[:, s:e]); base_free[(n, k)] = (eu + ev) / 2

# ---- fit, select ridge on validation (short-horizon), evaluate --------------
rows, free, short, models, lams = [], {}, {}, {}, {}
with np.errstate(all="ignore"):
    for r in R_LIST:
        Ur, a_tr = Ur_all[:, :r], A_all[:r, :n_tr]
        best = None
        for lam in LAMS:                                       # choose lam on validation only
            t0 = time.perf_counter(); A = stabilize(fit_dmd(a_tr, lam)); tf = time.perf_counter() - t0
            score = short_curve(dmd_pred(A, r), *win["val"]).mean()
            if best is None or score < best[0]: best = (score, lam, A, tf)
        _, lam_b, A, t_fit = best
        raw = np.linalg.eigvals(fit_dmd(a_tr, lam_b)); lams[r], models[r] = raw, A
        row = dict(r=r, lam=lam_b, max_abs_lambda=np.abs(raw).max(), n_unstable=int((np.abs(raw) > 1).sum()), fit_s=t_fit)
        for n, (s, e) in win.items():
            t0 = time.perf_counter(); Xh = x_mean + Ur @ rollout(A, A_all[:r, s], e - s); tp = time.perf_counter() - t0
            eu, ev = rel_err(Xh, X[:, s:e]); free[(n, r)] = (eu + ev) / 2
            pu, pv = rel_err(x_mean + Ur @ A_all[:r, s:e], X[:, s:e])
            short[(n, r)] = short_curve(dmd_pred(A, r), s, e)
            row.update({f"{n}_free_u": eu.mean(), f"{n}_free_v": ev.mean(),
                        f"{n}_floor": (pu.mean() + pv.mean()) / 2, f"{n}_short48": short[(n, r)].mean()})
        row["pred_s"] = tp
        rows.append(row)

# ---- save numbers -----------------------------------------------------------
with open(OUT / "dmd_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
    for q in rows: w.writerow({k: f"{v:.6g}" for k, v in q.items()})
np.savez(DATA / "dmd_models.npz", **{f"A_{r}": A for r, A in models.items()})

mu = np.log(lams[25].astype(complex))
with np.errstate(divide="ignore"):
    per = np.where(np.abs(mu.imag) > 1e-9, 2 * np.pi / np.abs(mu.imag), np.inf)
np.savetxt(OUT / "dmd_eigs_r25.csv", np.c_[np.abs(lams[25]), mu.real, per], delimiter=",",
           fmt="%.6g", header="abs_lambda,growth_per_h,period_h", comments="")

L = ["Free run over full window (mean of eps_u, eps_v) and 48 h restarted error (mean over lead time)",
     "  r |  lam  | max|lam| #unst | val free u v | test free u v | val48 test48 | test floor"]
for q in rows:
    L.append(f"{q['r']:4d} | {q['lam']:.0e} | {q['max_abs_lambda']:.4f} {q['n_unstable']:3d} | "
             f"{q['val_free_u']:.3f} {q['val_free_v']:.3f} | {q['test_free_u']:.3f} {q['test_free_v']:.3f} | "
             f"{q['val_short48']:.3f} {q['test_short48']:.3f} | {q['test_floor']:.3f}")
L += ["", "Baselines (mean of eps_u, eps_v):"]
for k in ("persist", "clim"):
    L.append(f"{k:8s} 48h val {base_short[('val', k)].mean():.3f} test {base_short[('test', k)].mean():.3f} | "
             f"free val {base_free[('val', k)].mean():.3f} test {base_free[('test', k)].mean():.3f}")
(OUT / "summary.txt").write_text("\n".join(L)); print("\n".join(L))

# ---- Fig 1: eigenvalues (r = 25, unstabilized fit) --------------------------
fig, ax = plt.subplots(figsize=(W1, 3.0))
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), color="gray", lw=0.6, ls="--")
ax.plot(lams[25].real, lams[25].imag, "o", color=C["blue"], ms=3)
ax.set(xlabel=r"Re($\lambda$)", ylabel=r"Im($\lambda$)", aspect="equal")
save(fig, OUT, "eigenvalues")

# ---- Fig 2: 48 h error vs r with baselines and POD floor (test) -------------
R = [q["r"] for q in rows]
fig, ax = plt.subplots(figsize=(W1, 2.5))
ax.plot(R, [q["test_short48"] for q in rows], "-o", color=C["blue"], label="POD-DMD")
ax.plot(R, [q["test_floor"] for q in rows], "--", color=C["black"], label="POD floor")
ax.axhline(base_short[("test", "persist")].mean(), color=C["orange"], ls="-.", label="Persistence")
ax.axhline(base_short[("test", "clim")].mean(), color=C["green"], ls=":", label="Climatology")
ax.set(xlabel="Number of modes $r$", ylabel=r"48 h mean $(\varepsilon_u+\varepsilon_v)/2$")
ax.legend(fontsize=6)
save(fig, OUT, "error_vs_r")

# ---- Fig 3: error vs lead time (test): (a) 48 h restarted, (b) full free run
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
for r, col, ls in zip(R_MAIN, (C["blue"], C["orange"], C["green"]), ("-", "--", ":")):
    ax[0].plot(short[("test", r)], color=col, ls=ls, label=f"DMD $r={r}$")
    ax[1].plot(free[("test", r)], color=col, ls=ls, label=f"DMD $r={r}$")
ax[0].plot(base_short[("test", "persist")], color="k", ls="-.", label="Persistence")
ax[0].plot(base_short[("test", "clim")], color="gray", ls=":", label="Climatology")
ax[1].plot(base_free[("test", "persist")], color="k", ls="-.", label="Persistence")
ax[1].plot(base_free[("test", "clim")], color="gray", ls=":", label="Climatology")
ax[0].set(xlabel="Lead time (h)", ylabel=r"$(\varepsilon_u+\varepsilon_v)/2$", xlim=(0, H))
ax[1].set(xlabel="Lead time (h)", ylabel=r"$(\varepsilon_u+\varepsilon_v)/2$", ylim=(0, 2))
ax[0].legend(fontsize=6); ax[1].legend(fontsize=6)
for a, s in zip(ax, "ab"):
    a.text(0.03, 0.90, f"({s})", transform=a.transAxes, fontsize=8, fontweight="bold")
save(fig, OUT, "error_vs_lead")