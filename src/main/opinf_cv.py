# src/main/opinf_cv.py -- POD-OpInf (linear, quadratic) on the blocked-CV bases
# Needs: data/pod_cv.npz (pod_cv.py), data/X_snapshots.npz, (optional) data/dmd_cv_curves.npz
# Out:   results/opinf_cv/*.pdf|png|csv|txt, data/opinf_cv_curves.npz
import sys, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SRC  = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.append(str(SRC))
from style import W1, W2, C, save

DATA = ROOT / "data"
OUT  = ROOT / "results" / "opinf_cv"; OUT.mkdir(parents=True, exist_ok=True)

R_LIST = [5, 10, 25, 51]
R_PLOT = 25
G_LIN  = [1e-3, 1e-2, 1e-1, 1]            # ridge on (c, A)
G_QUAD = [1e-1, 1, 10, 100]               # ridge on H
H, STRIDE, SEL_STRIDE = 48, 6, 12
INNER, GAP_IN = 168, 48                   # inner validation: last 168 h of training, 48 h buffer
BLOW = 3.0                                # divergence bound = BLOW x max training norm

# ---- load -------------------------------------------------------------------
cv = np.load(DATA / "pod_cv.npz")
X = np.load(DATA / "X_snapshots.npz")["X"].astype(np.float64)
Ng, K = int(cv["Ng"]), len(cv["edges"]) - 1
dmd_f = DATA / "dmd_cv_curves.npz"
DMDC = np.load(dmd_f) if dmd_f.exists() else None

# ---- helpers ----------------------------------------------------------------
def pairs(idx):
    """(a_k, a_k+1) index pairs inside contiguous segments only."""
    segs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return np.concatenate([s[:-1] for s in segs]), np.concatenate([s[1:] for s in segs])

def feats(a, iu, q):
    p = [np.ones((1, a.shape[1])), a]
    if q: p.append(a[iu[0]] * a[iu[1]])
    return np.vstack(p)

def fit(A1, A2, g1, g2, q):
    r = A1.shape[0]; iu = np.triu_indices(r)
    D = feats(A1, iu, q)
    reg = np.r_[0.0, np.full(r, g1), np.full(len(iu[0]) if q else 0, g2)] * D.shape[1]
    return np.linalg.solve(D @ D.T + np.diag(reg), D @ A2.T).T

def rollout(O, q, iu, a0, bound):
    a = np.empty((len(a0), H + 1)); a[:, 0] = a0
    for k in range(1, H + 1):
        x = a[:, k - 1]
        y = O @ (np.r_[1.0, x, x[iu[0]] * x[iu[1]]] if q else np.r_[1.0, x])
        n = np.linalg.norm(y)
        if not np.isfinite(n): y = x
        elif n > bound: y = y * bound / n
        a[:, k] = y
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
res, chosen = {}, {r: [] for r in R_LIST}          # chosen: (g_lin, g_quad_best, quad_kept) per fold
with np.errstate(all="ignore"):
    for f in range(K):
        Uf, mu = cv[f"Ur_{f}"].astype(np.float64), cv[f"mean_{f}"]
        a, tr, te = cv[f"A_{f}"].astype(np.float64), cv[f"train_{f}"], cv[f"test_{f}"]
        test_rs = range(te[0], te[-1] + 1 - H, STRIDE)
        vs = tr[-INNER]; fit_idx = tr[tr < vs - GAP_IN]
        inner_rs = range(vs, tr[-1] + 1 - H, SEL_STRIDE)
        i1f, i2f = pairs(fit_idx); i1, i2 = pairs(tr)

        for r in R_LIST:
            Ur, ar = Uf[:, :r], a[:r]
            SC = ar[0, tr].std(); iu = np.triu_indices(r)
            bound = BLOW * np.linalg.norm(ar[:, tr] / SC, axis=0).max()
            pred = lambda O, q: (lambda k0: mu + Ur @ (SC * rollout(O, q, iu, ar[:, k0] / SC, bound)))

            best = {}
            for q in (False, True):                                    # tune on inner validation only
                for g1 in G_LIN:
                    for g2 in (G_QUAD if q else [0]):
                        O = fit(ar[:, i1f] / SC, ar[:, i2f] / SC, g1, g2, q)
                        s = curve(pred(O, q), inner_rs).mean()
                        s = np.inf if not np.isfinite(s) else s
                        if q not in best or s < best[q][0]: best[q] = (s, g1, g2)
            chosen[r].append((best[False][1], best[True][1], best[True][2], best[True][0] < best[False][0]))

            for q, name in ((False, "lin"), (True, "quad")):           # refit on all training
                _, g1, g2 = best[q]
                O = fit(ar[:, i1] / SC, ar[:, i2] / SC, g1, g2, q)
                res.setdefault((r, name), []).append(curve(pred(O, q), test_rs))
            res.setdefault((r, "persist_r"), []).append(
                curve(lambda k0: mu + Ur @ np.repeat(ar[:, [k0]], H + 1, 1), test_rs))
        res.setdefault(("all", "persist"), []).append(curve(lambda k0: np.repeat(X[:, [k0]], H + 1, 1), test_rs))
        res.setdefault(("all", "clim"), []).append(curve(lambda k0: np.repeat(mu, H + 1, 1), test_rs))
        print(f"fold {f+1}/{K} done ({len(test_rs)} restarts)")

# ---- numbers ----------------------------------------------------------------
fm = lambda key: np.array([c.mean() for c in res[key]])                # per-fold 48 h mean
fp, fc = fm(("all", "persist")), fm(("all", "clim"))
dmd_fm = lambda r: DMDC[f"{r}_dmd"].mean(1) if DMDC is not None and f"{r}_dmd" in DMDC.files else None

L = [f"POD-OpInf, blocked CV ({K} folds): 48 h restarted error, mean (eps_u+eps_v)/2",
     f"Full-space persistence {fp.mean():.3f} +- {fp.std(ddof=1):.3f} | climatology {fc.mean():.3f} +- {fc.std(ddof=1):.3f}", "",
     "  r | lin             | quad            | DMD (dmd_cv)    | persist_r       | folds lin<p  quad<p  quad<lin | quad-lin",
     ]
rows = []
for r in R_LIST:
    l, q, p, d = fm((r, "lin")), fm((r, "quad")), fm((r, "persist_r")), dmd_fm(r)
    ds = f"{d.mean():.3f} +- {d.std(ddof=1):.3f}" if d is not None else "      n/a      "
    L.append(f"{r:4d} | {l.mean():.3f} +- {l.std(ddof=1):.3f} | {q.mean():.3f} +- {q.std(ddof=1):.3f} | {ds} | "
             f"{p.mean():.3f} +- {p.std(ddof=1):.3f} | {int((l<p).sum())}/{K}   {int((q<p).sum())}/{K}   "
             f"{int((q<l).sum())}/{K}  | {(q-l).mean():+.4f}")
    rows.append([r, l.mean(), l.std(ddof=1), q.mean(), q.std(ddof=1), p.mean(), p.std(ddof=1),
                 d.mean() if d is not None else np.nan, int((l < p).sum()), int((q < p).sum()),
                 int((q < l).sum()), (q - l).mean()])
L += ["", "Ridge chosen per fold (g_lin, g_quad, quad kept by inner validation):"]
for r in R_LIST:
    L.append(f"  r={r}: " + "; ".join(f"({g1:g}, {g2:g}, {'Q' if k else 'L'})" for _, g1, g2, k in
                                      [(0, c[1], c[2], c[3]) for c in chosen[r]]))
(OUT / "summary.txt").write_text("\n".join(L)); print("\n".join(L))
with open(OUT / "opinf_cv_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["r", "lin_mean", "lin_std", "quad_mean", "quad_std", "persist_r_mean", "persist_r_std",
                "dmd_mean", "folds_lin_better", "folds_quad_better", "folds_quad_beats_lin", "quad_minus_lin"])
    w.writerows(rows)
np.savez(DATA / "opinf_cv_curves.npz", **{f"{k[0]}_{k[1]}": np.array(v) for k, v in res.items()})

# ---- Fig 1: error vs r ------------------------------------------------------
R, rw = np.array(R_LIST), np.array(rows)
fig, ax = plt.subplots(figsize=(W1, 2.7))
ax.errorbar(R - 0.8, rw[:, 1], rw[:, 2], color=C["green"], marker="s", capsize=2, label="OpInf linear")
ax.errorbar(R, rw[:, 3], rw[:, 4], color=C["blue"], marker="o", capsize=2, label="OpInf quadratic")
ax.errorbar(R + 0.8, rw[:, 5], rw[:, 6], color="gray", marker="+", ls="-.", capsize=2, label="Persistence (rank $r$)")
if DMDC is not None: ax.plot(R, rw[:, 7], "--^", color=C["orange"], label="POD-DMD")
ax.axhline(fp.mean(), color="k", ls="-.", lw=0.6, label="Persistence")
ax.axhline(fc.mean(), color="k", ls=":", lw=0.8, label="Climatology")
ax.set(xlabel="Number of modes $r$", ylabel=r"48 h mean $(\varepsilon_u+\varepsilon_v)/2$", xticks=R_LIST)
ax.legend(fontsize=6); save(fig, OUT, "cv_error_vs_r")

# ---- Fig 2: error vs lead time at r = R_PLOT --------------------------------
fig, ax = plt.subplots(figsize=(W1, 2.7))
items = [((R_PLOT, "lin"), C["green"], "-", "OpInf linear"), ((R_PLOT, "quad"), C["blue"], "-", "OpInf quadratic"),
         ((R_PLOT, "persist_r"), "gray", "-.", "Persistence (rank $r$)"),
         (("all", "persist"), "k", "-.", "Persistence"), (("all", "clim"), "k", ":", "Climatology")]
for key, col, ls, lab in items:
    cc = np.array(res[key]); m, s = cc.mean(0), cc.std(0, ddof=1)
    ax.plot(m, color=col, ls=ls, label=lab); ax.fill_between(range(H + 1), m - s, m + s, color=col, alpha=0.12, lw=0)
if DMDC is not None and f"{R_PLOT}_dmd" in DMDC.files:
    ax.plot(DMDC[f"{R_PLOT}_dmd"].mean(0), color=C["orange"], ls="--", label="POD-DMD")
ax.set(xlabel="Lead time (h)", ylabel=r"$(\varepsilon_u+\varepsilon_v)/2$", xlim=(0, H))
ax.legend(fontsize=6); save(fig, OUT, "cv_error_vs_lead")

# ---- Fig 3: per-fold quad - lin at each r -----------------------------------
fig, ax = plt.subplots(figsize=(W1, 2.4))
for r in R_LIST:
    d = fm((r, "quad")) - fm((r, "lin"))
    ax.plot(np.full(K, r), d, ".", color="gray", ms=4)
    ax.plot(r, d.mean(), "_", color=C["blue"], ms=10, mew=1.5)
ax.axhline(0, color="k", lw=0.6)
ax.set(xlabel="Number of modes $r$", ylabel="Quadratic $-$ linear 48 h error", xticks=R_LIST)
save(fig, OUT, "cv_quad_minus_lin")