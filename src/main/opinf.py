# src/main/opinf.py -- Phase 3: POD-OpInf (affine-linear and quadratic) vs POD-DMD
# Needs: data/pod_basis.npz, data/X_snapshots.npz, (optional) data/dmd_models.npz
# Out:   results/opinf/*.pdf|png|csv|txt, data/opinf_models.npz
import sys, time, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SRC  = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.append(str(SRC))
from style import W1, W2, C, save

DATA = ROOT / "data"
OUT  = ROOT / "results" / "opinf"; OUT.mkdir(parents=True, exist_ok=True)

R_LIST = [5, 10, 15, 25, 35, 51]               # quadratic OpInf is only well posed for small r
R_PLOT = 25                                    # r used in the lead-time figure
G_LIN  = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]       # ridge on c, A
G_QUAD = [1e-2, 1e-1, 1, 10, 100]              # ridge on H
H, STRIDE, SEL_STRIDE = 48, 6, 12              # 48 h test; coarser restart stride for tuning
BLOW = 3.0                                     # divergence bound = BLOW x max training norm

# ---- load -------------------------------------------------------------------
b = np.load(DATA / "pod_basis.npz")
Ur_all, x_mean, A_all = b["Ur"], b["x_mean"], b["A_all"]
n_tr, n_val, Ng = int(b["n_tr"]), int(b["n_val"]), int(b["Ng"])
X = np.load(DATA / "X_snapshots.npz")["X"].astype(np.float64)
win = {"val": (n_tr, n_tr + n_val), "test": (n_tr + n_val, X.shape[1])}
SC = A_all[0, :n_tr].std()                     # one global scale keeps the energy weighting
dmd_file = DATA / "dmd_models.npz"
DMD = np.load(dmd_file) if dmd_file.exists() else None

# ---- helpers ----------------------------------------------------------------
def rel_err(Xh, Xt):
    f = lambda s: np.linalg.norm(Xh[s] - Xt[s], axis=0) / np.linalg.norm(Xt[s], axis=0)
    return f(slice(0, Ng)), f(slice(Ng, None))

def features(a, iu, q):
    parts = [np.ones((1, a.shape[1])), a]
    if q: parts.append(a[iu[0]] * a[iu[1]])
    return np.vstack(parts)

def fit(a_tr, g1, g2, q):
    """Ridge OpInf: a_{k+1} = c + A a_k (+ H (a_k x a_k) compact). Returns O = [c A (H)]."""
    r = a_tr.shape[0]; iu = np.triu_indices(r)
    D, Y = features(a_tr[:, :-1], iu, q), a_tr[:, 1:]
    reg = np.r_[0.0, np.full(r, g1), np.full(len(iu[0]) if q else 0, g2)] * D.shape[1]
    return np.linalg.solve(D @ D.T + np.diag(reg), D @ Y.T).T

def rollout(m, a0, K):
    """Free run in scaled coordinates; states beyond the bound are saturated and counted."""
    r, O, q, iu = m["r"], m["O"], m["q"], m["iu"]
    a = np.empty((r, K)); a[:, 0] = a0
    for k in range(1, K):
        x = a[:, k - 1]
        f = np.r_[1.0, x, x[iu[0]] * x[iu[1]]] if q else np.r_[1.0, x]
        y = O @ f; n = np.linalg.norm(y)
        if not np.isfinite(n): y = x.copy(); m["ndiv"] += 1
        elif n > m["bound"]:   y = y * m["bound"] / n; m["ndiv"] += 1
        a[:, k] = y
    return a

def opinf_pred(m):
    def f(k0, K=H + 1):
        return x_mean + Ur_all[:, :m["r"]] @ (SC * rollout(m, A_all[:m["r"], k0] / SC, K))
    return f

def dmd_pred(A, r):
    def f(k0, K=H + 1):
        a = np.empty((r, K)); a[:, 0] = A_all[:r, k0]
        for k in range(1, K): a[:, k] = A @ a[:, k - 1]
        return x_mean + Ur_all[:, :r] @ a
    return f

def short_curve(pred, s, e, stride):
    out = []
    for k0 in range(s, e - H, stride):
        eu, ev = rel_err(pred(k0), X[:, k0:k0 + H + 1]); out.append((eu + ev) / 2)
    return np.mean(out, axis=0)

def free_curve(pred, s, e):
    eu, ev = rel_err(pred(s, e - s), X[:, s:e]); return (eu + ev) / 2, eu.mean(), ev.mean()

persist = lambda k0, K=H + 1: np.repeat(X[:, [k0]], K, axis=1)
clim    = lambda k0, K=H + 1: np.repeat(x_mean, K, axis=1)
persist_r = lambda r: (lambda k0, K=H + 1: x_mean + Ur_all[:, :r] @ np.repeat(A_all[:r, k0:k0 + 1], K, axis=1))

# ---- baselines --------------------------------------------------------------
base = {(n, k): (short_curve(f, *win[n], STRIDE), free_curve(f, *win[n])[0])
        for n in win for k, f in (("persist", persist), ("clim", clim))}

# ---- fit, select (validation only), evaluate --------------------------------
rows, curves, grid25, models = [], {}, None, {}
with np.errstate(all="ignore"):
    for r in R_LIST:
        iu = np.triu_indices(r)
        a_tr = A_all[:r, :n_tr] / SC
        bound = BLOW * np.linalg.norm(a_tr, axis=0).max()
        mk = lambda O, q: dict(O=O, r=r, q=q, iu=iu, bound=bound, ndiv=0)

        sel, grid = {}, {}
        for q in (False, True):
            for g1 in G_LIN:
                for g2 in (G_QUAD if q else [0]):
                    sc = short_curve(opinf_pred(mk(fit(a_tr, g1, g2, q), q)), *win["val"], SEL_STRIDE).mean()
                    sc = np.inf if not np.isfinite(sc) else sc
                    grid[(q, g1, g2)] = sc
                    if q not in sel or sc < sel[q][0]: sel[q] = (sc, g1, g2)
        if r == R_PLOT: grid25 = grid
        keep_quad = sel[True][0] < sel[False][0]                # keep H only if validation improves

        fl = {n: free_curve(persist_r(r), *win[n])[1:] for n in win}
        floor_t = np.mean([rel_err(x_mean + Ur_all[:, :r] @ A_all[:r, win["test"][0]:], X[:, win["test"][0]:])[i].mean() for i in (0, 1)])
        pr = {n: short_curve(persist_r(r), *win[n], STRIDE).mean() for n in win}
        dm = None
        if DMD is not None and f"A_{r}" in DMD.files:
            dm = {n: short_curve(dmd_pred(DMD[f"A_{r}"], r), *win[n], STRIDE) for n in win}

        for q, name in ((False, "lin"), (True, "quad")):
            _, g1, g2 = sel[q]
            t0 = time.perf_counter(); O = fit(a_tr, g1, g2, q); tf = time.perf_counter() - t0
            m = mk(O, q); models[(r, name)] = m
            row = dict(r=r, model=name, chosen=int(q == keep_quad), g_lin=g1, g_quad=g2,
                       n_params=r + r * r + (r * r * (r + 1) // 2 if q else 0), fit_s=tf,
                       floor_test=floor_t, persist_r_val48=pr["val"], persist_r_test48=pr["test"])
            for n, (s, e) in win.items():
                m["ndiv"] = 0
                sh = short_curve(opinf_pred(m), s, e, STRIDE)
                t0 = time.perf_counter(); fc, fu, fv = free_curve(opinf_pred(m), s, e); tp = time.perf_counter() - t0
                curves[(r, name, n)] = (sh, fc)
                row.update({f"{n}_short48": sh.mean(), f"{n}_free_u": fu, f"{n}_free_v": fv,
                            f"{n}_n_div": m["ndiv"]})
                if dm: row[f"dmd_{n}_short48"] = dm[n].mean(); curves[(r, "dmd", n)] = (dm[n], None)
            row["pred_s"] = tp
            rows.append(row)

# ---- save numbers -----------------------------------------------------------
keys = list(dict.fromkeys(k for q in rows for k in q))
with open(OUT / "opinf_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader()
    for q in rows: w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in q.items()})
chosen = {q["r"]: q["model"] for q in rows if q["chosen"]}
np.savez(DATA / "opinf_models.npz", scale=SC,
         **{f"O_{r}": models[(r, chosen[r])]["O"] for r in R_LIST},
         **{f"quad_{r}": chosen[r] == "quad" for r in R_LIST})

L = ["  r | model  * | g_lin g_quad | val48  test48 | test free u v | #div | dmd test48 | persist_r | floor"]
for q in rows:
    L.append(f"{q['r']:4d} | {q['model']:5s} {'*' if q['chosen'] else ' '} | {q['g_lin']:.0e} {q['g_quad']:.0e} | "
             f"{q['val_short48']:.3f} {q['test_short48']:.3f} | {q['test_free_u']:.3f} {q['test_free_v']:.3f} | "
             f"{q['test_n_div']:4d} | {q.get('dmd_test_short48', float('nan')):.3f}      | "
             f"{q['persist_r_test48']:.3f}     | {q['floor_test']:.3f}")
L += ["", "Full-space baselines, 48 h mean (eps_u+eps_v)/2:  (* = model kept by validation)"]
for k in ("persist", "clim"):
    L.append(f"{k:8s} val {base[('val', k)][0].mean():.3f}  test {base[('test', k)][0].mean():.3f} | "
             f"free val {base[('val', k)][1].mean():.3f} test {base[('test', k)][1].mean():.3f}")
(OUT / "summary.txt").write_text("\n".join(L)); print("\n".join(L))

# ---- Fig 1: 48 h test error vs r --------------------------------------------
ch = lambda name, key: [q[key] for q in rows if q["model"] == name]
fig, ax = plt.subplots(figsize=(W1, 2.6))
ax.plot(R_LIST, ch("lin", "test_short48"), "-s", color=C["green"], label="OpInf linear")
ax.plot(R_LIST, ch("quad", "test_short48"), "-o", color=C["blue"], label="OpInf quadratic")
if DMD is not None: ax.plot(R_LIST, ch("lin", "dmd_test_short48"), "--^", color=C["orange"], label="POD-DMD")
ax.plot(R_LIST, ch("lin", "persist_r_test48"), "-.", color="gray", label="Persistence (rank $r$)")
ax.plot(R_LIST, ch("lin", "floor_test"), ":", color=C["black"], label="POD floor")
ax.axhline(base[("test", "persist")][0].mean(), color="k", ls="-.", lw=0.6)
ax.set(xlabel="Number of modes $r$", ylabel=r"48 h mean $(\varepsilon_u+\varepsilon_v)/2$")
ax.legend(fontsize=6); save(fig, OUT, "error_vs_r")

# ---- Fig 2: error vs lead time at r = R_PLOT --------------------------------
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
sty = {"dmd": ("POD-DMD", C["orange"], "--"), "lin": ("OpInf linear", C["green"], "-"),
       "quad": ("OpInf quadratic", C["blue"], "-")}
for k, (lab, col, ls) in sty.items():
    if (R_PLOT, k, "test") not in curves: continue
    sh, fc = curves[(R_PLOT, k, "test")]
    ax[0].plot(sh, color=col, ls=ls, label=lab)
    if fc is not None: ax[1].plot(fc, color=col, ls=ls, label=lab)
ax[0].plot(short_curve(persist_r(R_PLOT), *win["test"], STRIDE), color="gray", ls="-.", label="Persistence (rank $r$)")
ax[0].plot(base[("test", "persist")][0], color="k", ls="-.", lw=0.7, label="Persistence")
ax[0].plot(base[("test", "clim")][0], color="k", ls=":", label="Climatology")
ax[1].plot(base[("test", "persist")][1], color="k", ls="-.", lw=0.7, label="Persistence")
ax[1].plot(base[("test", "clim")][1], color="k", ls=":", label="Climatology")
ax[0].set(xlabel="Lead time (h)", ylabel=r"$(\varepsilon_u+\varepsilon_v)/2$", xlim=(0, H))
ax[1].set(xlabel="Lead time (h)", ylabel=r"$(\varepsilon_u+\varepsilon_v)/2$", ylim=(0, 2))
ax[0].legend(fontsize=6); ax[1].legend(fontsize=6)
for a, s in zip(ax, "ab"): a.text(0.03, 0.90, f"({s})", transform=a.transAxes, fontsize=8, fontweight="bold")
save(fig, OUT, "error_vs_lead")

# ---- Fig 3: validation score over the (g_lin, g_quad) grid, r = R_PLOT ------
M = np.array([[grid25[(True, g1, g2)] for g2 in G_QUAD] for g1 in G_LIN])
fig, ax = plt.subplots(figsize=(W1, 2.6))
im = ax.imshow(np.log10(np.clip(M, 1e-3, 1e3)), origin="lower", cmap="viridis", aspect="auto")
ax.set_xticks(range(len(G_QUAD)), [f"{g:g}" for g in G_QUAD]); ax.set_yticks(range(len(G_LIN)), [f"{g:g}" for g in G_LIN])
ax.set(xlabel=r"$\gamma_H$ (quadratic)", ylabel=r"$\gamma_{c,A}$ (linear)"); ax.grid(False)
fig.colorbar(im, ax=ax).set_label(r"$\log_{10}$ validation 48 h error")
save(fig, OUT, "gamma_grid")