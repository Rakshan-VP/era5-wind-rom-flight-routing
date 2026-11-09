import sys, time, warnings
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC))
from style import W1, W2, C, save
warnings.filterwarnings("ignore")

OUT = {k: Path(f"results/{k}") for k in ("dmd", "opinf", "comparison")}
for p in OUT.values(): p.mkdir(parents=True, exist_ok=True)
LBL = {"dmd": "POD-DMD", "opinf": "POD-OpInf"}

# ---- load data + POD basis (train = 2022-25, test = 2026) ----
d = np.load("data/snapshot_matrix.npz"); pod = np.load("data/pod.npz")
X, years = d["X"].astype(np.float64), d["years"]
U, S, xbar = pod["U"], pod["S"], pod["xbar"]
n = U.shape[0]; Ng = n // 2
tr = years < years.max(); Xte = X[:, ~tr]; mte = Xte.shape[1]
ty = np.unique(years[tr]); yv = ty[-1]                       # OpInf validation year = 2025
E = np.cumsum(S**2) / np.sum(S**2)
P = [0.90, 0.95, 0.99]
R = [int(np.searchsorted(E, q)) + 1 for q in P]
print("r for 90/95/99% energy:", R)
H, STRIDE = 72, 24
GAM = 10.0 ** np.arange(-1, 6)
cols = [C["blue"], C["orange"], C["green"]]
lab = {r: f"{q:.0%} ($r$={r})" for q, r in zip(P, R)}
day = np.arange(mte) / 24
smooth = lambda e: np.convolve(e, np.ones(24) / 24, mode="same")

def rel(Xh, Xt):                                             # rows: [eps_u, eps_v] per snapshot
    return np.stack([np.linalg.norm(Xh[:Ng] - Xt[:Ng], axis=0) / np.linalg.norm(Xt[:Ng], axis=0),
                     np.linalg.norm(Xh[Ng:] - Xt[Ng:], axis=0) / np.linalg.norm(Xt[Ng:], axis=0)])

# ---- shared helpers (both models work on std-scaled coefficients z) ----
def guard(Zn):                                               # blow-up -> NaN
    Zn[:, ~np.isfinite(Zn).all(0) | (np.abs(Zn).max(0) > 1e3)] = np.nan
    return Zn

def pairs(Z, yrs):                                           # pairs inside each January only
    idx = [np.where(years == y)[0] for y in yrs]
    return np.hstack([Z[:, i[:-1]] for i in idx]), np.hstack([Z[:, i[1:]] for i in idx])

def lead(f, Z, s, Ur, idx):                                  # restart from ERA5, run H h
    e = np.full((2, H + 1), np.nan); Zc = Z[:, idx].copy()
    for h in range(H + 1):
        if h: Zc = f(Zc)
        if np.isfinite(Zc).any():
            e[:, h] = np.nanmean(rel(xbar + Ur @ (Zc * s[:, None]), X[:, idx + h]), axis=1)
    return e

# ---- OpInf helpers ----
def feats(Z, quad):                                          # [1; z; q(z)]
    F = [np.ones((1, Z.shape[1])), Z]
    if quad: i, j = np.triu_indices(Z.shape[0]); F.append(Z[i] * Z[j])
    return np.vstack(F)

def ridge(D, Y, gams):                                       # D (N,d), Y (N,r) -> list of W (d,r)
    N, dd = D.shape
    if dd <= N:
        w, V = np.linalg.eigh(D.T @ D); B = V.T @ (D.T @ Y)
        return [V @ (B / (np.maximum(w, 0) + g)[:, None]) for g in gams]
    w, V = np.linalg.eigh(D @ D.T); B = V.T @ Y
    return [D.T @ (V @ (B / (np.maximum(w, 0) + g)[:, None])) for g in gams]

mk = lambda W, quad: (lambda Zc: guard(W.T @ feats(Zc, quad)))
score = lambda e: np.nanmean(e) if np.isfinite(e).all() else np.inf

# ---- evaluate one fitted model (identical metrics for DMD and OpInf) ----
def run(f, r, q, Z, s, Ur, e_proj, **kw):
    zte = Z[:, ~tr]
    t0 = time.perf_counter()
    zf = np.full((r, mte), np.nan); zf[:, 0] = zte[:, 0]; z = zf[:, [0]]
    for k in range(mte - 1):
        z = f(z); zf[:, k + 1] = z[:, 0]
        if np.isnan(z).any(): break
    a = zf * s[:, None]; xh = xbar + Ur @ a
    t_pred = time.perf_counter() - t0
    e_free = rel(xh, Xte)
    blow = int(np.argmax(np.isnan(zf[0]))) if np.isnan(zf[0]).any() else -1
    tst = np.where(~tr)[0][np.arange(0, mte - H, STRIDE)]
    e_lead = lead(f, Z, s, Ur, tst)
    m = dict(r=r, energy=q, eps_u=np.nanmean(e_free[0]), eps_v=np.nanmean(e_free[1]),
             eps_u_24=e_lead[0, 24], eps_v_24=e_lead[1, 24], eps_u_72=e_lead[0, 72], eps_v_72=e_lead[1, 72],
             proj_u=e_proj[0], proj_v=e_proj[1], t_pred=t_pred, blow=blow, **kw)
    m["eps_free"] = (m["eps_u"] + m["eps_v"]) / 2; m["proj"] = e_proj.mean()
    m["eps_24"] = (m["eps_u_24"] + m["eps_v_24"]) / 2; m["eps_72"] = (m["eps_u_72"] + m["eps_v_72"]) / 2
    m["ex_24"] = 100 * (m["eps_24"] - m["proj"]) / m["proj"]      # excess over projection floor [%]
    m["ex_72"] = 100 * (m["eps_72"] - m["proj"]) / m["proj"]
    m["storage"] = (n * r + m["params"] + n) * 8 / 2**20
    return dict(m=m, e_free=e_free, e_lead=e_lead, a=a)

res = {"dmd": {}, "opinf": {}}
for q, r in zip(P, R):
    Ur = U[:, :r]; Ac = Ur.T @ (X - xbar)
    s = Ac[:, tr].std(axis=1); Z = Ac / s[:, None]
    e_proj = rel(xbar + Ur @ Ac[:, ~tr], Xte).mean(axis=1)

    # ================= POD-DMD =================
    t0 = time.perf_counter()
    Z1, Z2 = pairs(Z, ty); Az = Z2 @ np.linalg.pinv(Z1)
    t_fit = time.perf_counter() - t0
    lam = np.linalg.eigvals(Az)
    v = run(lambda Zc: guard(Az @ Zc), r, q, Z, s, Ur, e_proj, rho=np.abs(lam).max(), t_fit=t_fit,
            params=r * r, model="linear", gamma=np.nan)
    v["lam"] = lam; res["dmd"][r] = v
    np.savez(f"data/dmd_r{r}.npz", A=Az * s[:, None] / s[None, :], s=s, a_free=v["a"], lam=lam)

    # ================= POD-OpInf =================
    ivs = np.where(years == yv)[0]; vst = ivs[np.arange(0, len(ivs) - H, STRIDE)]
    Z1, Z2 = pairs(Z, ty[:-1]); val = {}
    for quad in (False, True):
        if quad and (r * (r + 1) // 2 + r + 1) * Z1.shape[1] * 8 > 3e9: continue   # memory guard
        Ws = ridge(feats(Z1, quad).T, Z2.T, GAM)
        val[quad] = np.array([score(lead(mk(W, quad), Z, s, Ur, vst)) for W in Ws])
    quad = True in val and val[True].min() < val[False].min()
    g = GAM[np.argmin(val[quad])]
    t0 = time.perf_counter()
    Z1, Z2 = pairs(Z, ty); W = ridge(feats(Z1, quad).T, Z2.T, [g])[0]
    t_fit = time.perf_counter() - t0
    rho = np.abs(np.linalg.eigvals(W[1:1 + r].T)).max()
    npar = r + r * r + (r * r * (r + 1) // 2 if quad else 0)
    v = run(mk(W, quad), r, q, Z, s, Ur, e_proj, rho=rho, t_fit=t_fit, params=npar,
            model="quadratic" if quad else "linear", gamma=g)
    v["val"] = val; res["opinf"][r] = v
    np.savez(f"data/opinf_r{r}.npz", W=W, s=s, quad=quad, a_free=v["a"])

    # ---- individual txt per method ----
    for name in res:
        m = res[name][r]["m"]
        (OUT[name] / f"metrics_{name}_r{r}.txt").write_text(
            f"{LBL[name]}  r={r} ({q:.0%} energy)\n"
            + (f"model / gamma         : {m['model']} / {m['gamma']:g}\n" if name == "opinf" else "")
            + "free-run eps_u        : {eps_u:.4f}\nfree-run eps_v        : {eps_v:.4f}\n"
              "24 h lead eps_u / v   : {eps_u_24:.4f} / {eps_v_24:.4f}\n"
              "72 h lead eps_u / v   : {eps_u_72:.4f} / {eps_v_72:.4f}\n"
              "projection floor u / v: {proj_u:.4f} / {proj_v:.4f}\n"
              "spectral radius       : {rho:.5f}\nfit time [s]          : {t_fit:.4f}\n"
              "predict time [s]      : {t_pred:.4f}\nparameters            : {params}\n"
              "storage [MiB]         : {storage:.2f}\n".format(**m)
            + (f"blow-up hour (-1=none): {m['blow']}\n" if name == "opinf" else ""))

# ---- summary table per method ----
for name in res:
    hdr = f"{'r':>5}{'E':>6}{'eps_u':>8}{'eps_v':>8}{'u24h':>8}{'v24h':>8}{'u72h':>8}{'v72h':>8}{'projU':>8}{'projV':>8}{'rho':>8}{'fit[s]':>8}{'pred[s]':>9}{'MiB':>8}"
    if name == "opinf": hdr += f"{'model':>11}{'gamma':>8}{'blow':>6}"
    rows = []
    for m in (v["m"] for v in res[name].values()):
        row = (f"{m['r']:>5}{m['energy']:>6.0%}{m['eps_u']:>8.3f}{m['eps_v']:>8.3f}{m['eps_u_24']:>8.3f}{m['eps_v_24']:>8.3f}"
               f"{m['eps_u_72']:>8.3f}{m['eps_v_72']:>8.3f}{m['proj_u']:>8.3f}{m['proj_v']:>8.3f}{m['rho']:>8.4f}"
               f"{m['t_fit']:>8.3f}{m['t_pred']:>9.3f}{m['storage']:>8.1f}")
        if name == "opinf": row += f"{m['model']:>11}{m['gamma']:>8g}{m['blow']:>6}"
        rows.append(row)
    tab = f"{LBL[name]} summary (test year 2026)\n" + hdr + "\n" + "\n".join(rows) + "\n"
    print(tab); (OUT[name] / f"summary_{name}.txt").write_text(tab)

# ================= comparison tables (txt + LaTeX) =================
# (key, txt label, latex label, format)
ACC = [("eps_u", "free-run eps_u", "$\\varepsilon_u$ free-run", ".4f"),
       ("eps_v", "free-run eps_v", "$\\varepsilon_v$ free-run", ".4f"),
       ("eps_u_24", "24 h eps_u", "$\\varepsilon_u$ 24 h", ".4f"),
       ("eps_v_24", "24 h eps_v", "$\\varepsilon_v$ 24 h", ".4f"),
       ("eps_u_72", "72 h eps_u", "$\\varepsilon_u$ 72 h", ".4f"),
       ("eps_v_72", "72 h eps_v", "$\\varepsilon_v$ 72 h", ".4f"),
       ("eps_free", "free-run mean eps_uv", "$\\bar\\varepsilon_{u,v}$ free-run", ".4f"),
       ("eps_24", "24 h mean eps_uv", "$\\bar\\varepsilon_{u,v}$ 24 h", ".4f"),
       ("eps_72", "72 h mean eps_uv", "$\\bar\\varepsilon_{u,v}$ 72 h", ".4f"),
       ("ex_24", "24 h excess/floor [%]", "24 h excess over floor [\\%]", ".1f"),
       ("ex_72", "72 h excess/floor [%]", "72 h excess over floor [\\%]", ".1f")]
COST = [("rho", "spectral radius", "$\\rho(A)$", ".4f"),
        ("t_fit", "fit time [s]", "Fit time [s]", ".4f"),
        ("t_pred", "predict time [s]", "Predict time [s]", ".4f"),
        ("params", "parameters", "Parameters", "d"),
        ("storage", "storage [MiB]", "Storage [MiB]", ".2f")]
pct = lambda a, b: 100 * (b - a) / a if a else np.nan          # change of OpInf w.r.t. DMD; <0 = OpInf lower

blocks = ["POD-DMD vs POD-OpInf, test year 2026\n"
          "change [%] = 100*(OpInf - DMD)/DMD ; negative = OpInf lower (for errors: OpInf better)\n"]
for r in R:
    D, O = res["dmd"][r]["m"], res["opinf"][r]["m"]
    L = [f"r={r} ({D['energy']:.0%} energy) | projection floor {D['proj']:.4f} | "
         f"OpInf: {O['model']}, gamma={O['gamma']:g}, blow-up h={O['blow']}",
         f"{'metric':<24}{'DMD':>11}{'OpInf':>11}{'change[%]':>11}"]
    for k, t, _, fm in ACC + COST:
        L.append(f"{t:<24}{D[k]:>11{fm}}{O[k]:>11{fm}}{pct(D[k], O[k]):>+11.1f}")
    txt = "\n".join(L) + "\n"
    (OUT["comparison"] / f"comparison_r{r}.txt").write_text(txt); blocks.append(txt)
allt = "\n".join(blocks); print(allt); (OUT["comparison"] / "summary_comparison.txt").write_text(allt)

def tex(items, cap, lb):
    L = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{cap}}}", f"\\label{{{lb}}}",
         "\\begin{tabular}{rlrrr}", "\\toprule", "$r$ & Metric & DMD & OpInf & $\\Delta$ [\\%] \\\\", "\\midrule"]
    for i, r in enumerate(R):
        D, O = res["dmd"][r]["m"], res["opinf"][r]["m"]
        for j, (k, _, tx, fm) in enumerate(items):
            L.append(f"{r if j == 0 else ''} & {tx} & {D[k]:{fm}} & {O[k]:{fm}} & {pct(D[k], O[k]):+.1f} \\\\")
        if i < len(R) - 1: L.append("\\midrule")
    return "\n".join(L + ["\\bottomrule", "\\end{tabular}", "\\end{table}"])

nf = {r: (res["opinf"][r]["m"]["model"], res["opinf"][r]["m"]["gamma"]) for r in R}
(OUT["comparison"] / "comparison_table.tex").write_text(
    "% OpInf model/gamma: " + "; ".join(f"r={r}: {a}, {b:g}" for r, (a, b) in nf.items()) + "\n"
    "% Delta = 100(OpInf-DMD)/DMD, negative = OpInf lower\n"
    + tex(ACC, "Forecast accuracy of POD-DMD and POD-OpInf on the 2026 test month (same $r$, same basis).", "tab:rom_acc")
    + "\n\n" + tex(COST, "Cost and stability of POD-DMD and POD-OpInf.", "tab:rom_cost"))

# ================= plots (single column) =================
def err_fig(name):
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.2))
    for (r, v), c in zip(res[name].items(), cols):
        ax[0].plot(np.arange(H + 1), v["e_lead"].mean(0), c=c, label=lab[r])
        ax[0].axhline(v["m"]["proj"], c=c, ls=":", lw=0.6)
        e = v["e_free"].mean(0)
        ax[1].plot(day, e, c=c, lw=0.4, alpha=0.35)
        ax[1].plot(day[12:-12], smooth(e)[12:-12], c=c, label=lab[r])
    for a in ax: a.set_ylim(bottom=0)
    ax[0].set(xlabel="Lead time [h]", ylabel="Mean $\\varepsilon_{u,v}$", title=f"(a) {LBL[name]}, restarted every 24 h",
              xlim=(0, H), xticks=range(0, H + 1, 12))
    ax[0].legend(title="Energy (dotted = projection floor)", title_fontsize=6)
    ax[1].set(xlabel="Test day (free-run)", ylabel="$\\varepsilon_{u,v}(k)$", title="(b) Free-run, 24 h running mean",
              xlim=(0, mte / 24))
    ax[1].legend(title="Energy", title_fontsize=6)
    for a in ax: a.title.set_fontsize(8)
    save(fig, OUT[name], f"{name}_error")
for name in res: err_fig(name)

# ---- DMD eigenvalues ----
th = np.linspace(0, 2 * np.pi, 400); ph = 2 * np.pi / 24
fig, ax = plt.subplots(figsize=(W1, 3.3))
ax.plot(np.cos(th), np.sin(th), "k", lw=0.6, label="Unit circle")
ax.plot([0, np.cos(ph)], [0, np.sin(ph)], "--", c="gray", lw=0.6, label="24 h (daily) angle")
for q, r, c, m_, s_ in zip(P, R, cols, ["o", "s", "^"], [4.5, 3.5, 2.5]):
    lam = res["dmd"][r]["lam"]
    ax.plot(lam.real, lam.imag, m_, mfc="none", mec=c, ms=s_, mew=0.6,
            label=f"{q:.0%} ($r$={r}, $\\rho$={np.abs(lam).max():.4f})")
    out = np.abs(lam) > 1
    if out.any(): ax.plot(lam.real[out], lam.imag[out], "x", c="r", ms=4, label=f"{q:.0%}: $|\\lambda|>1$")
ax.set(xlabel="Re($\\lambda$)", ylabel="Im($\\lambda$)", aspect="equal", xlim=(-0.2, 1.1), ylim=(-0.65, 0.65))
ax.legend(loc="lower left", fontsize=6, borderaxespad=0.3)
save(fig, OUT["dmd"], "dmd_eigs")

# ---- OpInf validation (model selection) ----
fig, ax = plt.subplots(figsize=(W1, 2.6))
for (r, v), c in zip(res["opinf"].items(), cols):
    ax.loglog(GAM, v["val"][False], "--", c=c, label=f"{lab[r]} linear")
    if True in v["val"]: ax.loglog(GAM, v["val"][True], "-", c=c, label=f"{lab[r]} quadratic")
ax.set(xlabel="Ridge $\\gamma$", ylabel=f"Validation error ({yv}, mean over 72 h)")
ax.legend(fontsize=5.5)
save(fig, OUT["opinf"], "opinf_validation")

# ---- comparison: DMD (solid) vs OpInf (dashed) on the same axes ----
ls = {"dmd": "-", "opinf": "--"}
fig, ax = plt.subplots(2, 1, figsize=(W1, 4.4))
for name in res:
    for (r, v), c, q in zip(res[name].items(), cols, P):
        ax[0].plot(np.arange(H + 1), v["e_lead"].mean(0), c=c, ls=ls[name], label=f"{LBL[name]} {q:.0%}")
        ax[1].plot(day[12:-12], smooth(v["e_free"].mean(0))[12:-12], c=c, ls=ls[name], label=f"{LBL[name]} {q:.0%}")
for (r, v), c in zip(res["dmd"].items(), cols): ax[0].axhline(v["m"]["proj"], c=c, ls=":", lw=0.6)
for a in ax: a.set_ylim(bottom=0)
ax[0].set(xlabel="Lead time [h]", ylabel="Mean $\\varepsilon_{u,v}$", title="(a) Restarted every 24 h (dotted = floor)",
          xlim=(0, H), xticks=range(0, H + 1, 12))
ax[1].set(xlabel="Test day (free-run)", ylabel="$\\varepsilon_{u,v}(k)$", title="(b) Free-run, 24 h running mean",
          xlim=(0, mte / 24))
ax[0].legend(fontsize=5.5, ncol=2)
for a in ax: a.title.set_fontsize(8)
save(fig, OUT["comparison"], "comparison_error")

# ---- comparison: % change of OpInf w.r.t. DMD ----
fig, ax = plt.subplots(figsize=(W1, 2.6)); w = 0.26
for i, (k, t, c) in enumerate([("eps_free", "free-run", cols[0]), ("eps_24", "24 h", cols[1]), ("eps_72", "72 h", cols[2])]):
    ch = [pct(res["dmd"][r]["m"][k], res["opinf"][r]["m"][k]) for r in R]
    ax.bar(np.arange(len(R)) + (i - 1) * w, ch, w, color=c, label=t)
ax.axhline(0, c="k", lw=0.6)
ax.set(xticks=range(len(R)), xticklabels=[f"{q:.0%}\n($r$={r})" for q, r in zip(P, R)],
       ylabel="OpInf vs DMD: change in $\\bar\\varepsilon_{u,v}$ [%]")
ax.legend(title="Error type (<0: OpInf better)", title_fontsize=6, fontsize=6)
save(fig, OUT["comparison"], "comparison_change")