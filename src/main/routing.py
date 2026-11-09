import sys, time, heapq
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC))
from style import W1, W2, C, save

OUT = Path("results/routing"); OUT.mkdir(parents=True, exist_ok=True)
LBL = {"dmd": "POD-DMD", "opinf": "POD-OpInf"}
COL = {"dmd": (C["blue"], "-"), "opinf": ("#D55E00", "--")}

# ---- data + POD basis ----
d = np.load("data/snapshot_matrix.npz"); pod = np.load("data/pod.npz")
X, years = d["X"].astype(np.float64), d["years"]
lat, lon = d["lat"].astype(float), d["lon"].astype(float)
U = pod["U"].astype(np.float64)
S = pod["S"].astype(np.float64).ravel()
xbar = pod["xbar"].astype(np.float64).ravel()
n = U.shape[0]; Ng = n // 2; NLA, NLO = len(lat), len(lon); N = NLA * NLO
tr = years < years.max(); tst = np.where(~tr)[0]; mte = len(tst)
E = np.cumsum(S**2) / np.sum(S**2); P = [0.90, 0.95, 0.99]
R = [int(np.searchsorted(E, q)) + 1 for q in P]
flip = lat[0] > lat[-1]; lat = np.sort(lat)
dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]

# ---- settings ----
Re = 6371e3; HW = 24
DEP = [day * 24 + h for day in (13, 15) for h in (0, 6, 12, 18)]
DAY_KD = {13: 0, 15: 4}
assert max(DEP) + HW < mte
CHN, SIN = (13.00, 80.18), (1.36, 103.99)
VA = 250.0
CASES = {"A": (np.array([VA]), lambda v: np.ones_like(v), 0.0)}
NS = 200

# ---- graph on the ERA5 grid (16 neighbours), edges sorted by source node ----
I, J = np.meshgrid(np.arange(NLA), np.arange(NLO), indexing="ij"); I, J = I.ravel(), J.ravel()
OFF = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)]
ea, eb = [], []
for di, dj in OFF:
    ok = (I + di >= 0) & (I + di < NLA) & (J + dj >= 0) & (J + dj < NLO)
    ea.append((I * NLO + J)[ok]); eb.append(((I + di) * NLO + J + dj)[ok])
ea, eb = np.concatenate(ea), np.concatenate(eb); o = np.argsort(ea, kind="stable"); ea, eb = ea[o], eb[o]
ptr = np.searchsorted(ea, np.arange(N + 1)); NE = len(ea)
LA, LO = lat[np.arange(N) // NLO], lon[np.arange(N) % NLO]

def hav(la1, lo1, la2, lo2):
    a = np.sin(np.radians(la2 - la1) / 2)**2 + np.cos(np.radians(la1)) * np.cos(np.radians(la2)) * np.sin(np.radians(lo2 - lo1) / 2)**2
    return 2 * Re * np.arcsin(np.sqrt(a))

def xyz(P):
    la, lo = np.radians(P[:, 0]), np.radians(P[:, 1])
    return Re * np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)

latm, lonm = (LA[ea] + LA[eb]) / 2, (LO[ea] + LO[eb]) / 2
Dl = hav(LA[ea], LO[ea], LA[eb], LO[eb])
de = np.cos(np.radians(latm)) * np.radians(LO[eb] - LO[ea]); dn = np.radians(LA[eb] - LA[ea])
Ee = np.stack([de, dn], 1) / np.hypot(de, dn)[:, None]
En = np.stack([-Ee[:, 1], Ee[:, 0]], 1)
fi, fj = (latm - lat[0]) / dlat, (lonm - lon[0]) / dlon
i0, j0 = np.clip(fi.astype(int), 0, NLA - 2), np.clip(fj.astype(int), 0, NLO - 2); fy, fx = fi - i0, fj - j0
snap = lambda p: int(np.argmin(hav(LA, LO, p[0], p[1])))
S0, D0 = snap(CHN), snap(SIN)
print(f"grid nodes: Chennai ({LA[S0]:.2f}N,{LO[S0]:.2f}E)  Singapore ({LA[D0]:.2f}N,{LO[D0]:.2f}E)  edges: {NE}")

def mid(Xc):
    nt = Xc.shape[1]; out = np.empty((nt, NE, 2), np.float32)
    for k, f in enumerate((Xc[:Ng].T.reshape(nt, NLA, NLO), Xc[Ng:].T.reshape(nt, NLA, NLO))):
        if flip: f = f[:, ::-1]
        out[..., k] = ((1 - fy) * (1 - fx) * f[:, i0, j0] + (1 - fy) * fx * f[:, i0, j0 + 1]
                       + fy * (1 - fx) * f[:, i0 + 1, j0] + fy * fx * f[:, i0 + 1, j0 + 1])
    return out

# ---- wind sources: ERA5 truth, POD-DMD, POD-OpInf (restart from ERA5 at departure) ----
def feats(Z, quad):
    F = [np.ones((1, Z.shape[1])), Z]
    if quad: i, j = np.triu_indices(Z.shape[0]); F.append(Z[i] * Z[j])
    return np.vstack(F)

_M = {}
def rom(method, r, x0, nt):
    if (method, r) not in _M: _M[(method, r)] = dict(np.load(f"data/{method}_r{r}.npz"))
    m = _M[(method, r)]; s = m["s"]; Ur = U[:, :r]
    if method == "dmd": f = lambda a: m["A"] @ a
    else:
        W, quad = m["W"], bool(m["quad"]); f = lambda a: s * (W.T @ feats((a / s)[:, None], quad))[:, 0]
    a = np.zeros((r, nt)); a[:, 0] = Ur.T @ (x0 - xbar); blow = -1
    with np.errstate(all="ignore"):
        for h in range(1, nt):
            an = f(a[:, h - 1])
            if not np.isfinite(an).all() or np.abs(an / s).max() > 1e3:
                an = a[:, h - 1]; blow = h if blow < 0 else blow
            a[:, h] = an
    return xbar[:, None] + Ur @ a, blow

def rel(Xh, Xt):
    return np.stack([np.linalg.norm(Xh[:Ng] - Xt[:Ng], axis=0) / np.linalg.norm(Xt[:Ng], axis=0),
                     np.linalg.norm(Xh[Ng:] - Xt[Ng:], axis=0) / np.linalg.norm(Xt[Ng:], axis=0)])

# ---- edge cost, time-dependent Dijkstra, evaluation ----
def edge(w, e, nn, dd, case):
    VAS, md, ct = CASES[case]
    wl, wn = (w * e).sum(1)[:, None], (w * nn).sum(1)[:, None]; ro = VAS[None]**2 - wn**2
    Vg = wl + np.sqrt(np.maximum(ro, 0)); Vg = np.where((ro > 0) & (Vg > 10), Vg, np.nan)
    dt = dd[:, None] / Vg; Jc = np.where(np.isnan(dt), np.inf, (md(VAS)[None] + ct) * dt)
    k = Jc.argmin(1); ar = np.arange(len(dd))
    return np.where(np.isnan(dt[ar, k]), np.inf, dt[ar, k]), Jc[ar, k], k

def wat(Wm, i, t):
    h = min(t / 3600, HW - 1e-6); h0 = int(h); f = h - h0
    return (1 - f) * Wm[h0, i].astype(float) + f * Wm[h0 + 1, i]

def route(Wm, case):
    VAS = CASES[case][0]; Jb = np.full(N, np.inf); Tb = np.zeros(N); pe = np.full(N, -1); pv = np.zeros(N)
    Jb[S0] = 0; hp = [(0.0, S0)]
    while hp:
        J0, u = heapq.heappop(hp)
        if J0 > Jb[u]: continue
        if u == D0: break
        sl = slice(ptr[u], ptr[u + 1]); h = min(Tb[u] / 3600, HW - 1e-6); h0 = int(h); f = h - h0
        w = (1 - f) * Wm[h0, sl].astype(float) + f * Wm[h0 + 1, sl]
        dt, Jc, k = edge(w, Ee[sl], En[sl], Dl[sl], case)
        for q in range(ptr[u + 1] - ptr[u]):
            v = eb[ptr[u] + q]; Jn = J0 + Jc[q]
            if Jn < Jb[v]: Jb[v] = Jn; Tb[v] = Tb[u] + dt[q]; pe[v] = ptr[u] + q; pv[v] = VAS[k[q]]; heapq.heappush(hp, (Jn, v))
    eids, va, v = [], [], D0
    while v != S0: eids.append(pe[v]); va.append(pv[v]); v = ea[pe[v]]
    eids, va = eids[::-1], va[::-1]; nodes = [ea[eids[0]]] + [eb[i] for i in eids]
    return dict(eids=eids, va=va, P=np.c_[LA[nodes], LO[nodes]], T_own=Tb[D0])

def evaluate(rt, Wm, case):
    md = CASES[case][1]; t = F = 0.0
    for i, v in zip(rt["eids"], rt["va"]):
        w = wat(Wm, i, t); dt = Dl[i] / (w @ Ee[i] + np.sqrt(v * v - (w @ En[i])**2))
        t += dt; F += float(md(np.float64(v))) * dt
    return t, F

def resamp(Pl, ns=NS):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz(Pl), axis=0), axis=1))]; u = np.linspace(0, 1, ns) * s[-1]
    return np.c_[np.interp(u, s, Pl[:, 0]), np.interp(u, s, Pl[:, 1])], u / 1e3

def xte(Pq, Qref):
    p, q = xyz(Pq), xyz(Qref); a, ab = q[:-1], np.diff(q, axis=0)
    t = np.clip(((p[:, None] - a) * ab).sum(-1) / (ab**2).sum(-1), 0, 1); c = a + t[..., None] * ab
    dd = np.linalg.norm(p[:, None] - c, axis=-1); j = dd.argmin(1); ar = np.arange(len(p))
    side = np.sign(np.einsum("ij,ij->i", np.cross(ab[j], p - c[ar, j]), a[j]))
    return side * dd[ar, j] / 1e3

# ---- main loop: same solver / departures / parameters on all wind sources (Case A) ----
rows, routes = [], {}
for case in CASES:
    for kd, dep in enumerate(DEP):
        g = tst[dep]; truth = X[:, g:g + HW + 1]; Wt = mid(truth)
        t0 = time.perf_counter(); ref = route(Wt, case); t_ref = time.perf_counter() - t0
        Tref, Fref = evaluate(ref, Wt, case); Pr, kmr = resamp(ref["P"])
        routes[(case, kd, "era5", 0)] = dict(P=ref["P"], km=kmr, T=Tref / 3600)
        nw = route(np.zeros_like(Wt), case); Tnw, _ = evaluate(nw, Wt, case)
        routes[(case, kd, "nowind", 0)] = dict(P=nw["P"], T=Tnw / 3600)
        for r in R:
            for nm in LBL:
                t0 = time.perf_counter(); xr, blow = rom(nm, r, X[:, g], HW + 1); Wr = mid(xr); t_wind = time.perf_counter() - t0
                t0 = time.perf_counter(); rt = route(Wr, case); t_route = time.perf_counter() - t0
                T, F = evaluate(rt, Wt, case)
                Pq, _ = resamp(rt["P"]); xt = xte(Pq, ref["P"])
                rows.append(dict(case=case, dep=dep, name=nm, r=r, Tref=Tref / 3600, T=T / 3600, T_own=rt["T_own"] / 3600,
                                 eps_T=100 * abs(T - Tref) / Tref, eps_F=100 * abs(F - Fref) / Fref,
                                 delta=hav(Pq[:, 0], Pq[:, 1], Pr[:, 0], Pr[:, 1]).mean() / 1e3,
                                 xte_mean=np.abs(xt).mean(), xte_max=np.abs(xt).max(), eps_win=rel(xr, truth).mean(),
                                 t_wind=t_wind, t_route=t_route, blow=blow))
                routes[(case, kd, nm, r)] = dict(P=rt["P"], km=kmr, xte=xt, T=T / 3600)
        print(f"case {case}, dep {dep}h done  (ERA5 T = {Tref / 3600:.3f} h, no-wind route T = {Tnw / 3600:.3f} h)")

# ---- individual txt per method/r + aggregated summary ----
KEYS = ["Tref", "T", "T_own", "eps_T", "eps_F", "delta", "xte_mean", "xte_max", "eps_win", "t_wind", "t_route"]
AG = {}
for case in CASES:
    for nm in LBL:
        for r in R:
            rr = [x for x in rows if x["case"] == case and x["name"] == nm and x["r"] == r]
            AG[(case, nm, r)] = {k: np.mean([x[k] for x in rr]) for k in KEYS} | {"xte_max": max(x["xte_max"] for x in rr),
                                                                                  "nblow": sum(x["blow"] >= 0 for x in rr)}
            L = [f"Route case {case}, {LBL[nm]}, r={r}  (Chennai -> Singapore, 13 & 15 Jan 2026, {HW} h windows)",
                 f"{'dep[h]':>7}{'Tref[h]':>9}{'Ttrue[h]':>10}{'Town[h]':>9}{'epsT[%]':>9}{'epsF[%]':>9}{'delta[km]':>11}{'XTEmean':>9}{'XTEmax':>8}{'eps_uv':>8}{'blow':>6}"]
            L += [f"{x['dep']:>7}{x['Tref']:>9.3f}{x['T']:>10.3f}{x['T_own']:>9.3f}{x['eps_T']:>9.3f}{x['eps_F']:>9.3f}{x['delta']:>11.2f}"
                  f"{x['xte_mean']:>9.2f}{x['xte_max']:>8.2f}{x['eps_win']:>8.3f}{x['blow']:>6}" for x in rr]
            a = AG[(case, nm, r)]
            L.append(f"{'mean':>7}{a['Tref']:>9.3f}{a['T']:>10.3f}{a['T_own']:>9.3f}{a['eps_T']:>9.3f}{a['eps_F']:>9.3f}{a['delta']:>11.2f}"
                     f"{a['xte_mean']:>9.2f}{a['xte_max']:>8.2f}{a['eps_win']:>8.3f}{a['nblow']:>6}")
            (OUT / f"metrics_route_{case}_{nm}_r{r}.txt").write_text("\n".join(L) + "\n")

hdr = f"{'case':>5}{'method':>11}{'r':>5}{'eps_uv':>8}{'epsT[%]':>9}{'epsF[%]':>9}{'delta[km]':>11}{'XTEmean':>9}{'XTEmax':>8}{'wind[s]':>9}{'route[s]':>10}"
tab = ["Routing summary: mean over eight departures (XTEmax = max). XTE = cross-track error vs ERA5 route [km]; eps_uv = wind error over the window", hdr]
for (case, nm, r), a in AG.items():
    tab.append(f"{case:>5}{LBL[nm]:>11}{r:>5}{a['eps_win']:>8.3f}{a['eps_T']:>9.3f}{a['eps_F']:>9.3f}{a['delta']:>11.2f}"
               f"{a['xte_mean']:>9.2f}{a['xte_max']:>8.2f}{a['t_wind']:>9.3f}{a['t_route']:>10.3f}")
ev = np.array([x["eps_win"] for x in rows])
tab.append("\nDoes lower wind error give a better route? Pearson corr(eps_uv, .) over all method/r/departure runs:")
for k in ("eps_T", "eps_F", "delta", "xte_mean"):
    tab.append(f"  corr(eps_uv, {k:<8}) = {np.corrcoef(ev, [x[k] for x in rows])[0, 1]:+.3f}")
tab = "\n".join(tab) + "\n"; print(tab); (OUT / "summary_routing.txt").write_text(tab)

tx = ["\\begin{table}[t]", "\\centering", "\\small",
      "\\caption{Route quality on true ERA5 winds, mean over departures on 13 and 15 Jan 2026.}", "\\label{tab:route}",
      "\\begin{tabular}{llrrrrrr}", "\\toprule",
      "Case & Method & $r$ & $\\bar\\varepsilon_{u,v}$ & $\\varepsilon_T$ [\\%] & $\\varepsilon_F$ [\\%] & $\\delta_{route}$ [km] & XTE$_{max}$ [km] \\\\", "\\midrule"]
for (case, nm, r), a in AG.items():
    tx.append(f"{case} & {LBL[nm]} & {r} & {a['eps_win']:.3f} & {a['eps_T']:.3f} & {a['eps_F']:.3f} & {a['delta']:.1f} & {a['xte_max']:.1f} \\\\")
(OUT / "routing_table.tex").write_text("\n".join(tx + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]))

# ---- Fig 1: cross-track error, three ranks for each date (six plots total) ----
for day, kd in DAY_KD.items():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True, constrained_layout=True)
    print(f"Case A, {day} Jan 2026, departure +{DEP[kd]} h (times on TRUE ERA5 winds)")
    for a, r, q in zip(ax, R, P):
        for nm in LBL:
            v = routes[("A", kd, nm, r)]
            a.plot(v["km"], v["xte"], c=COL[nm][0], ls=COL[nm][1], lw=1.1, label=LBL[nm])
            print(f"  {day} Jan, r={r:<3} {LBL[nm]:<10} T = {v['T']:.4f} h  dT vs ERA5 = {(v['T'] - routes[('A', kd, 'era5', 0)]['T']) * 3600:+7.1f} s"
                  f"  max|XTE| = {np.abs(v['xte']).max():5.1f} km")
        a.axhline(0, c="k", lw=0.8, label="ERA5 route")
        a.set(title=f"{int(q * 100)}% ($r$={r})", xlabel="Along-route distance [km]"); a.title.set_fontsize(8)
    ax[0].set_ylabel("Cross-track error vs ERA5 [km]"); ax[0].legend(fontsize=5.5, loc="best")
    save(fig, OUT, f"route_xte_{day}Jan")

# ---- Fig 2: eps_T, eps_F, delta_route vs r (case A, mean over both dates) ----
fig, ax = plt.subplots(3, 1, figsize=(W2, 5.0), sharex=True, constrained_layout=True)
for a, (k, yl) in zip(ax, [("eps_T", "$\\varepsilon_T$ [%]"), ("eps_F", "$\\varepsilon_F$ [%]"), ("delta", "$\\delta_{route}$ [km]")]):
    for nm in LBL:
        a.plot(R, [AG[("A", nm, r)][k] for r in R], c=COL[nm][0], marker="o", ms=3, label=LBL[nm])
    a.set_ylabel(yl); a.set_ylim(bottom=0)
ax[0].legend(fontsize=5.5); ax[-1].set(xlabel="Reduced dimension $r$", xticks=R)
save(fig, OUT, "route_metrics_vs_r")

# ---- Video: time-synchronised aircraft positions + closeup of deviation ----
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle

kd, rm, NV, ZH = 0, R[1], 300, 0.5
SRCS = [("era5", "ERA5", "k"), ("dmd", "POD-DMD", COL["dmd"][0]), ("opinf", "POD-OpInf", COL["opinf"][0])]
full = {k: routes[("A", kd, k, 0 if k == "era5" else rm)]["P"] for k, _, _ in SRCS}
nwP = routes[("A", kd, "nowind", 0)]["P"]

# Restore the correct departure's TRUE ERA5 winds (Wt was overwritten by the main loop)
g = tst[DEP[kd]]
truth = X[:, g:g + HW + 1]
Wt = mid(truth)

# Directed edge lookup: reconstruct route edges from their grid-node coordinates
edge_lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ea, eb))}

def timed_profile(Pl):
    """Cumulative flight time and distance along a route, flown in true ERA5 winds."""
    nodes = [snap(p) for p in Pl]
    eids = [edge_lookup[(nodes[j], nodes[j + 1])] for j in range(len(nodes) - 1)]
    times, distances = [0.0], [0.0]
    t = dist = 0.0
    for ei in eids:
        w = wat(Wt, ei, t)
        vg = w @ Ee[ei] + np.sqrt(VA**2 - (w @ En[ei])**2)
        if not np.isfinite(vg) or vg <= 10:
            raise ValueError(f"Invalid ground speed on animation edge {ei}: {vg:.2f} m/s")
        dt = Dl[ei] / vg
        t += dt; dist += Dl[ei]
        times.append(t); distances.append(dist / 1000.0)
    return np.asarray(times), np.asarray(distances)

def position_at(Pl, times, t):
    """Interpolate aircraft position at elapsed flight time t."""
    if t >= times[-1]: return Pl[-1].copy()
    j = max(0, np.searchsorted(times, t, side="right") - 1)
    f = (t - times[j]) / (times[j + 1] - times[j])
    return Pl[j] + f * (Pl[j + 1] - Pl[j])

# Calculate each route's actual flight-time profile using the same true winds
profiles = {k: timed_profile(full[k]) for k, _, _ in SRCS}
TMAX = max(profiles[k][0][-1] for k, _, _ in SRCS)
t_anim = np.linspace(0.0, TMAX, NV)

# All aircraft are evaluated at the SAME elapsed flight time
trk = {k: np.array([position_at(full[k], profiles[k][0], t) for t in t_anim]) for k, _, _ in SRCS}

# ERA5 distance travelled at each common elapsed time
km = np.interp(t_anim, profiles["era5"][0], profiles["era5"][1])

# Cross-track error at matching elapsed times
xv = {k: xte(trk[k], full["era5"]) for k in ("dmd", "opinf")}
ylm = max(1.15 * max(np.abs(v).max() for v in xv.values()), 1.0)

# Wind magnitude background at departure
f0 = truth[:, 0]
u0, v0 = f0[:Ng].reshape(NLA, NLO), f0[Ng:].reshape(NLA, NLO)
if flip: u0, v0 = u0[::-1], v0[::-1]
spd = np.hypot(u0, v0); vmx = spd.max()

fig = plt.figure(figsize=(W2, 4.4), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1])
aO, aZ, aX = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
mO, mZ, dots = {}, {}, {}

for a in (aO, aZ):
    a.pcolormesh(lon, lat, spd, cmap="YlGn", shading="auto", vmin=0, vmax=vmx)
    a.plot(nwP[:, 1], nwP[:, 0], c="0.45", lw=2.2, label="Shortest (no wind)")
    for k, lb, c in SRCS:
        a.plot(full[k][:, 1], full[k][:, 0], c=c, lw=0.9 if a is aO else 1.2, label=lb)
    a.plot(*CHN[::-1], "ko", ms=3); a.plot(*SIN[::-1], "ks", ms=3)
    a.set(xlabel="Longitude [$^\\circ$E]", ylabel="Latitude [$^\\circ$N]")

for k, lb, c in SRCS:
    mO[k], = aO.plot([], [], "o", c=c, mec="w", mew=0.5, ms=4)
    mZ[k], = aZ.plot([], [], "o", c=c, mec="w", mew=0.5, ms=6)

aO.set(xlim=(78, 106), ylim=(0, 16), aspect="equal", title="Overview")
aZ.set(aspect="equal", title="Closeup")
aO.legend(fontsize=5, loc="upper right")
box = aO.add_patch(Rectangle((0, 0), 2 * ZH, 2 * ZH, fill=False, ec="r", lw=0.8))

for nm in ("dmd", "opinf"):
    aX.plot(km, xv[nm], c=COL[nm][0], ls=COL[nm][1], lw=1.0, label=LBL[nm])
    dots[nm], = aX.plot([], [], "o", c=COL[nm][0], ms=4)

aX.axhline(0, c="k", lw=0.8, label="ERA5 route")
cur = aX.axvline(0, c="r", lw=0.8)
aX.set(xlim=(0, km[-1]), ylim=(-ylm, ylm), xlabel="ERA5 along-route distance [km]", ylabel="Cross-track error [km]")
aX.legend(fontsize=5.5, loc="upper right")

for a in (aO, aZ, aX): a.title.set_fontsize(8)
txt = aZ.text(0.03, 0.04, "", transform=aZ.transAxes, fontsize=6, bbox=dict(fc="w", alpha=0.8, lw=0))

def upd(i):
    t = t_anim[i]
    y0, x0 = trk["era5"][i]
    box.set_xy((x0 - ZH, y0 - ZH))
    aZ.set_xlim(x0 - ZH, x0 + ZH); aZ.set_ylim(y0 - ZH, y0 + ZH)
    for k, _, _ in SRCS:
        p = trk[k][i]
        mO[k].set_data([p[1]], [p[0]])
        mZ[k].set_data([p[1]], [p[0]])
    cur.set_xdata([km[i], km[i]])
    for nm in dots:
        dots[nm].set_data([km[i]], [xv[nm][i]])
    txt.set_text(f"Elapsed: {t / 3600:.2f} h   ERA5 distance: {km[i]:.0f} km\n"
                 f"XTE: DMD {xv['dmd'][i]:+.1f} km, OpInf {xv['opinf'][i]:+.1f} km")
    return []

ani = FuncAnimation(fig, upd, frames=NV, blit=False)
try:
    ani.save(OUT / "route_deviation.mp4", writer=FFMpegWriter(fps=25, bitrate=2500), dpi=200)
except Exception as e:
    print("ffmpeg unavailable, writing gif:", e)
    ani.save(OUT / "route_deviation.gif", writer=PillowWriter(fps=20), dpi=120)

plt.close(fig)
print("Saved time-synchronised route animation.")