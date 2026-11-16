"""Audit of the HY hazard raw-edge paper book (docs/results_audit.md, section 7).

Usage: python scripts/audit/hy_edge_audit.py
"""
from common import *  # noqa: F401,F403

edge = signals.hazard_edge_fit(p, hy, 1.0, 2).edges
sig = edge.shift(1)
start = sig.dropna(how="all").index.min()
print("HY funds:", hy)

# 1. missing signals
miss = sig.loc[start:].isna().sum(axis=1)
print("\n[1] fund-days with no signal by year:", miss.groupby(miss.index.year).sum().to_dict(),
      "| days with fewer than 5 of 6:", int((sig.notna().sum(axis=1) < 5).loc[start:].sum()))
print("    2020 by month:", miss[miss.index.year == 2020].groupby(miss[miss.index.year == 2020].index.month).sum().to_dict())

# 2. three treatments of a missing signal, by holding period
r = sig.rank(axis=1)
r = r.sub(r.mean(axis=1), axis=0)
w_nan = r.div(r.abs().sum(axis=1).where(sig.notna().sum(axis=1) >= 5), axis=0)   # NaN where the fund has no signal
treatments = {
    "A. signal carried forward, all funds re-ranked": carried_weights(sig, hy)[0],
    "B. no position in a fund without a signal": w_nan.fillna(0.0),
    "C. previous weight kept for that fund only (flawed)": per_gross(w_nan.ffill(limit=CARRY_DAYS).fillna(0.0)),
}
rows, yrs = {}, {}
for name, w0 in treatments.items():
    for h in (1, 5, 20, 60, 120):
        m, d = evaluate(held(w0, h), xr, H, start)
        rows[(name, h)] = m
        if name.startswith("A"):
            yrs[h] = by_year(d)
out = pd.DataFrame(rows).T
print("\n[2] HY raw edge by holding period (per $1 of fund gross)")
print(out.round(2).to_string())
print("    treatment A, Sharpe by year after costs:")
print(pd.DataFrame(yrs).T.to_string())
out.to_csv(OUT / "hy_edge_treatments.csv")

# 3. what the book holds
w0, first = carried_weights(sig, hy)
w1 = held(w0, 1)
print("\n[3] weights by fund (treatment A, 1-day book)")
print(pd.DataFrame({"mean_w": w1.loc[first:].mean(), "sd_w": w1.loc[first:].std(), "share_long": (w1.loc[first:] > 0).mean(),
                    "mean_edge_bp": edge[hy].mean(), "median_price": p.close[hy].median()}).round(3).to_string())
m_all, d_all = evaluate(w1, xr, H, first)
static = pd.DataFrame(np.tile(w1.loc[first:].mean().to_numpy(), (len(sig), 1)), index=sig.index, columns=hy)
m_static, d_static = evaluate(static, xr, H, first)
m_dyn, d_dyn = evaluate(w1 - static, xr, H, first)
show = ("ann_ret_pct", "sharpe_project", "after_cost_ex2020")
print("    fixed average weights (full-sample mean, diagnostic only):", {k: round(m_static[k], 2) for k in show}, by_year(d_static, "pnl"))
print("    deviations from the average weights:", {k: round(m_dyn[k], 2) for k in show}, by_year(d_dyn, "pnl"))
pick = (w1.shift(1) * gs[hy].shift(1)).sum(axis=1).loc[first:] / 100
print("    spread pickup, % a year:", round(pick.mean(), 2), "| realised return, % a year:", round(m_all["ann_ret_pct"], 2))
mm = d_all.pnl[d_all.index.year == 2020]
print("    2020 PnL by month, %:", (mm.groupby(mm.index.month).sum() * 100).round(2).to_dict())
print("    ten largest |PnL| days, bp:", {str(k.date()): round(v * 1e4) for k, v in d_all.pnl[d_all.pnl.abs().nlargest(10).index].items()})

# 4. slow part of the edge vs its deviation (both known at T)
trail = edge.rolling(250, min_periods=120).mean()
rows = {}
for name, s in {"trailing 250d mean of edge": trail.shift(1), "edge minus its trailing 250d mean": (edge - trail).shift(1)}.items():
    w0s, f = carried_weights(s, hy)
    for h in (1, 20):
        m, _ = evaluate(held(w0s, h), xr, H, f)
        rows[(name, h)] = {k: m[k] for k in ("ann_ret_pct", "sharpe_net_fin", "sharpe_after_cost", "after_cost_ex2020", "turnover_day")}
print("\n[4] slow part vs deviation")
print(pd.DataFrame(rows).T.round(2).to_string())

# 5. timing checks
m2, _ = evaluate(w1, p.ret - rates_return(p.ret, dur.shift(1)), H, first)   # hedge ratio one more day stale
m3, _ = evaluate(w1.shift(1).fillna(0.0), xr, H, first)                      # trade one day later
m4, _ = evaluate(w1, p.ret, H * 0.0, first)                                  # no rates hedge
print("\n[5] 1-day book, Sharpe net of financing:", round(m_all["sharpe_net_fin"], 2),
      "| hedge duration lagged one more day:", round(m2["sharpe_net_fin"], 2),
      "| entry one day later:", round(m3["sharpe_net_fin"], 2), "| no rates hedge:", round(m4["sharpe_project"], 2))
ll = {t: {f"k={k:+d}": gs[t].diff().corr(xr[t].shift(-k)) for k in (-2, -1, 0, 1, 2)} for t in hy + ["BKLN", "SRLN"]}
print("    correlation of the day-t g-spread change with the hedged return on day t+k:")
print(pd.DataFrame(ll).T.round(3).to_string())

# 6. commission per dollar traded and the 3% cap
names = hy + ["BKLN", "SRLN", "LQD", "VCSH", "SPSB", "IGSB", "FLOT"] + anch
print("\n[6] commission per $ traded, bp (0.0035 / median close):")
print((COST / p.close.median()[names] * 1e4).round(2).to_string())
cap = (0.03 * (p.volume * p.close)).groupby(p.close.index.year).median() / 1e6
print("    3% cap, $MM a day, median by year:")
print(cap[hy + ["BKLN", "SRLN"]].round(2).to_string())
