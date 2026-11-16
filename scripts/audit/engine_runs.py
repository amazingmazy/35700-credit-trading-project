"""Engine runs of the books that were audited (docs/results_audit.md, section 8).

Weights for decision date T are built from EOD data through T-1 (signals are lagged one day; dollar weights and
hedge ratios use durations through T-1) and handed to the scaffold engine as a lookup. The engine supplies the
3% volume cap, fills at the day-T close, $0.0035/share on every leg, corporate actions, GMV checks and PnL.

Usage: python scripts/audit/engine_runs.py
"""
from common import *  # noqa: F401,F403

SHOW = ["avg_gmv_mm", "ann_pnl_k", "ann_costs_k", "ann_vol_pct", "sharpe_project", "sharpe_net_fin", "t_net_fin", "net_fin_ex2020",
        "net_fin_ex_MarApr20", "maxdd_pct", "worst_day_k", "avg_net_expo_mm", "avg_dv01", "max_abs_dv01", "breach_days", "p90_unfilled"]
rows, yrs = {}, {}


def record(name, w_full, scale):
    rows[name], yrs[name], res = run_engine(name, w_full, scale)
    print("done:", name, flush=True)
    return res


# ---- loans vs HYG (seniority): position per leg from the implied-hazard gap; HYG takes the other side
dur_alt = dur.copy()
dur_alt[["BKLN", "SRLN"]] = 0.25  # loans as floating-rate: quarter-year duration
H_alt = anchor_dollars(dur_alt)
pos0 = {leg: pair_position(leg).fillna(0.0) for leg in ("BKLN", "SRLN")}
for legs in (("BKLN", "SRLN"), ("BKLN",), ("SRLN",)):
    for h in (1, 20, 30, 60, 90, 120):
        if len(legs) == 1 and h != 20:
            continue
        for dlabel, H_ in (("data duration", H), ("loan duration 0.25", H_alt)):
            w = pd.DataFrame({leg: (pos0[leg].rolling(h, min_periods=1).mean() if h > 1 else pos0[leg]) for leg in legs})
            w["HYG"] = -w.sum(axis=1)
            scale = 0.9 / (2.1 * len(legs) + (1.1 * len(legs) if dlabel.startswith("loan") else 0.2))
            record(f"{'+'.join(legs)} vs HYG, {h}d, {dlabel}", with_hedge(w, H_), scale)

# ---- HY hazard raw edge
edge = signals.hazard_edge_fit(p, hy, 1.0, 2).edges
w0, first = carried_weights(edge.shift(1), hy)
for h in (1, 60):
    w = held(w0, h)
    w.loc[w.index < first] = 0.0
    record(f"HY raw edge, {h}d", with_hedge(w, H), 0.9)

# ---- IG spread carry + roll-down per year of duration
fit_ig = signals.hazard_edge_fit(p, ig, 3.0, 3)
roll_ig = signals.rolldown_pct(fit_ig, ttm[ig], dur[ig], per_bp=True)
w0, first = carried_weights(((gs[ig] / 100 + roll_ig) / dur[ig]).shift(1), ig)


def ig_dollars(h):
    w = w0.rolling(h, min_periods=1).mean() if h > 1 else w0
    w = per_gross((w / dlag[ig]).fillna(0.0))
    w.loc[w.index < first] = 0.0
    return w


def ig_run(name, w):
    full = with_hedge(w, H)
    return record(name, full, 0.95 / full.abs().sum(axis=1).max())


for h in (1, 5, 20, 30, 60, 90):
    last = ig_run(f"IG carry + roll-down, {h}d", ig_dollars(h))
w1 = ig_dollars(1)
static = pd.DataFrame(np.tile(w1.loc[first:].mean().to_numpy(), (len(w1), 1)), index=w1.index, columns=ig)
static.loc[static.index < first] = 0.0
ig_run("IG carry + roll-down, fixed average weights", static)

out = pd.DataFrame(rows).T
print("\nEngine runs ($10MM capital, 3% cap, fills at the close)")
print(out[SHOW].round(2).to_string())
print("\nSharpe by year, net of financing")
print(pd.DataFrame(yrs).T.to_string())
out.to_csv(OUT / "engine_runs.csv")
pd.DataFrame(yrs).T.to_csv(OUT / "engine_runs_by_year.csv")

# ---- IG book (90-day run): what it holds, and fills against targets
inv = {v: k for k, v in sym.items()}
names = ig + anch
held_mm = last.positions.rename(columns=inv)[names] * p.close[names] / 1e6
tgt_mm = last.targets.reindex(columns=market.symbols).fillna(0.0).rename(columns=inv)[names] * p.close[names] / 1e6
cap_year = (0.03 * (p.volume * p.close)).groupby(p.close.index.year).median() / 1e6
wm = w1.loc[first:]
print("\nIG book: 1-day weights, average 90-day engine position ($MM), 3% cap ($MM a day)")
print(pd.DataFrame({"mean_w": wm.mean(), "sd_w": wm.std(), "share_long": (wm > 0).mean(), "median_price": p.close[ig].median(),
                    "avg_pos_mm": held_mm[ig].loc[first:].mean(), "cap_mm_2020": cap_year.loc[2020, ig],
                    "cap_mm_2026": cap_year.loc[2026, ig]}).sort_values("mean_w", ascending=False).round(3).to_string())
print("hedge legs, average $MM:", held_mm[anch].loc[first:].mean().round(1).to_dict())
d = last.daily
yr = held_mm.index.year
short = (held_mm - tgt_mm).abs().sum(axis=1) / tgt_mm.abs().sum(axis=1)
print("unfilled share of target gross, mean by year:", short.groupby(yr).mean().round(3).to_dict())
print("SCHJ and SPSB month-end target vs held ($MM), first 6 months:")
print(pd.DataFrame({"SCHJ_target": tgt_mm["SCHJ"], "SCHJ_held": held_mm["SCHJ"], "SPSB_target": tgt_mm["SPSB"],
                    "SPSB_held": held_mm["SPSB"]}).resample("ME").last().head(6).round(2).to_string())
print("eight worst days ($k):", {str(k.date()): round(v / 1e3) for k, v in d["pnl"].nsmallest(8).items()})
mm = d["pnl"][d.index.year == 2020]
print("2020 PnL by month ($k):", (mm.groupby(mm.index.month).sum() / 1e3).round(0).to_dict())
print("PnL by year ($k):", (d["pnl"].groupby(d.index.year).sum() / 1e3).round(0).to_dict())
