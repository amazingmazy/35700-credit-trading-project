"""Paper portfolios for every measured signal, by holding period (docs/results_audit.md, sections 4-6).

Before the 3% volume cap. Usage: python scripts/audit/paper_books.py
"""
from common import *  # noqa: F401,F403

HOLDS = (1, 5, 20, 60)
KEEP = ["ann_ret_pct", "vol_pct", "sharpe_net_fin", "sharpe_after_cost", "t_after_cost", "after_cost_ex2020",
        "after_cost_ex_MarApr20", "top10_share_of_pnl", "turnover_day", "cost_funds_pct", "cost_hedge_pct", "maxdd_pct", "worst_day_bp"]
rows, yearly, ics = {}, {}, {}


def book(label, sig, cols, grp=None, per_dur=False, ret_h=None, Hm=None, ic_target=None):
    """Rank book on `sig` over `cols`; `per_dur` turns duration-unit weights into dollars (w / duration)."""
    cols = [c for c in cols if c in sig.columns]
    w0, first = carried_weights(sig, cols, grp)
    for h in HOLDS:
        w = w0.rolling(h, min_periods=1).mean() if h > 1 else w0
        if per_dur:
            w = (w / dlag[cols]).fillna(0.0)
        m, d = evaluate(per_gross(w), xr if ret_h is None else ret_h, H if Hm is None else Hm, first)
        rows[(label, len(cols), h)] = {k: m[k] for k in KEEP}
        if h in (1, 20):
            yearly[(label, h)] = by_year(d)
    tgt = (xr if ic_target is None else ic_target)[cols]
    g = None if grp is None else grp[cols]
    a = stats.ic_table(sig[cols], tgt, (1, 5, 20), g)
    b = stats.ic_table(sig[cols].ffill(limit=CARRY_DAYS), tgt, (1, 5, 20), g)
    ics[label] = {f"{h}d {v}": f"{t.loc[h, 'mean_ic']:+.3f} ({t.loc[h, 't_stat']:+.1f}) n={int(t.loc[h, 'n_days'])}"
                  for h in (1, 5, 20) for v, t in (("as measured", a), ("carried", b))}


# NAV basis (lecture 2)
sig_nav, sig_nav3 = signals.nav_discount_lagged(p), signals.nav_discount_3pm(p)
book("NAV discount T-1 close", sig_nav, credit, groups)
book("NAV discount T-1 close (liquid)", sig_nav, liquid, groups)
book("NAV discount 3PM", sig_nav3, credit, groups)
book("NAV discount 3PM (liquid)", sig_nav3, liquid, groups)

# 3PM factor residual: the target is the residual itself, so the reference-ETF and Treasury legs are not costed
beta = signals.reference_beta(p)
sig_res = signals.residual_3pm_reversal(p)
resid_cc = signals.factor_residual(p, p.ret, beta)
res_cols = list(sig_res.columns)
book("3PM residual reversal", sig_res, res_cols, groups, ret_h=resid_cc, Hm=H * 0.0, ic_target=resid_cc)
book("3PM residual reversal (liquid)", sig_res, [c for c in res_cols if c in liquid], groups, ret_h=resid_cc, Hm=H * 0.0, ic_target=resid_cc)

# hazard-curve edges (lectures 4-5)
fit_ig, fit_hy = signals.hazard_edge_fit(p, ig, 3.0, 3), signals.hazard_edge_fit(p, hy, 1.0, 2)
book("Hazard edge deviation z, IG", signals.edge_deviation_z(fit_ig.edges), ig)
book("Hazard edge deviation z, HY", signals.edge_deviation_z(fit_hy.edges), hy)
book("Hazard raw edge, IG", fit_ig.edges.shift(1), ig)
book("Hazard raw edge, HY", fit_hy.edges.shift(1), hy)

# roll-down (lecture 1): signals per year of duration
roll_ig = signals.rolldown_pct(fit_ig, ttm[ig], dur[ig], per_bp=True)
book("IG spread carry + roll-down", ((gs[ig] / 100 + roll_ig) / dur[ig]).shift(1), ig, per_dur=True, ic_target=xr[ig] / dlag[ig])
book("IG spread roll-down only", (roll_ig / dur[ig]).shift(1), ig, per_dur=True, ic_target=xr[ig] / dlag[ig])
fit_ust = hazard.fit_curve_panel(ylds[ust], ttm[ust], None, lam=2.0, n_terms=3)
roll_ust = signals.rolldown_pct(fit_ust, ttm[ust], dur[ust])
curve_ust = [t for t in ust if dur[t].median() >= 1.0]
book("Rates carry + roll-down", ((ylds[ust].sub(bill * 100, axis=0) + roll_ust) / dur[ust]).shift(1), curve_ust,
     per_dur=True, ret_h=p.ret, Hm=H * 0.0, ic_target=p.ret[ust].sub(bill / 252, axis=0) / dlag[ust])

# Merton tier model (lecture 6) and the naive benchmark; rank weights net to zero, so HYG drops out
tier, sig_lev, sig_naive = merton_tier_signals()
rel = xr[tier].sub(xr["HYG"], axis=0)
book("Merton leverage edge vs HYG", sig_lev, tier, ic_target=rel)
book("Naive spread-gap edge vs HYG", sig_naive, tier, ic_target=rel)

out = pd.DataFrame(rows).T
out.index.names = ["signal", "n_etfs", "hold_d"]
print("\nCross-sectional paper books")
print(out.round(2).to_string())
out.to_csv(OUT / "paper_books.csv")
print("\nSharpe by year after costs")
print(pd.DataFrame(yearly).T.to_string())
ic = pd.DataFrame(ics).T
print("\nIC (overlap-adjusted t) and days: signal as measured vs carried forward")
print(ic.to_string())
ic.to_csv(OUT / "ic_as_measured_vs_carried.csv")

# pairs against HYG (lecture 4 seniority for the two loan funds; same-rank funds for comparison)
dur_alt = dur.copy()
dur_alt[["BKLN", "SRLN"]] = 0.25  # loans as floating-rate: quarter-year duration
xr_alt, H_alt = hedged_return(p.ret, dur_alt), anchor_dollars(dur_alt)
PKEEP = ["ann_ret_pct", "vol_pct", "sharpe_net_fin", "sharpe_after_cost", "t_after_cost", "after_cost_ex2020",
         "after_cost_ex_MarApr20", "top10_share_of_pnl", "turnover_day", "maxdd_pct", "worst_day_bp"]
prow, pyr = {}, {}
for leg in ["BKLN", "SRLN", "SPHY", "USHY", "JNK", "SHYG", "SJNK"]:
    pos0 = pair_position(leg)
    first_p = pos0.dropna().index.min()
    pos0 = pos0.fillna(0.0)
    for label, r_, H_ in (("data duration", xr, H), ("loan duration 0.25", xr_alt, H_alt)):
        if label.startswith("loan") and leg not in ("BKLN", "SRLN"):
            continue
        for h in HOLDS:
            pos = pos0.rolling(h, min_periods=1).mean() if h > 1 else pos0
            m, d = evaluate(pd.DataFrame({leg: pos, "HYG": -pos}), r_, H_, first_p)
            prow[(f"{leg} vs HYG", label, h)] = {k: m[k] for k in PKEEP}
            if h in (1, 20):
                pyr[(f"{leg} vs HYG", label, h)] = by_year(d)
pairs = pd.DataFrame(prow).T
print("\nPairs against HYG")
print(pairs.round(2).to_string())
print(pd.DataFrame(pyr).T.to_string())
pairs.to_csv(OUT / "pairs.csv")
