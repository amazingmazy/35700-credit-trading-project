"""Standard report charts: cumulative net PnL, drawdown, gross exposure and DV01."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.performance import drawdowns  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from src.backtester import BacktestResult


def plot_result(result: "BacktestResult", path: str | None = None):
    d, c = result.daily, result.config
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(d.index, d["pnl"].cumsum() / 1e6)
    axes[0].set_ylabel("Cum. net PnL ($MM)")
    axes[0].set_title(f"{result.strategy_name}")
    dd_usd, _ = drawdowns(d["pnl"], c.initial_capital)
    axes[1].fill_between(d.index, -dd_usd / 1e6, 0, alpha=0.5)
    axes[1].set_ylabel("Drawdown ($MM)")
    axes[2].plot(d.index, d["gmv"] / 1e6)
    axes[2].axhline(c.gmv_cap / 1e6, ls="--", lw=1)
    axes[2].set_ylabel("GMV ($MM)")
    axes[3].plot(d.index, d["dv01"] / 1e3)
    axes[3].set_ylabel("DV01 ($K / -1bp)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
        plt.close(fig)
    return fig
