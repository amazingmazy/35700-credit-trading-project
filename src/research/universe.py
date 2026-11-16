"""Bucket labels for the ETFs.

The symbology file has no category field, so the buckets below were assigned by hand from the fund names
(and checked against median duration and spread). Tickers not listed are labelled "Other".
"""
from __future__ import annotations

import pandas as pd

_GROUPS: dict[str, str] = {
    "UST": ("BIL GBIL SGOV SHV TBLL TFLO USFR SHY SCHO SPTS VGSH IEI SCHR SPTI VGIT IEF TLH TLT SCHQ SPTL VGLT "
            "EDV ZROZ GOVT GOVI IBTG IBTH IBTI IBTJ IBTK"),
    "IG": ("IGSB SLQD SPSB VCSH SCHJ QSIG SUSB BSCQ BSCR IBDR IBDS IGIB SPIB VCIT SCHI BSCS BSCT IBDT IBDU IBDV "
           "IGLB SPLB VCLT LKOR LQD USIG SPBO VTC CORP FCOR FLCO GIGB IGEB PFIG QIG QLTA LQDB SKOR SUSC BBCB"),
    "FRN": "FLOT FLRN FLTR VRIG",
    "HY": ("HYG JNK USHY SPHY HYLB SHYG SJNK HYS SHYL ANGL FALN BBHY BSJQ FLHY GHYB HYDB HYDW HYUP HYGV HYXF "
           "PHB PHYL QHY"),
    "HY-Global": "GHYG PGHY IHY EUHY",
    "Loan": "BKLN SRLN",
    "Pref": "FPEI PREF PSK",
}

IG_MATURITY: dict[str, str] = {
    "short": "IGSB SLQD SPSB VCSH SCHJ QSIG SUSB BSCQ BSCR IBDR IBDS",
    "intermediate": "IGIB SPIB VCIT SCHI BSCS BSCT IBDT IBDU IBDV",
    "long": "IGLB SPLB VCLT LKOR",
    "broad": "LQD USIG SPBO VTC CORP FCOR FLCO GIGB IGEB PFIG QIG QLTA LQDB SKOR SUSC BBCB",
}

BUCKET_OF: dict[str, str] = {t: b for b, s in _GROUPS.items() for t in s.split()}
CREDIT_BUCKETS = ("IG", "FRN", "HY", "HY-Global", "Loan", "Pref")
# Treasury ETFs used to build the duration-matched rates hedge (all have complete dividend records).
HEDGE_ANCHORS = ("SHY", "IEI", "IEF", "TLH", "TLT")
# Reference ETF per bucket for the 3PM factor residual.
CREDIT_REFERENCE = {"IG": "LQD", "FRN": "LQD", "HY": "HYG", "HY-Global": "HYG", "Loan": "HYG", "Pref": "HYG"}
# Recovery assumption by seniority (lecture 4): loans ~80%, senior unsecured ~40%, preferreds ~0%.
RECOVERY = {"Loan": 0.80, "HY": 0.40, "HY-Global": 0.40, "IG": 0.40, "FRN": 0.40, "Pref": 0.0}


def bucket_of(tickers) -> pd.Series:
    """Series ticker -> bucket ("Other" when the ticker is not classified)."""
    idx = pd.Index(tickers)
    return pd.Series([BUCKET_OF.get(t, "Other") for t in idx], index=idx)


def in_bucket(tickers, *buckets: str) -> list[str]:
    """The tickers that belong to any of `buckets`, in their original order."""
    return [t for t in tickers if BUCKET_OF.get(t, "Other") in buckets]
