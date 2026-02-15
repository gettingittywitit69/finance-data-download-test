"""
Extended Fama-French data loaders that return tidy long-format DataFrames.

Each loader targets a specific family of strategies and returns a DataFrame with
columns:
    - date (Timestamp, month-end)
    - group (str)
    - strategy_id (str)
    - return_excess (float, decimal monthly excess return)

The functions rely on pandas_datareader's Fama-French interface and reuse the
U.S. risk-free rate when appropriate to compute excess returns.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import pandas as pd

from .datasets import DEFAULT_START, KenFrenchLoader, fetch_french49_excess

__all__ = [
    "load_all_strategies_long",
    "load_us_ff5_factors",
    "load_us_industries_30",
    "load_us_industries_49",
    "load_us_momentum_factor",
    "load_us_research_factors_wide",
    "load_us_size_bm_25",
    "load_us_size_deciles",
    "pivot_family",
]

_FF_LOADER = KenFrenchLoader()


def _fetch_ff_table(
    dataset: str,
    table: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    loader: Optional[KenFrenchLoader] = None,
) -> pd.DataFrame:
    """Download a single Fama-French table and clean obvious sentinels."""
    loader = loader or _FF_LOADER
    return loader.load_table(dataset, table=table, start_date=start_date, end_date=end_date)


def _to_long(df: pd.DataFrame, group: str, rename_map: Optional[dict] = None) -> pd.DataFrame:
    """Convert a wide DataFrame to long with standardized column names."""
    rename_map = rename_map or {}
    tidy_df = df.copy()
    tidy_df.index.name = "date"
    tidy = tidy_df.rename(columns=rename_map).reset_index()
    tidy = tidy.rename(columns={"Date": "date", "index": "date"})
    tidy = tidy.melt(id_vars="date", var_name="strategy_id", value_name="return_excess")
    tidy["group"] = group
    tidy = tidy.dropna(subset=["return_excess"]).sort_values(["date", "group", "strategy_id"])
    tidy = tidy[["date", "group", "strategy_id", "return_excess"]].reset_index(drop=True)
    return tidy


def load_us_ff5_factors(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load U.S. Fama-French 5 factors (monthly) and return tidy factors + RF.

    Factors are already excess returns; values are converted to decimals.
    """
    df = _fetch_ff_table("F-F_Research_Data_5_Factors_2x3", table=0, start_date=start_date, end_date=end_date)
    rf = (df["RF"] / 100.0).rename("RF")
    factors = df.drop(columns="RF") / 100.0
    rename_map = {c: c.replace("-", "_") for c in factors.columns}
    long = _to_long(factors, group="US_factors_5", rename_map=rename_map)
    return long, rf


def load_us_research_factors_wide(
    start_date: Optional[str] = DEFAULT_START,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load U.S. research factors (Mkt-RF, SMB, HML) plus RF in wide monthly format.

    Values are returned in decimal form with a month-end DatetimeIndex.
    """
    df = _fetch_ff_table("F-F_Research_Data_Factors", table=0, start_date=start_date, end_date=end_date)
    df = df / 100.0
    rf = df["RF"].rename("RF")
    factors = df.drop(columns="RF")
    return factors, rf


def load_us_momentum_factor(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the U.S. momentum factor (UMD) and return a tidy excess series.

    Series are already excess returns; values are converted to decimals.
    """
    df = _fetch_ff_table("F-F_Momentum_Factor", table=0, start_date=start_date, end_date=end_date)
    df = df / 100.0
    rename_map = {df.columns[0]: "UMD"}
    return _to_long(df, group="US_momentum_factor", rename_map=rename_map)


def load_us_size_deciles(
    rf: pd.Series,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load value-weighted 10 size portfolios (monthly) and convert to excess returns.
    """
    df = _fetch_ff_table("Portfolios_Formed_on_ME", table=0, start_date=start_date, end_date=end_date)
    dec_cols = ["Lo 10", "2-Dec", "3-Dec", "4-Dec", "5-Dec", "6-Dec", "7-Dec", "8-Dec", "9-Dec", "Hi 10"]
    dec_df = df[dec_cols] / 100.0
    rf_aligned = rf.reindex(dec_df.index)
    excess = dec_df.sub(rf_aligned, axis=0)
    rename_map = {col: f"SIZE_{i+1}" for i, col in enumerate(dec_cols)}
    return _to_long(excess, group="US_size_10", rename_map=rename_map)


def load_us_size_bm_25(
    rf: pd.Series,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load value-weighted 25 portfolios formed on Size and Book-to-Market and convert to excess.
    """
    df = _fetch_ff_table("25_Portfolios_5x5", table=0, start_date=start_date, end_date=end_date)
    ret_df = df / 100.0
    rf_aligned = rf.reindex(ret_df.index)
    excess = ret_df.sub(rf_aligned, axis=0)
    rename_map = {col: col.replace(" ", "_").replace("-", "_") for col in excess.columns}
    return _to_long(excess, group="US_size_BM_25", rename_map=rename_map)


def load_us_industries_30(
    rf: pd.Series,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load 30 industry portfolios (value-weighted) and convert to excess returns."""
    df = _fetch_ff_table("30_Industry_Portfolios", table=0, start_date=start_date, end_date=end_date)
    df = df / 100.0
    rf_aligned = rf.reindex(df.index)
    excess = df.sub(rf_aligned, axis=0)
    return _to_long(excess, group="US_industries_30")


def load_us_industries_49(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Reuse the existing helper for 49 industries and return tidy excess returns."""
    excess, _ = fetch_french49_excess(start=start_date, end=end_date, value_weighted=True)
    excess = excess.copy()
    return _to_long(excess, group="US_industries_49")


def load_all_strategies_long(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and combine all supported strategy families into a single long DataFrame.

    Families included by default:
      - US 49 Industry portfolios
      - US 30 Industry portfolios
      - US FF5 factors + UMD momentum factor
      - US size deciles (10)
      - US size-BM 25 portfolios
    """
    us_ff5_long, rf_us = load_us_ff5_factors(start_date=start_date, end_date=end_date)
    us_mom_long = load_us_momentum_factor(start_date=start_date, end_date=end_date)
    us_size10_long = load_us_size_deciles(rf=rf_us, start_date=start_date, end_date=end_date)
    us_sizebm_long = load_us_size_bm_25(rf=rf_us, start_date=start_date, end_date=end_date)
    us_ind30_long = load_us_industries_30(rf=rf_us, start_date=start_date, end_date=end_date)
    us_ind49_long = load_us_industries_49(start_date=start_date, end_date=end_date)

    frames = [
        us_ind49_long,
        us_ind30_long,
        us_ff5_long,
        us_mom_long,
        us_size10_long,
        us_sizebm_long,
    ]

    frames = [f for f in frames if f is not None and not f.empty]
    combined = pd.concat(frames, ignore_index=True, sort=False)

    if start_date is not None:
        combined = combined[combined["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        combined = combined[combined["date"] <= pd.to_datetime(end_date)]

    combined = combined.sort_values(["date", "group", "strategy_id"]).reset_index(drop=True)
    return combined


def pivot_family(long_df: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    """
    Convenience helper to pivot one or more groups into a wide date x strategy table.
    """
    fam = long_df[long_df["group"].isin(groups)].copy()
    wide = fam.pivot(index="date", columns="strategy_id", values="return_excess")
    wide = wide.sort_index()
    return wide.dropna(how="all")
