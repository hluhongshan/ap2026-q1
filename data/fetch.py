#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import signal
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CFFEX_DIR = RAW / "cffex"
Q3_DIR = RAW / "q3"
CACHE = ROOT / ".cache"
METADATA = ROOT / "data" / "metadata.json"
MACRO_PANEL = CLEAN / "macro_panel.csv"
MACRO_META = CLEAN / "macro_meta.csv"
SHIBOR_CLEAN = CLEAN / "shibor.csv"

SAMPLE_MONTHS = [
    "2024-03",
    "2024-09",
    "2025-03",
    "2025-09",
    "2026-03",
    "2026-09",
]
CFFEX_URL = "http://www.cffex.com.cn/sj/hqsj/rtj/{ym}/{dd}/{ymd}_1.csv"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
OPTION_COLUMNS = [
    "date",
    "contract",
    "expiry",
    "cp",
    "K",
    "open",
    "high",
    "low",
    "close",
    "settle",
    "volume",
    "oi",
    "S",
]
SHIBOR_COLUMNS = ["date", "ON", "1W", "2W", "1M", "3M", "6M", "9M", "1Y"]
SINA_COLUMNS = [
    "date",
    "contract",
    "sina_symbol",
    "cffex_close",
    "sina_close",
    "diff",
    "status",
    "error",
]
FUTURES_SYMBOLS = {"IF": "IF0", "RB": "RB0", "CU": "CU0", "SC": "SC0"}
FUTURES_COLUMNS = ["date", "open", "high", "low", "close", "volume", "oi"]

log = logging.getLogger("q1-fetch")
console = Console()
INTERFACES: list[dict] = []


class CallTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise CallTimeout("call exceeded its wall-clock limit")


@contextmanager
def time_limit(seconds: int):
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def record(name: str, status: str, detail: str = "", path: str | None = None):
    INTERFACES.append(
        {
            "interface": name,
            "status": status,
            "detail": detail,
            "path": path,
            "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weekdays_in_month(month: str):
    year, mon = map(int, month.split("-"))
    current = date(year, mon, 1)
    while current.month == mon:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def third_friday(year: int, month: int) -> date:
    current = date(year, month, 1)
    fridays = []
    while current.month == month:
        if current.weekday() == 4:
            fridays.append(current)
        current += timedelta(days=1)
    return fridays[2]


def fetch_cffex(session: requests.Session):
    CFFEX_DIR.mkdir(parents=True, exist_ok=True)
    ok = skipped = non200 = noncsv = errors = 0
    failed_dates: list[dict] = []
    for month in SAMPLE_MONTHS:
        for day in weekdays_in_month(month):
            path = CFFEX_DIR / f"{day:%Y%m%d}_1.csv"
            if path.exists() and path.stat().st_size > 0:
                existing = path.read_text(encoding="utf-8", errors="replace")
                skipped += 1
                if "合约代码" not in existing:
                    noncsv += 1
                    failed_dates.append({"date": day.isoformat(), "status": "non_csv_200"})
                    log.info("CFFEX %s cached response is a non-CSV holiday page; not refetching", day)
                continue
            url = CFFEX_URL.format(ym=f"{day:%Y%m}", dd=f"{day:%d}", ymd=f"{day:%Y%m%d}")
            try:
                response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=(10, 30))
                if response.status_code != 200:
                    non200 += 1
                    failed_dates.append({"date": day.isoformat(), "status": response.status_code})
                    log.warning("CFFEX %s returned %s (probable holiday)", day, response.status_code)
                else:
                    text = response.content.decode("gbk", errors="replace")
                    if "合约代码" not in text:
                        noncsv += 1
                        failed_dates.append({"date": day.isoformat(), "status": "non_csv_200"})
                        log.warning("CFFEX %s returned a non-CSV page with 200 (probable holiday)", day)
                    else:
                        path.write_text(text, encoding="utf-8")
                        ok += 1
            except Exception as exc:
                errors += 1
                failed_dates.append(
                    {"date": day.isoformat(), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                )
                log.error("CFFEX %s failed: %s: %s", day, type(exc).__name__, exc)
            time.sleep(0.3)
    detail = f"downloaded={ok} skipped={skipped} non200={non200} noncsv200={noncsv} errors={errors}"
    status = "ok" if errors == 0 else "partial"
    record("CFFEX daily quotes", status, detail, str(CFFEX_DIR))
    if failed_dates:
        log.info("CFFEX non-200/error dates: %s", failed_dates)


def read_table(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, dtype=str)


def fetch_csi300():
    path = RAW / "csi300.csv"
    source_path = RAW / "csi300_source.txt"
    if path.exists() and path.stat().st_size > 0:
        source = source_path.read_text(encoding="utf-8").strip() if source_path.exists() else "cached"
        record("CSI 300 close", "skipped", f"using cached source={source}", str(path))
        log.info("CSI 300 close skipped; cached source=%s", source)
        return
    import akshare as ak

    attempts = [
        (
            "akshare.index_zh_a_hist",
            lambda: ak.index_zh_a_hist(
                symbol="000300",
                period="daily",
                start_date="20240101",
                end_date="20261231",
            ),
            "日期",
            "收盘",
        ),
        (
            "akshare.stock_zh_index_daily",
            lambda: ak.stock_zh_index_daily(symbol="sh000300"),
            "date",
            "close",
        ),
    ]
    failures = []
    for name, call, date_col, close_col in attempts:
        try:
            with time_limit(180):
                frame = call()
            frame = frame[[date_col, close_col]].copy()
            frame.columns = ["date", "close"]
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame = frame.dropna().drop_duplicates("date", keep="last").sort_values("date")
            if frame.empty:
                raise ValueError("empty response")
            frame.to_csv(path, index=False, encoding="utf-8")
            source_path.write_text(name, encoding="utf-8")
            record("CSI 300 close", "ok", f"source={name} rows={len(frame)}", str(path))
            log.info("CSI 300 close source used: %s (%d rows)", name, len(frame))
            return
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(f"{name}: {message}")
            log.error("CSI 300 %s failed: %s", name, message)
    record("CSI 300 close", "failed", " | ".join(failures), str(path))


def fetch_ak_csv(name: str, path: Path, call, attempts: int = 1, delay: float = 0.0, **kwargs):
    if path.exists() and path.stat().st_size > 0:
        record(name, "skipped", f"cached {path.name}", str(path))
        log.info("%s skipped; cached", name)
        return
    failures = []
    for attempt in range(1, attempts + 1):
        try:
            with time_limit(180):
                frame = call(**kwargs)
            if frame is None or len(frame) == 0:
                raise ValueError("empty response")
            frame.to_csv(path, index=False, encoding="utf-8")
            record(name, "ok", f"rows={len(frame)}", str(path))
            log.info("%s ok (%d rows)", name, len(frame))
            return
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(f"attempt {attempt}: {message}")
            log.error("%s attempt %d failed: %s", name, attempt, message)
            if attempt < attempts and delay:
                time.sleep(delay)
    record(name, "failed", " | ".join(failures), str(path))


def fetch_etf(code: str):
    import akshare as ak

    path = RAW / f"etf_{code}.csv"
    source_path = RAW / f"etf_{code}_source.txt"
    if path.exists() and path.stat().st_size > 0:
        source = source_path.read_text(encoding="utf-8").strip() if source_path.exists() else "cached"
        record(f"ETF {code}", "skipped", f"using cached source={source}", str(path))
        log.info("ETF %s skipped; cached source=%s", code, source)
        return
    attempts = [
        (
            "akshare.fund_etf_hist_em",
            lambda: ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date="20240101",
                end_date="20261231",
                adjust="",
            ),
        ),
        ("akshare.fund_etf_hist_sina", lambda: ak.fund_etf_hist_sina(symbol=f"sh{code}")),
    ]
    failures = []
    for name, call in attempts:
        try:
            with time_limit(120):
                frame = call()
            if frame is None or len(frame) == 0:
                raise ValueError("empty response")
            frame.to_csv(path, index=False, encoding="utf-8")
            source_path.write_text(name, encoding="utf-8")
            record(f"ETF {code}", "ok", f"source={name} rows={len(frame)}", str(path))
            log.info("ETF %s source used: %s (%d rows)", code, name, len(frame))
            return
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(f"{name}: {message}")
            log.error("ETF %s %s failed: %s", code, name, message)
    record(f"ETF {code}", "failed", " | ".join(failures), str(path))


Q3_INTERFACES = [
    "macro_china_pmi",
    "macro_china_non_man_pmi",
    "macro_china_cx_pmi_yearly",
    "macro_china_cx_services_pmi_yearly",
    "macro_china_cpi",
    "macro_china_cpi_yearly",
    "macro_china_ppi",
    "macro_china_ppi_yearly",
    "macro_china_money_supply",
    "macro_china_shrzgm",
    "macro_china_new_financial_credit",
    "macro_china_lpr",
    "macro_china_consumer_goods_retail",
    "macro_china_industrial_production_yoy",
    "macro_china_gyzjz",
    "macro_china_gdzctz",
    "macro_china_exports_yoy",
    "macro_china_imports_yoy",
    "macro_china_trade_balance",
    "macro_china_hgjck",
    "macro_china_xfzxx",
    "macro_china_society_electricity",
    "macro_china_gdp",
    "macro_china_gdp_yearly",
    "macro_china_real_estate",
    "macro_china_enterprise_boom_index",
    "macro_china_fx_gold",
    "macro_china_urban_unemployment",
    "macro_china_czsr",
    "macro_china_qyspjg",
    "macro_china_stock_market_cap",
    "macro_china_commodity_price_index",
    "macro_china_society_traffic_volume",
    "macro_china_central_bank_balance",
    "macro_china_bank_financing",
    "macro_china_retail_price_index",
    "macro_china_market_margin_sh",
]
Q3_BOND_INTERFACE = "bond_zh_us_rate"
Q3_PROFIT_INTERFACE = "macro_china_nbs_nation"


def configure_q3_logging():
    CACHE.mkdir(parents=True, exist_ok=True)
    log_path = CACHE / "q3_fetch.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    log.info("Q3 log file: %s", log_path)
    return log_path


def fetch_q3_raw():
    import akshare as ak

    Q3_DIR.mkdir(parents=True, exist_ok=True)
    log_path = configure_q3_logging()
    for name in Q3_INTERFACES:
        path = Q3_DIR / f"{name}.csv"
        if not hasattr(ak, name):
            record(name, "missing", "akshare interface does not exist", str(path))
            log.warning("%s missing from akshare; skipped", name)
            continue
        fetch_ak_csv(name, path, getattr(ak, name), attempts=3, delay=3.0)

    bond_path = Q3_DIR / f"{Q3_BOND_INTERFACE}.csv"
    if hasattr(ak, Q3_BOND_INTERFACE):
        fetch_ak_csv(
            Q3_BOND_INTERFACE,
            bond_path,
            ak.bond_zh_us_rate,
            attempts=3,
            delay=3.0,
            start_date="20100101",
        )
    else:
        record(Q3_BOND_INTERFACE, "missing", "akshare interface does not exist", str(bond_path))
        log.warning("%s missing from akshare; skipped", Q3_BOND_INTERFACE)

    profit_path = Q3_DIR / "macro_china_nbs_industrial_profit.csv"
    if hasattr(ak, Q3_PROFIT_INTERFACE):
        if profit_path.exists() and profit_path.stat().st_size > 0:
            record(
                f"{Q3_PROFIT_INTERFACE}:industrial_profit",
                "skipped",
                f"cached {profit_path.name}",
                str(profit_path),
            )
            log.info("industrial profit skipped; cached")
        else:
            try:
                with time_limit(180):
                    frame = ak.macro_china_nbs_nation(
                        kind="月度数据",
                        path="工业 > 工业企业主要经济指标",
                        period="2000-",
                    )
                if frame is None or len(frame) == 0:
                    raise ValueError("empty response")
                out = frame.copy()
                out.insert(0, "indicator", out.index.astype(str))
                out.to_csv(profit_path, index=False, encoding="utf-8")
                record(
                    f"{Q3_PROFIT_INTERFACE}:industrial_profit",
                    "ok",
                    f"rows={len(out)} path=工业 > 工业企业主要经济指标",
                    str(profit_path),
                )
                log.info("industrial profit ok (%d indicator rows)", len(out))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                record(f"{Q3_PROFIT_INTERFACE}:industrial_profit", "failed", message, str(profit_path))
                log.error("industrial profit failed: %s", message)
    else:
        record(
            f"{Q3_PROFIT_INTERFACE}:industrial_profit",
            "missing",
            "akshare interface does not exist",
            str(profit_path),
        )
        log.warning("%s missing from akshare; industrial profit skipped", Q3_PROFIT_INTERFACE)

    if SHIBOR_CLEAN.exists() and SHIBOR_CLEAN.stat().st_size > 0:
        record("committed SHIBOR", "ok", "using data/clean/shibor.csv", str(SHIBOR_CLEAN))
        log.info("using committed SHIBOR snapshot %s", SHIBOR_CLEAN)
    else:
        record("committed SHIBOR", "failed", "data/clean/shibor.csv missing", str(SHIBOR_CLEAN))
        log.error("committed SHIBOR snapshot missing: %s", SHIBOR_CLEAN)

    log.info("Q3 fetch log written to %s", log_path)
    return log_path


Q3_PANEL_START = "2010-01"
Q3_REQUIRED = ["pmi", "gdp_yoy", "cgb10y", "spread_10y_2y", "shibor_3m"]

Q3_RAW_INSPECTION = {
    "macro_china_pmi": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_non_man_pmi": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_cx_pmi_yearly": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_cx_services_pmi_yearly": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_cpi": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_cpi_yearly": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_ppi": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_ppi_yearly": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_money_supply": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_shrzgm": ("月份", "YYYYMM", "monthly", "reference month", "stale"),
    "macro_china_new_financial_credit": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_lpr": ("TRADE_DATE", "YYYY-MM-DD", "monthly", "reference month", "yes"),
    "macro_china_consumer_goods_retail": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_industrial_production_yoy": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_gyzjz": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_gdzctz": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_exports_yoy": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_imports_yoy": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_trade_balance": ("日期", "YYYY-MM-DD", "monthly", "announcement date", "stale"),
    "macro_china_hgjck": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_xfzxx": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_society_electricity": ("统计时间", "YYYY.M", "monthly", "reference month", "yes"),
    "macro_china_gdp": ("季度", "YYYY年第Q季度", "quarterly", "reference quarter", "yes"),
    "macro_china_gdp_yearly": ("日期", "YYYY-MM-DD", "quarterly", "announcement date", "yes"),
    "macro_china_real_estate": ("日期", "YYYY-MM-01", "monthly", "reference month", "stale"),
    "macro_china_enterprise_boom_index": ("季度", "YYYY年第Q季度", "quarterly", "reference quarter", "unused"),
    "macro_china_fx_gold": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_urban_unemployment": ("date", "YYYYMM", "monthly", "reference month", "yes"),
    "macro_china_czsr": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_qyspjg": ("月份", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_stock_market_cap": ("数据日期", "YYYY年MM月份", "monthly", "reference month", "yes"),
    "macro_china_commodity_price_index": ("日期", "YYYY-MM-DD", "daily/weekly", "reference date", "yes"),
    "macro_china_society_traffic_volume": ("统计时间", "unknown", "unknown", "unknown", "failed"),
    "macro_china_central_bank_balance": ("统计时间", "YYYY.M", "monthly", "reference month", "yes"),
    "macro_china_bank_financing": ("日期", "YYYY-MM-01", "monthly", "reference month", "unused"),
    "macro_china_retail_price_index": ("统计月份", "YYYY.M", "monthly", "reference month", "stale"),
    "macro_china_market_margin_sh": ("日期", "YYYY-MM-DD", "daily", "reference date", "yes"),
    "bond_zh_us_rate": ("日期", "YYYY-MM-DD", "daily", "reference date", "yes"),
    "macro_china_nbs_industrial_profit": ("2026年9月", "YYYY年M月", "monthly", "reference month", "yes"),
}


def _read_q3_raw(name: str) -> pd.DataFrame | None:
    path = Q3_DIR / f"{name}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, dtype=str)


def _period_month(value) -> pd.Period | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.search(r"(\d{4})年(\d{1,2})月", str(value))
    if not match:
        return None
    return pd.Period(year=int(match.group(1)), month=int(match.group(2)), freq="M")


def _period_quarter(value) -> tuple[pd.Period, int] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = re.match(r"\s*(\d{4})年第(\d)(?:-(\d))?季度\s*", str(value))
    if not match:
        return None
    year = int(match.group(1))
    quarter = int(match.group(3) or match.group(2))
    return pd.Period(year=year, month=quarter * 3, freq="M"), quarter


def _period_yyyymm(value) -> pd.Period | None:
    match = re.match(r"\s*(\d{4})(\d{2})\s*", str(value))
    if not match:
        return None
    return pd.Period(year=int(match.group(1)), month=int(match.group(2)), freq="M")


def _period_dot_month(value) -> pd.Period | None:
    match = re.match(r"\s*(\d{4})\.(\d{1,2})\s*", str(value))
    if not match:
        return None
    return pd.Period(year=int(match.group(1)), month=int(match.group(2)), freq="M")


def _monthly(periods, values, agg: str = "last") -> pd.Series:
    frame = pd.DataFrame(
        {
            "period": list(periods),
            "value": pd.to_numeric(pd.Series(values), errors="coerce"),
        }
    ).dropna(subset=["period", "value"])
    if frame.empty:
        return pd.Series(dtype="float64")
    grouped = frame.groupby("period")["value"]
    result = grouped.mean() if agg == "mean" else grouped.last()
    return result.sort_index().astype("float64")


def _full_months(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    index = pd.period_range(series.index.min(), series.index.max(), freq="M")
    return series.reindex(index)


def _yoy_from_level(series: pd.Series) -> pd.Series:
    series = _full_months(series)
    return ((series / series.shift(12) - 1.0) * 100.0).astype("float64")


def _rolling12_yoy(series: pd.Series) -> pd.Series:
    series = _full_months(series)
    rolling = series.rolling(12, min_periods=12).sum()
    return ((rolling / rolling.shift(12) - 1.0) * 100.0).astype("float64")


def _daily_monthly(
    frame: pd.DataFrame, date_col: str, value_col: str, agg: str = "mean", min_obs: int = 10
) -> pd.Series:
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    work = pd.DataFrame({"period": dates.dt.to_period("M"), "value": values}).dropna()
    if work.empty:
        return pd.Series(dtype="float64")
    grouped = work.groupby("period")["value"]
    if agg == "mean":
        result = grouped.mean()
        counts = grouped.count()
        result = result.where(counts >= min_obs)
    else:
        result = grouped.last()
    return result.sort_index().astype("float64")


def _panel_months() -> pd.PeriodIndex:
    today = date.today()
    last_complete = pd.Period(today, freq="M") - 1
    return pd.period_range(Q3_PANEL_START, last_complete, freq="M")


def _quarter_key(period: pd.Period) -> tuple[int, int]:
    return period.year, period.quarter


def _derived_quarterly_gdp() -> tuple[pd.Series, dict]:
    raw = _read_q3_raw("macro_china_gdp")
    derived = pd.Series(dtype="float64")
    info = {
        "derived_quarters": 0,
        "validation_quarters": 0,
        "validation_max_abs_diff_pp": None,
        "validation_mean_abs_diff_pp": None,
    }
    if raw is not None:
        parsed = raw["季度"].map(_period_quarter)
        frame = pd.DataFrame(
            {
                "period": [item[0] if item else None for item in parsed],
                "quarter": [item[1] if item else None for item in parsed],
                "level": pd.to_numeric(raw["国内生产总值-绝对值"], errors="coerce"),
                "cum_yoy": pd.to_numeric(raw["国内生产总值-同比增长"], errors="coerce"),
            }
        ).dropna()
        frame = frame.drop_duplicates("period", keep="last").sort_values("period").reset_index(drop=True)
        if not frame.empty:
            first_year = int(frame["period"].iloc[0].year)
            real_ytd: dict[pd.Period, float] = {}
            for row in frame.itertuples(index=False):
                year, quarter = row.period.year, int(row.quarter)
                if year == first_year:
                    real_ytd[row.period] = float(row.level)
                else:
                    prior = pd.Period(year=year - 1, month=quarter * 3, freq="M")
                    real_ytd[row.period] = real_ytd[prior] * (1.0 + float(row.cum_yoy) / 100.0)
            frame["real_ytd"] = [real_ytd[period] for period in frame["period"]]
            frame["real_q"] = frame["real_ytd"] - frame.groupby(frame["period"].dt.year)["real_ytd"].shift(1).fillna(0.0)
            frame["prior_q"] = frame.groupby(frame["period"].dt.quarter)["real_q"].shift(1)
            frame["qyoy"] = (frame["real_q"] / frame["prior_q"] - 1.0) * 100.0
            derived = pd.Series(frame["qyoy"].values, index=frame["period"]).dropna().sort_index()
            info["derived_quarters"] = int(len(derived))

    official = _read_q3_raw("macro_china_gdp_yearly")
    if official is not None and not derived.empty:
        dates = pd.to_datetime(official["日期"], errors="coerce")
        values = pd.to_numeric(official["今值"], errors="coerce")
        rows = []
        for stamp, value in zip(dates, values):
            if pd.isna(stamp) or pd.isna(value):
                continue
            if stamp.month <= 3:
                reference = pd.Period(year=stamp.year - 1, month=12, freq="M")
            elif stamp.month <= 6:
                reference = pd.Period(year=stamp.year, month=3, freq="M")
            elif stamp.month <= 9:
                reference = pd.Period(year=stamp.year, month=6, freq="M")
            else:
                reference = pd.Period(year=stamp.year, month=9, freq="M")
            rows.append((reference, stamp, float(value)))
        if rows:
            overlay = pd.DataFrame(rows, columns=["period", "announcement", "value"])
            overlay = overlay.sort_values("announcement").drop_duplicates("period", keep="last")
            official_series = pd.Series(overlay["value"].values, index=overlay["period"]).sort_index()
            comparison = pd.concat([derived.rename("derived"), official_series.rename("official")], axis=1).dropna()
            if not comparison.empty:
                diff = (comparison["derived"] - comparison["official"]).abs()
                info["validation_quarters"] = int(len(comparison))
                info["validation_max_abs_diff_pp"] = float(diff.max())
                info["validation_mean_abs_diff_pp"] = float(diff.mean())
    return derived, info


def _q3_series() -> tuple[dict[str, pd.Series], list[dict], dict]:
    series: dict[str, pd.Series] = {}
    meta: list[dict] = []
    gdp_info = {
        "derived_quarters": 0,
        "validation_quarters": 0,
        "validation_max_abs_diff_pp": None,
        "validation_mean_abs_diff_pp": None,
    }

    def add(name, values, source_interface, source_column, transform, lag, lag_basis):
        if values is None:
            return
        series[name] = values.astype("float64").sort_index()
        meta.append(
            {
                "series": name,
                "source_interface": source_interface,
                "source_column": source_column,
                "transform": transform,
                "lag": int(lag),
                "lag_basis": lag_basis,
            }
        )

    pmi_raw = _read_q3_raw("macro_china_pmi")
    if pmi_raw is not None:
        periods = pmi_raw["月份"].map(_period_month)
        add(
            "pmi",
            _monthly(periods, pmi_raw["制造业-指数"]),
            "macro_china_pmi",
            "制造业-指数",
            "level (diffusion index; source seasonally adjusted)",
            0,
            "brief default: NBS manufacturing PMI published around month end (assumption; raw has no release date)",
        )
        add(
            "pmi_non_man",
            _monthly(periods, pmi_raw["非制造业-指数"]),
            "macro_china_pmi",
            "非制造业-指数",
            "level (diffusion index; source seasonally adjusted)",
            0,
            "brief default: NBS non-manufacturing PMI published around month end (assumption; raw has no release date)",
        )

    gdp, gdp_info = _derived_quarterly_gdp()
    add(
        "gdp_yoy",
        gdp,
        "macro_china_gdp",
        "国内生产总值-同比增长",
        "quarterly real YoY %, derived by chaining official cumulative growth and differencing YTD real levels (current revised vintage); Jin10 GDP report used only as an external cross-check",
        1,
        "brief default: GDP published about mid-month after quarter end (assumption; raw has no release date)",
    )

    cpi_raw = _read_q3_raw("macro_china_cpi")
    if cpi_raw is not None:
        add(
            "cpi_yoy",
            _monthly(cpi_raw["月份"].map(_period_month), cpi_raw["全国-同比增长"]),
            "macro_china_cpi",
            "全国-同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: CPI published in following month (assumption; raw has no release date)",
        )

    ppi_raw = _read_q3_raw("macro_china_ppi")
    if ppi_raw is not None:
        add(
            "ppi_yoy",
            _monthly(ppi_raw["月份"].map(_period_month), ppi_raw["当月同比增长"]),
            "macro_china_ppi",
            "当月同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: PPI published in following month (assumption; raw has no release date)",
        )

    money_raw = _read_q3_raw("macro_china_money_supply")
    if money_raw is not None:
        periods = money_raw["月份"].map(_period_month)
        add(
            "m2_yoy",
            _monthly(periods, money_raw["货币和准货币(M2)-同比增长"]),
            "macro_china_money_supply",
            "货币和准货币(M2)-同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: money supply published in following month (assumption; raw has no release date)",
        )
        add(
            "m1_yoy",
            _monthly(periods, money_raw["货币(M1)-同比增长"]),
            "macro_china_money_supply",
            "货币(M1)-同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: money supply published in following month (assumption; raw has no release date)",
        )

    tsf_raw = _read_q3_raw("macro_china_shrzgm")
    if tsf_raw is not None:
        tsf_level = _monthly(tsf_raw["月份"].map(_period_yyyymm), tsf_raw["社会融资规模增量"])
        add(
            "tsf_12m_yoy",
            _rolling12_yoy(tsf_level),
            "macro_china_shrzgm",
            "社会融资规模增量",
            "12-month rolling sum, then YoY % (no seasonal adjustment)",
            1,
            "brief default: TSF published in following month (assumption; raw has no release date)",
        )

    credit_raw = _read_q3_raw("macro_china_new_financial_credit")
    if credit_raw is not None:
        add(
            "new_credit_yoy",
            _monthly(credit_raw["月份"].map(_period_month), credit_raw["当月-同比增长"]),
            "macro_china_new_financial_credit",
            "当月-同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: credit published in following month (assumption; raw has no release date)",
        )

    lpr_raw = _read_q3_raw("macro_china_lpr")
    if lpr_raw is not None:
        periods = pd.to_datetime(lpr_raw["TRADE_DATE"], errors="coerce").dt.to_period("M")
        add(
            "lpr_1y",
            _monthly(periods, lpr_raw["LPR1Y"]),
            "macro_china_lpr",
            "LPR1Y",
            "month-end last published rate, level in percent",
            0,
            "LPR is announced on the 20th of the reference month; public by month end (source convention)",
        )
        add(
            "lpr_5y",
            _monthly(periods, lpr_raw["LPR5Y"]),
            "macro_china_lpr",
            "LPR5Y",
            "month-end last published rate, level in percent",
            0,
            "LPR is announced on the 20th of the reference month; public by month end (source convention)",
        )

    retail_raw = _read_q3_raw("macro_china_consumer_goods_retail")
    if retail_raw is not None:
        add(
            "retail_yoy",
            _monthly(retail_raw["月份"].map(_period_month), retail_raw["同比增长"]),
            "macro_china_consumer_goods_retail",
            "同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: retail sales published in following month (assumption; raw has no release date)",
        )

    ip_raw = _read_q3_raw("macro_china_gyzjz")
    if ip_raw is not None:
        add(
            "industrial_production_yoy",
            _monthly(ip_raw["月份"].map(_period_month), ip_raw["同比增长"]),
            "macro_china_gyzjz",
            "同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: industrial output published in following month (assumption; raw has no release date)",
        )

    fai_raw = _read_q3_raw("macro_china_gdzctz")
    if fai_raw is not None:
        fai_level = _monthly(fai_raw["月份"].map(_period_month), fai_raw["自年初累计"])
        add(
            "fai_ytd_yoy",
            _yoy_from_level(fai_level),
            "macro_china_gdzctz",
            "自年初累计",
            "YTD level, YoY % vs same month previous year (no seasonal adjustment)",
            1,
            "brief default: fixed investment published in following month (assumption; raw has no release date)",
        )

    trade_raw = _read_q3_raw("macro_china_hgjck")
    if trade_raw is not None:
        periods = trade_raw["月份"].map(_period_month)
        add(
            "exports_yoy",
            _monthly(periods, trade_raw["当月出口额-同比增长"]),
            "macro_china_hgjck",
            "当月出口额-同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: trade published in following month (assumption; raw has no release date)",
        )
        add(
            "imports_yoy",
            _monthly(periods, trade_raw["当月进口额-同比增长"]),
            "macro_china_hgjck",
            "当月进口额-同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: trade published in following month (assumption; raw has no release date)",
        )

    confidence_raw = _read_q3_raw("macro_china_xfzxx")
    if confidence_raw is not None:
        add(
            "consumer_confidence",
            _monthly(confidence_raw["月份"].map(_period_month), confidence_raw["消费者信心指数-指数值"]),
            "macro_china_xfzxx",
            "消费者信心指数-指数值",
            "level (index; no seasonal adjustment)",
            1,
            "brief default: consumer confidence published in following month (assumption; raw has no release date)",
        )

    electricity_raw = _read_q3_raw("macro_china_society_electricity")
    if electricity_raw is not None:
        add(
            "electricity_yoy",
            _monthly(electricity_raw["统计时间"].map(_period_dot_month), electricity_raw["全社会用电量同比"]),
            "macro_china_society_electricity",
            "全社会用电量同比",
            "YTD YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: electricity published in following month (assumption; raw has no release date)",
        )

    estate_raw = _read_q3_raw("macro_china_real_estate")
    if estate_raw is not None:
        add(
            "real_estate_climate",
            _monthly(
                pd.to_datetime(estate_raw["日期"], errors="coerce").dt.to_period("M"),
                estate_raw["最新值"],
            ),
            "macro_china_real_estate",
            "最新值",
            "level (climate index; no seasonal adjustment)",
            1,
            "brief default: real-estate climate index published in following month (assumption; raw has no release date)",
        )

    fx_raw = _read_q3_raw("macro_china_fx_gold")
    if fx_raw is not None:
        add(
            "fx_reserves_yoy",
            _monthly(fx_raw["月份"].map(_period_month), fx_raw["国家外汇储备-同比"]),
            "macro_china_fx_gold",
            "国家外汇储备-同比",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: FX reserves published in following month (assumption; raw has no release date)",
        )

    unemployment_raw = _read_q3_raw("macro_china_urban_unemployment")
    if unemployment_raw is not None:
        subset = unemployment_raw[unemployment_raw["item"].str.strip() == "全国城镇调查失业率"]
        add(
            "unemployment",
            _monthly(subset["date"].map(_period_yyyymm), subset["value"]),
            "macro_china_urban_unemployment",
            "value (item=全国城镇调查失业率)",
            "level in percent (no seasonal adjustment)",
            1,
            "brief default: labour survey published in following month (assumption; raw has no release date)",
        )

    fiscal_raw = _read_q3_raw("macro_china_czsr")
    if fiscal_raw is not None:
        add(
            "fiscal_revenue_yoy",
            _monthly(fiscal_raw["月份"].map(_period_month), fiscal_raw["当月-同比增长"]),
            "macro_china_czsr",
            "当月-同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: fiscal revenue published in following month (assumption; raw has no release date)",
        )

    qyspjg_raw = _read_q3_raw("macro_china_qyspjg")
    if qyspjg_raw is not None:
        add(
            "commodity_price_yoy",
            _monthly(qyspjg_raw["月份"].map(_period_month), qyspjg_raw["总指数-同比增长"]),
            "macro_china_qyspjg",
            "总指数-同比增长",
            "YoY % as reported (no seasonal adjustment)",
            1,
            "brief default: corporate commodity price index published in following month (assumption; raw has no release date)",
        )

    cap_raw = _read_q3_raw("macro_china_stock_market_cap")
    if cap_raw is not None:
        total_cap = pd.to_numeric(cap_raw["市价总值-上海"], errors="coerce") + pd.to_numeric(
            cap_raw["市价总值-深圳"], errors="coerce"
        )
        cap_level = _monthly(cap_raw["数据日期"].map(_period_month), total_cap)
        add(
            "stock_mktcap_yoy",
            _yoy_from_level(cap_level),
            "macro_china_stock_market_cap",
            "市价总值-上海 + 市价总值-深圳",
            "total market cap level, YoY % vs same month previous year (no seasonal adjustment)",
            0,
            "month-end stock market data; public at month end (source convention)",
        )

    commodity_raw = _read_q3_raw("macro_china_commodity_price_index")
    if commodity_raw is not None:
        commodity_level = _daily_monthly(commodity_raw, "日期", "最新值", agg="mean", min_obs=4)
        add(
            "commodity_spot_yoy",
            _yoy_from_level(commodity_level),
            "macro_china_commodity_price_index",
            "最新值",
            "monthly mean of daily/weekly index, then YoY % (no seasonal adjustment)",
            0,
            "daily market index; monthly mean available at month end (source convention)",
        )

    central_raw = _read_q3_raw("macro_china_central_bank_balance")
    if central_raw is not None:
        central_level = _monthly(
            central_raw["统计时间"].map(_period_dot_month), central_raw["总资产"]
        )
        add(
            "central_bank_assets_yoy",
            _yoy_from_level(central_level),
            "macro_china_central_bank_balance",
            "总资产",
            "total assets level, YoY % vs same month previous year (no seasonal adjustment)",
            1,
            "brief default: central-bank balance sheet published with a lag (assumption; raw has no release date)",
        )

    margin_raw = _read_q3_raw("macro_china_market_margin_sh")
    if margin_raw is not None:
        margin_level = _daily_monthly(margin_raw, "日期", "融资融券余额", agg="mean", min_obs=10)
        add(
            "margin_balance_yoy",
            _yoy_from_level(margin_level),
            "macro_china_market_margin_sh",
            "融资融券余额",
            "monthly mean of daily margin balance, then YoY % (no seasonal adjustment)",
            0,
            "daily market data; monthly mean available at month end (source convention)",
        )

    bond_raw = _read_q3_raw("bond_zh_us_rate")
    if bond_raw is not None:
        for name, column, transform in [
            ("cgb2y", "中国国债收益率2年", "monthly mean of daily yield, level in percent"),
            ("cgb5y", "中国国债收益率5年", "monthly mean of daily yield, level in percent"),
            ("cgb10y", "中国国债收益率10年", "monthly mean of daily yield, level in percent"),
            ("cgb30y", "中国国债收益率30年", "monthly mean of daily yield, level in percent"),
            ("spread_10y_2y", "中国国债收益率10年-2年", "monthly mean of daily spread, level in percent"),
        ]:
            add(
                name,
                _daily_monthly(bond_raw, "日期", column, agg="mean", min_obs=10),
                "bond_zh_us_rate",
                column,
                transform,
                0,
                "market yield/spread; public at month end (source convention)",
            )

    shibor = read_table(SHIBOR_CLEAN)
    if shibor is not None:
        add(
            "shibor_3m",
            _daily_monthly(shibor, "date", "3M", agg="mean", min_obs=10),
            "data/clean/shibor.csv (committed snapshot; akshare.macro_china_shibor_all)",
            "3M",
            "monthly mean of daily rate, level in percent",
            0,
            "market rate; public at month end (source convention)",
        )

    profit_raw = _read_q3_raw("macro_china_nbs_industrial_profit")
    if profit_raw is not None:
        mask = profit_raw["indicator"].str.contains("利润总额累计增长", na=False)
        if mask.any():
            row = profit_raw[mask].iloc[0]
            pairs = [(column, row[column]) for column in profit_raw.columns if column != "indicator"]
            periods = [_period_month(column) for column, _ in pairs]
            values = [value for _, value in pairs]
            add(
                "industrial_profit_yoy",
                _monthly(periods, values),
                "macro_china_nbs_nation (path=工业 > 工业企业主要经济指标)",
                "利润总额累计增长(%)",
                "YTD cumulative YoY % as reported (no seasonal adjustment)",
                1,
                "brief default: industrial profits published in following month (assumption; NBS generic interface has no release date)",
            )

    return series, meta, gdp_info


def _outlier_mask(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(False, index=series.index)
    median = valid.median()
    mad = (valid - median).abs().median()
    if mad == 0:
        z = pd.Series(0.0, index=valid.index)
        z[valid != median] = float("inf")
    else:
        z = 0.67448975 * (valid - median) / mad
    mask = pd.Series(False, index=series.index)
    mask.loc[valid.index] = z.abs() > 5.0
    return mask


def build_macro_panel():
    series_map, meta_rows, gdp_info = _q3_series()
    months = _panel_months()
    if len(months) == 0:
        raise ValueError("empty panel month range")
    panel = pd.DataFrame(index=months)
    for name, values in series_map.items():
        aligned = values.reindex(months)
        panel[name] = aligned.astype("float64")

    ordered = [name for name in Q3_REQUIRED if name in panel.columns]
    ordered += [name for name in panel.columns if name not in ordered]
    panel = panel[ordered]

    dates = months.to_timestamp(how="end").normalize()
    panel.insert(0, "date", dates)
    panel = panel.sort_values("date").reset_index(drop=True)

    meta_index = {row["series"]: row for row in meta_rows}
    missing_rows = []
    outlier_rows = []
    for name in panel.columns:
        if name == "date":
            continue
        values = panel[name]
        valid = values.dropna()
        first = dates[values.notna()].min() if not valid.empty else None
        last = dates[values.notna()].max() if not valid.empty else None
        info = meta_index[name]
        info["first_date"] = first.strftime("%Y-%m-%d") if first is not None else ""
        info["last_date"] = last.strftime("%Y-%m-%d") if last is not None else ""
        info["missing_share"] = float(values.isna().mean())
        mask = _outlier_mask(values)
        info["outlier_count"] = int(mask.sum())
        missing_rows.append(
            {
                "series": name,
                "missing_count": int(values.isna().sum()),
                "missing_share": float(values.isna().mean()),
                "first_date": info["first_date"],
                "last_date": info["last_date"],
            }
        )
        for stamp in values.index[mask]:
            outlier_rows.append(
                {
                    "series": name,
                    "date": dates[stamp].strftime("%Y-%m-%d"),
                    "value": float(values.loc[stamp]),
                }
            )

    meta = pd.DataFrame(meta_rows)[
        [
            "series",
            "source_interface",
            "source_column",
            "transform",
            "lag",
            "lag_basis",
            "first_date",
            "last_date",
            "missing_share",
            "outlier_count",
        ]
    ]
    meta = meta.sort_values("series").reset_index(drop=True)
    outliers = pd.DataFrame(outlier_rows, columns=["series", "date", "value"])
    monthly_missing = []
    jan_feb_affected = [
        "retail_yoy",
        "industrial_production_yoy",
        "fai_ytd_yoy",
        "electricity_yoy",
        "industrial_profit_yoy",
    ]
    for name in jan_feb_affected:
        if name not in panel.columns:
            continue
        for month in [1, 2]:
            subset = panel.loc[panel["date"].dt.month == month, name]
            monthly_missing.append(
                {
                    "series": name,
                    "month": month,
                    "missing_share": float(subset.isna().mean()),
                    "reason": "January not published separately or combined with February; values left missing",
                }
            )
    return panel, meta, missing_rows, outliers, monthly_missing, gdp_info


def _q3_checks(panel: pd.DataFrame, meta: pd.DataFrame) -> list[dict]:
    checks = []

    def check(name: str, condition: bool, detail: str):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if condition:
            log.info("assertion passed: %s (%s)", name, detail)
        else:
            log.error("assertion FAILED: %s (%s)", name, detail)

    dates = pd.to_datetime(panel["date"])
    check("dates sorted", bool(dates.is_monotonic_increasing), f"n={len(dates)}")
    check("dates unique", bool(dates.is_unique), f"duplicates={int(dates.duplicated().sum())}")
    check(
        "month-end dates",
        bool(((dates + pd.offsets.MonthEnd(0)) == dates).all()),
        "all dates are calendar month ends",
    )
    series_columns = [column for column in panel.columns if column != "date"]
    check("d >= 20 series", len(series_columns) >= 20, f"d={len(series_columns)}")
    check("at least 120 months", len(panel) >= 120, f"months={len(panel)}")

    meta_names = set(meta["series"])
    missing_meta = sorted(set(series_columns) - meta_names)
    extra_meta = sorted(meta_names - set(series_columns))
    check("every panel column in macro_meta", not missing_meta, f"missing={missing_meta}")
    check("macro_meta has no extra series", not extra_meta, f"extra={extra_meta}")

    bad_lags = []
    for row in meta.itertuples(index=False):
        lag = row.lag
        if not isinstance(lag, (int, bool)) or int(lag) < 0:
            bad_lags.append(f"{row.series}={lag!r}")
    check("all lags non-negative integers", not bad_lags, f"bad={bad_lags}")

    pmi = panel["pmi"].dropna()
    check(
        "pmi in 30..70",
        bool((pmi >= 30).all() and (pmi <= 70).all()),
        f"min={pmi.min():.2f} max={pmi.max():.2f} n={len(pmi)}",
    )
    gdp = panel.loc[panel["gdp_yoy"].notna(), "date"]
    quarter_end = gdp.dt.month.isin([3, 6, 9, 12])
    check(
        "gdp_yoy only on quarter-end months",
        bool(quarter_end.all()),
        f"non_quarter_end={gdp.loc[~quarter_end].dt.strftime('%Y-%m-%d').tolist()}",
    )
    return checks


def _inspection_dates(name: str, frame: pd.DataFrame, date_col: str, date_format: str, frequency: str):
    if name == "macro_china_nbs_industrial_profit":
        periods = [_period_month(column) for column in frame.columns if column != "indicator"]
        periods = [period for period in periods if period is not None]
        if not periods:
            return "-", "-"
        return str(min(periods)), str(max(periods))
    if date_col not in frame.columns:
        return "-", "-"
    raw_values = frame[date_col].dropna()
    if raw_values.empty:
        return "-", "-"
    if date_format in {"YYYY年MM月份", "YYYY年M月"}:
        periods = [period for period in raw_values.map(_period_month) if period is not None]
        if not periods:
            return "-", "-"
        return str(min(periods)), str(max(periods))
    if date_format == "YYYY.M":
        periods = [period for period in raw_values.map(_period_dot_month) if period is not None]
        if not periods:
            return "-", "-"
        return str(min(periods)), str(max(periods))
    if date_format == "YYYYMM":
        periods = [period for period in raw_values.map(_period_yyyymm) if period is not None]
        if not periods:
            return "-", "-"
        return str(min(periods)), str(max(periods))
    if date_format == "YYYY年第Q季度":
        parsed = raw_values.map(_period_quarter)
        periods = [item[0] for item in parsed if item is not None]
        if not periods:
            return "-", "-"
        return str(min(periods)), str(max(periods))
    dates = pd.to_datetime(raw_values, errors="coerce").dropna()
    if dates.empty:
        return "-", "-"
    if frequency.startswith("daily"):
        return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
    return dates.min().strftime("%Y-%m"), dates.max().strftime("%Y-%m")


def _print_q3_raw_table():
    wide = Console(width=260)
    table = Table(title="Q3 raw files (inspected before build)")
    table.add_column("File", overflow="fold")
    table.add_column("Rows", justify="right")
    table.add_column("First")
    table.add_column("Last")
    table.add_column("Date format", overflow="fold")
    table.add_column("Columns", overflow="fold", max_width=90)
    table.add_column("Frequency", overflow="fold")
    table.add_column("Usable")
    for name, (date_col, date_format, frequency, usage, usable) in Q3_RAW_INSPECTION.items():
        frame = _read_q3_raw(name)
        if frame is None:
            table.add_row(
                name,
                "0",
                "-",
                "-",
                date_format,
                "-",
                frequency,
                "no (missing/failed)",
            )
            continue
        first, last = _inspection_dates(name, frame, date_col, date_format, frequency)
        if name == "macro_china_nbs_industrial_profit":
            columns = f"indicator + {len(frame.columns) - 1} period columns"
        else:
            columns = ", ".join(str(column) for column in frame.columns)
        table.add_row(
            name,
            f"{len(frame):,}",
            first,
            last,
            date_format,
            columns,
            f"{frequency} / {usage}",
            usable,
        )
    wide.print(table)


def _print_q3_report(
    panel: pd.DataFrame,
    meta: pd.DataFrame,
    missing_rows: list[dict],
    outliers: pd.DataFrame,
    monthly_missing: list[dict],
    checks: list[dict],
    gdp_info: dict,
    interfaces: list[dict],
):
    failure_table = Table(title="Q3 interface status (all)")
    failure_table.add_column("Interface", overflow="fold")
    failure_table.add_column("Status")
    failure_table.add_column("Detail", overflow="fold")
    if interfaces:
        for item in interfaces:
            failure_table.add_row(item["interface"], item["status"], item["detail"])
    else:
        failure_table.add_row("none", "-", "no fetch records in this invocation")
    console.print(failure_table)

    wide = Console(width=240)
    series_table = Table(title="Q3 final series")
    series_table.add_column("Series", overflow="fold")
    series_table.add_column("Lag", justify="right")
    series_table.add_column("Transform", overflow="fold", max_width=80)
    series_table.add_column("First")
    series_table.add_column("Last")
    series_table.add_column("Missing", justify="right")
    series_table.add_column("Outliers", justify="right")
    for row in meta.itertuples(index=False):
        series_table.add_row(
            row.series,
            str(row.lag),
            row.transform,
            row.first_date,
            row.last_date,
            f"{row.missing_share:.1%}",
            str(row.outlier_count),
        )
    wide.print(series_table)

    source_table = Table(title="Q3 sources and lag basis")
    source_table.add_column("Series", overflow="fold")
    source_table.add_column("Source interface", overflow="fold", max_width=45)
    source_table.add_column("Source column", overflow="fold", max_width=30)
    source_table.add_column("Lag", justify="right")
    source_table.add_column("Lag basis", overflow="fold", max_width=80)
    for row in meta.itertuples(index=False):
        source_table.add_row(
            row.series,
            row.source_interface,
            row.source_column,
            str(row.lag),
            row.lag_basis,
        )
    wide.print(source_table)

    missing_table = Table(title="Q3 missing-value summary")
    missing_table.add_column("Series")
    missing_table.add_column("Missing", justify="right")
    missing_table.add_column("Share", justify="right")
    missing_table.add_column("First")
    missing_table.add_column("Last")
    for item in missing_rows:
        missing_table.add_row(
            item["series"],
            str(item["missing_count"]),
            f"{item['missing_share']:.1%}",
            item["first_date"],
            item["last_date"],
        )
    console.print(missing_table)

    if monthly_missing:
        table = Table(title="January/February missing or combined")
        table.add_column("Series")
        table.add_column("Month", justify="right")
        table.add_column("Missing share", justify="right")
        for item in monthly_missing:
            table.add_row(item["series"], str(item["month"]), f"{item['missing_share']:.1%}")
        console.print(table)

    check_table = Table(title="Q3 assertions")
    check_table.add_column("Check")
    check_table.add_column("Result")
    check_table.add_column("Detail")
    for item in checks:
        check_table.add_row(item["name"], "PASS" if item["passed"] else "FAIL", item["detail"])
    console.print(check_table)

    start = panel["date"].iloc[0].date().isoformat()
    end = panel["date"].iloc[-1].date().isoformat()
    pmi = panel["pmi"]
    pmi_below_50 = float((pmi < 50).mean())
    threshold_rows = []
    for threshold in [50.0, 49.5]:
        flag = (pmi < threshold) & (pmi.shift(1) < threshold)
        threshold_rows.append(
            {
                "threshold": threshold,
                "pmi_below_share": float((pmi < threshold).mean()),
                "regime_share": float(flag.mean()),
                "regime_months": int(flag.sum()),
            }
        )
    console.print(
        f"Panel: {start} to {end}, months={len(panel)}, d={len(panel.columns) - 1}, "
        f"GDP quarters derived={gdp_info.get('derived_quarters', 0)}, "
        f"Jin10 cross-check quarters={gdp_info.get('validation_quarters', 0)}, "
        f"max |diff|={gdp_info.get('validation_max_abs_diff_pp')}"
    )
    console.print(f"PMI < 50 share: {pmi_below_50:.1%}")
    threshold_table = Table(title="PMI regime shares")
    threshold_table.add_column("Threshold", justify="right")
    threshold_table.add_column("PMI below", justify="right")
    threshold_table.add_column("d_t regime share", justify="right")
    threshold_table.add_column("Regime months", justify="right")
    for item in threshold_rows:
        threshold_table.add_row(
            f"{item['threshold']:.1f}",
            f"{item['pmi_below_share']:.1%}",
            f"{item['regime_share']:.1%}",
            str(item["regime_months"]),
        )
    console.print(threshold_table)

    samples = panel[["date", "pmi", "gdp_yoy"]].copy()
    selected = pd.concat(
        [
            samples.head(3),
            samples[(samples["date"] >= "2020-01-01") & (samples["date"] <= "2020-04-30")],
            samples.tail(8),
        ]
    ).drop_duplicates()
    selected["date"] = selected["date"].dt.strftime("%Y-%m-%d")
    sample_table = Table(title="Q3 sample rows")
    sample_table.add_column("Date")
    sample_table.add_column("pmi", justify="right")
    sample_table.add_column("gdp_yoy", justify="right")
    for row in selected.itertuples(index=False):
        sample_table.add_row(
            row.date,
            "" if pd.isna(row.pmi) else f"{row.pmi:.2f}",
            "" if pd.isna(row.gdp_yoy) else f"{row.gdp_yoy:.3f}",
        )
    console.print(sample_table)

    if not outliers.empty:
        CACHE.mkdir(parents=True, exist_ok=True)
        outliers.to_csv(CACHE / "macro_outliers.csv", index=False, encoding="utf-8")
        console.print(
            f"Outliers (full-sample robust |z|>5, not removed): {len(outliers)} rows; "
            f"listed in {(CACHE / 'macro_outliers.csv').relative_to(ROOT)}"
        )
        preview = Table(title="Q3 outlier preview")
        preview.add_column("Series")
        preview.add_column("Date")
        preview.add_column("Value", justify="right")
        for row in outliers.head(15).itertuples(index=False):
            preview.add_row(row.series, row.date, f"{row.value:.4f}")
        console.print(preview)
    else:
        console.print("Outliers (full-sample robust |z|>5, not removed): none")


def _update_q3_metadata(
    panel: pd.DataFrame,
    meta: pd.DataFrame,
    missing_rows: list[dict],
    checks: list[dict],
    monthly_missing: list[dict],
    threshold_rows: list[dict],
    gdp_info: dict,
    interfaces: list[dict],
):
    metadata = json.loads(METADATA.read_text(encoding="utf-8")) if METADATA.exists() else {}
    artifacts = metadata.setdefault("artifacts", {})
    for path, source, interface, schema in [
        (
            MACRO_PANEL,
            "Q3 monthly macro panel from akshare odds and committed SHIBOR",
            "see macro_meta.csv",
            list(panel.columns),
        ),
        (
            MACRO_META,
            "Q3 series metadata and conservative availability lags",
            "see source_interface column",
            list(meta.columns),
        ),
    ]:
        relative = str(path.relative_to(ROOT))
        artifacts[relative] = build_artifact(
            path,
            source,
            interface,
            [str(column) for column in schema],
            len(panel) if path == MACRO_PANEL else len(meta),
        )
    metadata["q3"] = {
        "panel_start": panel["date"].iloc[0].date().isoformat(),
        "panel_end": panel["date"].iloc[-1].date().isoformat(),
        "months": int(len(panel)),
        "d": int(len(panel.columns) - 1),
        "required_series": Q3_REQUIRED,
        "series": meta.to_dict(orient="records"),
        "lags": {row["series"]: int(row["lag"]) for row in meta.to_dict(orient="records")},
        "missing": missing_rows,
        "january_february_affected": monthly_missing,
        "pmi_regime": threshold_rows,
        "gdp_construction": gdp_info,
        "checks": checks,
        "interfaces": interfaces,
        "limitations": [
            "Data are the current revised vintage, not real-time vintages; revised history can differ from what was available in real time.",
            "Most raw interfaces carry no genuine release-date stamp; availability lags are conservative assumptions based on the brief defaults unless the source convention makes lag 0 definitional (market rates/yields).",
            "Jin10 announcement-date interfaces (non_man_pmi, cx_pmi, cx_services_pmi, cpi_yearly, ppi_yearly, industrial_production_yoy, exports_yoy, imports_yoy, trade_balance, gdp_yearly) were saved raw but mostly stop in 2025; reference-month Eastmoney/customs/NBS interfaces were used for the panel. macro_china_gdp_yearly is used only as a cross-check.",
            "macro_china_shrzgm (TSF) ends 2026-04 and macro_china_real_estate ends 2025-12 in the retrieved snapshot; both are retained with missing values after their last observation.",
            "The committed SHIBOR snapshot starts 2015-05-08 but is very sparse before 2017-03; the first monthly mean with at least 10 daily observations is 2017-03, so shibor_3m is missing earlier.",
            "macro_china_commodity_price_index is weekly before 2014-04 and daily afterwards; monthly means use available observations (at least four per month).",
            "January is not published separately for industrial production, fixed investment and industrial profits; January/February values are left missing rather than imputed.",
            "Industrial profits use the NBS generic interface macro_china_nbs_nation because akshare 1.20.25 has no dedicated industrial-profit interface.",
            "Outlier flags use the full sample only for reporting; models must standardize on training windows.",
        ],
    }
    metadata["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def q3_build():
    CLEAN.mkdir(parents=True, exist_ok=True)
    _print_q3_raw_table()
    panel, meta, missing_rows, outliers, monthly_missing, gdp_info = build_macro_panel()
    panel.to_csv(MACRO_PANEL, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    meta.to_csv(MACRO_META, index=False, encoding="utf-8")
    checks = _q3_checks(panel, meta)
    pmi = panel["pmi"]
    threshold_rows = []
    for threshold in [50.0, 49.5]:
        flag = (pmi < threshold) & (pmi.shift(1) < threshold)
        threshold_rows.append(
            {
                "threshold": threshold,
                "pmi_below_share": float((pmi < threshold).mean()),
                "regime_share": float(flag.mean()),
                "regime_months": int(flag.sum()),
            }
        )
    interfaces = INTERFACES if INTERFACES else []
    _update_q3_metadata(
        panel,
        meta,
        missing_rows,
        checks,
        monthly_missing,
        threshold_rows,
        gdp_info,
        interfaces,
    )
    _print_q3_report(
        panel,
        meta,
        missing_rows,
        outliers,
        monthly_missing,
        checks,
        gdp_info,
        interfaces,
    )
    failed = [item["name"] for item in checks if not item["passed"]]
    if failed:
        log.error("Q3 assertion failures: %s", failed)
        return 1
    log.info("Q3 macro panel built; all assertions passed")
    return 0


def q3_run():
    fetch_q3_raw()
    return q3_build()


def normalize_futures(frame: pd.DataFrame) -> pd.DataFrame:
    chinese = {
        "日期": "date",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "收盘价": "close",
        "成交量": "volume",
        "持仓量": "oi",
    }
    english = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "hold": "oi",
        "oi": "oi",
    }
    mapping = chinese if "日期" in frame.columns else english
    missing = [column for column in mapping if column not in frame.columns]
    if missing:
        raise ValueError(f"futures raw missing columns: {missing}")
    out = frame[list(mapping)].rename(columns=mapping).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in FUTURES_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    return out.sort_values("date").reset_index(drop=True)[FUTURES_COLUMNS]


def verify_futures(code: str, frame: pd.DataFrame) -> dict:
    duplicates = int(frame.duplicated("date").sum())
    price_columns = ["open", "high", "low", "close"]
    nonpositive = int((frame[price_columns] <= 0).sum().sum())
    missing_prices = int(frame[price_columns].isna().sum().sum())
    start = pd.Timestamp(frame["date"].min())
    end = pd.Timestamp(frame["date"].max())
    span_days = int((end - start).days)
    years = span_days / 365.25
    checks = {
        "dataset": code,
        "rows": int(len(frame)),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "span_days": span_days,
        "years": years,
        "duplicate_dates": duplicates,
        "nonpositive_prices": nonpositive,
        "missing_prices": missing_prices,
        "at_least_5_years": bool(years >= 5.0),
        "no_duplicate_dates": bool(duplicates == 0),
        "positive_prices": bool(nonpositive == 0 and missing_prices == 0),
    }
    return checks


def fetch_futures() -> tuple[list[tuple[str, str, pd.DataFrame]], list[dict]]:
    import akshare as ak

    RAW.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str, pd.DataFrame]] = []
    assertions: list[dict] = []
    for code, symbol in FUTURES_SYMBOLS.items():
        raw_path = RAW / f"futures_{code}.csv"
        source_path = RAW / f"futures_{code}_source.txt"
        source = "cached"
        if raw_path.exists() and raw_path.stat().st_size > 0:
            if source_path.exists():
                source = source_path.read_text(encoding="utf-8").strip()
            record(f"Futures {code}", "skipped", f"using cached source={source}", str(raw_path))
            log.info("Futures %s skipped; cached source=%s", code, source)
            raw = pd.read_csv(raw_path)
        else:
            attempts = [
                (
                    "akshare.futures_main_sina",
                    lambda: ak.futures_main_sina(
                        symbol=symbol, start_date="20050101", end_date="22220101"
                    ),
                ),
                ("akshare.futures_zh_daily_sina", lambda: ak.futures_zh_daily_sina(symbol=symbol)),
            ]
            failures = []
            raw = None
            for name, call in attempts:
                try:
                    with time_limit(180):
                        frame = call()
                    if frame is None or len(frame) == 0:
                        raise ValueError("empty response")
                    frame.to_csv(raw_path, index=False, encoding="utf-8")
                    source_path.write_text(name, encoding="utf-8")
                    source = name
                    raw = frame
                    record(f"Futures {code}", "ok", f"source={name} rows={len(frame)}", str(raw_path))
                    log.info("Futures %s source used: %s (%d rows)", code, name, len(frame))
                    break
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    failures.append(f"{name}: {message}")
                    log.error("Futures %s %s failed: %s", code, name, message)
            if raw is None:
                record(f"Futures {code}", "failed", " | ".join(failures), str(raw_path))
                assertions.append(
                    {
                        "dataset": code,
                        "rows": 0,
                        "start": None,
                        "end": None,
                        "span_days": 0,
                        "years": 0.0,
                        "duplicate_dates": 0,
                        "nonpositive_prices": 0,
                        "missing_prices": 0,
                        "at_least_5_years": False,
                        "no_duplicate_dates": False,
                        "positive_prices": False,
                    }
                )
                continue
        clean = normalize_futures(raw)
        clean_path = CLEAN / f"futures_{code}.csv"
        clean.to_csv(clean_path, index=False, encoding="utf-8")
        checks = verify_futures(code, clean)
        assertions.append(checks)
        results.append((code, source, clean))
        log.info(
            "Futures %s clean rows=%d %s..%s years=%.2f",
            code,
            checks["rows"],
            checks["start"],
            checks["end"],
            checks["years"],
        )
    return results, assertions


def update_futures_metadata(results: list[tuple[str, str, pd.DataFrame]], assertions: list[dict]):
    if not METADATA.exists():
        raise FileNotFoundError(f"metadata missing: {METADATA}")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    artifacts = metadata.setdefault("artifacts", {})
    for code, source, frame in results:
        relative = f"data/clean/futures_{code}.csv"
        artifacts[relative] = build_artifact(
            ROOT / relative,
            f"Sina main continuous futures {FUTURES_SYMBOLS[code]} ({code}); interface={source}",
            source,
            FUTURES_COLUMNS,
            len(frame),
        )
    metadata["futures_assertions"] = assertions
    metadata["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def print_futures_report(results: list[tuple[str, str, pd.DataFrame]], assertions: list[dict]):
    table = Table(title="Q2 futures snapshots")
    table.add_column("Dataset")
    table.add_column("Source")
    table.add_column("Rows", justify="right")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Years", justify="right")
    for code, source, frame in results:
        table.add_row(
            code,
            source,
            f"{len(frame):,}",
            str(frame["date"].iloc[0]),
            str(frame["date"].iloc[-1]),
            f"{(pd.Timestamp(frame['date'].iloc[-1]) - pd.Timestamp(frame['date'].iloc[0])).days / 365.25:.2f}",
        )
    console.print(table)

    check_table = Table(title="Futures assertions")
    check_table.add_column("Dataset")
    check_table.add_column("5y")
    check_table.add_column("No duplicate dates")
    check_table.add_column("Positive prices")
    for item in assertions:
        check_table.add_row(
            item["dataset"],
            "PASS" if item["at_least_5_years"] else "FAIL",
            "PASS" if item["no_duplicate_dates"] else "FAIL",
            "PASS" if item["positive_prices"] else "FAIL",
        )
    console.print(check_table)


def load_csi() -> tuple[pd.DataFrame, str]:
    path = RAW / "csi300.csv"
    source_path = RAW / "csi300_source.txt"
    frame = read_table(path)
    if frame is None:
        return pd.DataFrame(columns=["date", "close"]), "missing"
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna().drop_duplicates("date", keep="last").sort_values("date")
    source = source_path.read_text(encoding="utf-8").strip() if source_path.exists() else "cached"
    return frame, source


def resolve_expiry_dates(csi: pd.DataFrame, nominal_dates: set[date]) -> tuple[dict[date, date], dict]:
    calendar = sorted(csi["date"])
    calendar_set = set(calendar)
    if not calendar:
        mapping = {nominal: nominal for nominal in nominal_dates}
        validation = {
            "calendar_start": None,
            "calendar_end": None,
            "rolled": [],
            "verified": [],
            "unverified_future": sorted(nominal.isoformat() for nominal in nominal_dates),
        }
        return mapping, validation
    last = calendar[-1]
    mapping = {}
    rolled = []
    verified = []
    unverified_future = []
    for nominal in sorted(nominal_dates):
        if nominal > last:
            mapping[nominal] = nominal
            unverified_future.append(nominal.isoformat())
        elif nominal in calendar_set:
            mapping[nominal] = nominal
            verified.append(nominal.isoformat())
        else:
            rolled_to = next((day for day in calendar if day > nominal), None)
            if rolled_to is None:
                mapping[nominal] = nominal
                unverified_future.append(nominal.isoformat())
            else:
                mapping[nominal] = rolled_to
                rolled.append(
                    {
                        "nominal": nominal.isoformat(),
                        "expiry": rolled_to.isoformat(),
                        "reason": "holiday_roll",
                    }
                )
    validation = {
        "calendar_start": calendar[0].isoformat(),
        "calendar_end": last.isoformat(),
        "rolled": rolled,
        "verified": verified,
        "unverified_future": unverified_future,
    }
    return mapping, validation


def build_options(csi: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    csi_close = dict(zip(csi["date"], csi["close"]))
    csi_dates = set(csi["date"])
    frames = []
    files = sorted(CFFEX_DIR.glob("*_1.csv"))
    raw_io = 0
    ignored_non_io = 0
    invalid_codes = 0
    for path in files:
        day = datetime.strptime(path.stem.split("_")[0], "%Y%m%d").date()
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        frame.columns = [str(column).strip() for column in frame.columns]
        if "合约代码" not in frame.columns:
            log.warning("CFFEX %s has no 合约代码 column; skipped", path.name)
            continue
        frame["contract"] = frame["合约代码"].str.strip()
        is_io = frame["contract"].str.upper().str.startswith("IO", na=False)
        ignored_non_io += int((~is_io).sum())
        frame = frame[is_io].copy()
        if frame.empty:
            continue
        parsed = frame["contract"].str.extract(
            r"(?i)^IO(\d{2})(\d{2})-?([CP])-?(\d+(?:\.\d+)?)$"
        )
        invalid = parsed.isna().any(axis=1)
        invalid_codes += int(invalid.sum())
        frame = frame[~invalid].copy()
        parsed = parsed[~invalid].copy()
        if frame.empty:
            continue
        frame["date"] = day
        frame["expiry"] = [
            third_friday(2000 + int(year), int(month))
            for year, month in zip(parsed[0], parsed[1])
        ]
        frame["cp"] = parsed[2].str.upper()
        frame["K"] = pd.to_numeric(parsed[3], errors="coerce")
        for source, target in [
            ("今开盘", "open"),
            ("最高价", "high"),
            ("最低价", "low"),
            ("今收盘", "close"),
            ("今结算", "settle"),
            ("成交量", "volume"),
            ("持仓量", "oi"),
        ]:
            frame[target] = pd.to_numeric(frame[source], errors="coerce")
        frame["S"] = frame["date"].map(csi_close)
        raw_io += len(frame)
        frames.append(frame[["date", "contract", "expiry", "cp", "K", "open", "high", "low", "close", "settle", "volume", "oi", "S"]])
    if not frames:
        empty = pd.DataFrame(columns=OPTION_COLUMNS)
        return empty, {
            "raw_io_rows": 0,
            "ignored_non_io": ignored_non_io,
            "invalid_codes": invalid_codes,
            "dropped_on_or_after_expiry": 0,
            "expiry_flags": [],
            "expiry_validation": {
                "calendar_start": None,
                "calendar_end": None,
                "rolled": [],
                "verified": [],
                "unverified_future": [],
            },
        }
    options = pd.concat(frames, ignore_index=True)
    nominal_dates = set(options["expiry"])
    expiry_mapping, expiry_validation = resolve_expiry_dates(csi, nominal_dates)
    options["expiry"] = options["expiry"].map(expiry_mapping)
    options = options[options["K"] > 0].copy()
    dropped = options[options["date"] >= options["expiry"]].copy()
    dropped_count = len(dropped)
    if dropped_count:
        log.warning(
            "dropping %d IO rows with date >= expiry (expiry-day rows): %s",
            dropped_count,
            sorted({str(value) for value in dropped["expiry"].unique()}),
        )
    options = options[options["expiry"] > options["date"]].copy()
    options["date"] = pd.to_datetime(options["date"]).dt.strftime("%Y-%m-%d")
    options["expiry"] = pd.to_datetime(options["expiry"]).dt.strftime("%Y-%m-%d")
    options = options.sort_values(["date", "expiry", "cp", "K"]).reset_index(drop=True)
    expiry_flags = [
        {"expiry": expiry, "reason": "unverified_future"}
        for expiry in expiry_validation["unverified_future"]
    ]
    for item in expiry_validation["rolled"]:
        log.info("expiry %s rolled to %s", item["nominal"], item["expiry"])
    for expiry in expiry_validation["unverified_future"]:
        log.warning("expiry %s is unverified_future", expiry)
    stats = {
        "raw_io_rows": raw_io,
        "ignored_non_io": ignored_non_io,
        "invalid_codes": invalid_codes,
        "dropped_on_or_after_expiry": dropped_count,
        "expiry_flags": expiry_flags,
        "expiry_validation": expiry_validation,
    }
    return options, stats


def build_shibor() -> pd.DataFrame:
    raw = read_table(RAW / "shibor.csv")
    if raw is None:
        return pd.DataFrame(columns=SHIBOR_COLUMNS)
    mapping = {
        "日期": "date",
        "O/N-定价": "ON",
        "1W-定价": "1W",
        "2W-定价": "2W",
        "1M-定价": "1M",
        "3M-定价": "3M",
        "6M-定价": "6M",
        "9M-定价": "9M",
        "1Y-定价": "1Y",
    }
    missing = [column for column in mapping if column not in raw.columns]
    if missing:
        log.error("shibor raw missing columns: %s", missing)
        return pd.DataFrame(columns=SHIBOR_COLUMNS)
    frame = raw[list(mapping)].rename(columns=mapping)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in SHIBOR_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    return frame[SHIBOR_COLUMNS].reset_index(drop=True)


def choose_sample_dates(options: pd.DataFrame) -> list[str]:
    if options.empty:
        return []
    available = sorted(pd.to_datetime(options["date"]).dt.date.unique())
    chosen = []
    for month in SAMPLE_MONTHS:
        candidates = [day for day in available if day.strftime("%Y-%m") == month]
        if not candidates:
            log.warning("no trading dates for sample month %s", month)
            continue
        mid = [day for day in candidates if 10 <= day.day <= 20]
        pool = mid if mid else candidates
        ranked = sorted(pool, key=lambda day: (abs(day.day - 15), day.day))
        selection = None
        for day in ranked:
            mask = options["date"] == day.isoformat()
            if options.loc[mask, "expiry"].nunique() >= 4:
                selection = day
                break
        if selection is None:
            selection = ranked[0]
            log.warning("sample date %s has fewer than 4 expiries", selection)
        chosen.append(selection.isoformat())
    return chosen


def sina_symbol(contract: str) -> str:
    cleaned = contract.replace("-", "")
    return cleaned[:2].lower() + cleaned[2:]


def fetch_sina_comparison(options: pd.DataFrame, sample_dates: list[str]) -> pd.DataFrame:
    raw_path = RAW / "sina_comparison.csv"
    if raw_path.exists() and raw_path.stat().st_size > 0:
        cached = pd.read_csv(raw_path, dtype=str)
        covered = bool(sample_dates) and all(
            int((cached["date"] == day).sum()) >= 5 for day in sample_dates
        )
        if covered:
            record("Sina option cross-check", "skipped", "cached comparison", str(raw_path))
            log.info("Sina cross-check skipped; cached comparison covers all sample dates")
            return cached
        log.info("Sina cross-check cache does not cover current sample dates; refetching")
    import akshare as ak

    rows = []
    for day in sample_dates:
        subset = options[options["date"] == day].copy()
        subset["distance"] = (subset["K"] - subset["S"]).abs()
        near = subset.sort_values(["distance", "contract"]).drop_duplicates("contract").head(5)
        for item in near.itertuples(index=False):
            symbol = sina_symbol(item.contract)
            status = "ok"
            error = ""
            sina_close = None
            try:
                with time_limit(30):
                    frame = ak.option_cffex_hs300_daily_sina(symbol=symbol)
                frame.columns = [str(column).strip() for column in frame.columns]
                date_col = "date" if "date" in frame.columns else "日期"
                close_col = "close" if "close" in frame.columns else "收盘价"
                frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
                match = frame[frame[date_col] == day]
                if match.empty:
                    status = "no_matching_date"
                    error = f"no Sina row for {day}"
                else:
                    sina_close = float(pd.to_numeric(match[close_col], errors="coerce").iloc[0])
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                log.error("Sina cross-check %s %s failed: %s", day, item.contract, error)
            diff = None
            if sina_close is not None and pd.notna(item.close):
                diff = float(item.close) - sina_close
            rows.append(
                {
                    "date": day,
                    "contract": item.contract,
                    "sina_symbol": symbol,
                    "cffex_close": item.close,
                    "sina_close": sina_close,
                    "diff": diff,
                    "status": status,
                    "error": error,
                }
            )
            time.sleep(0.3)
    frame = pd.DataFrame(rows, columns=SINA_COLUMNS)
    frame.to_csv(raw_path, index=False, encoding="utf-8")
    ok = int((frame["status"] == "ok").sum())
    record("Sina option cross-check", "ok" if ok else "failed", f"compared={ok} rows={len(frame)}", str(raw_path))
    return frame


def run_assertions(
    options: pd.DataFrame,
    shibor: pd.DataFrame,
    sample_dates: list[str],
    csi: pd.DataFrame,
) -> list[dict]:
    checks = []

    def check(name: str, condition: bool, detail: str):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if condition:
            log.info("assertion passed: %s (%s)", name, detail)
        else:
            log.error("assertion FAILED: %s (%s)", name, detail)

    duplicates = int(options.duplicated(["date", "contract"]).sum()) if not options.empty else 0
    check("no duplicate date+contract", duplicates == 0, f"duplicates={duplicates}")

    bad_k = int((pd.to_numeric(options["K"], errors="coerce") <= 0).sum()) if not options.empty else 0
    check("K > 0", bad_k == 0, f"violations={bad_k}")

    if options.empty:
        cp_ok = False
        cp_detail = "options table is empty"
        expiries_per_date = {}
    else:
        grouped = options.groupby(["date", "expiry", "K"])["cp"].nunique()
        incomplete = int((grouped != 2).sum())
        cp_ok = incomplete == 0
        cp_detail = f"date+expiry+K groups without both C and P={incomplete}"
        expiries_per_date = options.groupby("date")["expiry"].nunique().to_dict()
    check("C and P both present", cp_ok, cp_detail)

    expiry_ok = bool((pd.to_datetime(options["expiry"]) > pd.to_datetime(options["date"])).all()) if not options.empty else False
    check("expiry after date", expiry_ok, "all retained rows have expiry > date")

    if options.empty or csi.empty:
        historical_ok = False
        historical_detail = "options or CSI calendar is empty"
    else:
        calendar = set(pd.to_datetime(csi["date"]).dt.date)
        last_calendar_day = max(calendar)
        expiries = sorted(pd.to_datetime(options["expiry"]).dt.date.unique())
        historical = [day for day in expiries if day <= last_calendar_day]
        missing_expiries = [day.isoformat() for day in historical if day not in calendar]
        historical_ok = bool(historical) and not missing_expiries
        historical_detail = f"historical_expiries={len(historical)} non_trading={missing_expiries}"
    check("each historical expiry is a trading day", historical_ok, historical_detail)

    missing_s = int(options["S"].isna().sum()) if not options.empty else 0
    check("S present on every date", missing_s == 0, f"rows missing S={missing_s}")

    if sample_dates:
        span = (date.fromisoformat(max(sample_dates)) - date.fromisoformat(min(sample_dates))).days
    else:
        span = 0
    check(
        "at least 5 sample dates spanning at least 2 years",
        len(sample_dates) >= 5 and span >= 730,
        f"sample_dates={len(sample_dates)} span_days={span}",
    )

    min_expiries = min((expiries_per_date.get(day, 0) for day in sample_dates), default=0)
    check(
        "at least 4 expiries per sample date",
        bool(sample_dates) and min_expiries >= 4,
        f"minimum expiries across sample dates={min_expiries}",
    )
    return checks


def build_artifact(path: Path, source: str, interface: str, schema: list[str], rows: int) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "source": source,
        "interface": interface,
        "retrieval_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema": schema,
        "rows": rows,
        "sha256": sha256(path),
    }


def write_metadata(
    options: pd.DataFrame,
    shibor: pd.DataFrame,
    sina: pd.DataFrame,
    csi_source: str,
    build_stats: dict,
    sample_dates: list[str],
    assertions: list[dict],
):
    import akshare

    month_report = []
    for month in SAMPLE_MONTHS:
        subset = options[options["date"].str.startswith(month)] if not options.empty else options
        month_report.append(
            {
                "month": month,
                "trading_days": int(subset["date"].nunique()) if not subset.empty else 0,
                "rows": int(len(subset)),
            }
        )
    date_report = []
    for day in sample_dates:
        subset = options[options["date"] == day]
        date_report.append(
            {
                "date": day,
                "rows": int(len(subset)),
                "expiries": int(subset["expiry"].nunique()),
            }
        )
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "akshare_version": akshare.__version__,
        "artifacts": {
            "data/clean/options_io.csv": build_artifact(
                CLEAN / "options_io.csv",
                f"CFFEX IO daily quotes; CSI 300 close from {csi_source}",
                "http://www.cffex.com.cn/sj/hqsj/rtj/ + akshare index interfaces",
                OPTION_COLUMNS,
                len(options),
            ),
            "data/clean/shibor.csv": build_artifact(
                CLEAN / "shibor.csv",
                "China Shibor",
                "akshare.macro_china_shibor_all",
                SHIBOR_COLUMNS,
                len(shibor),
            ),
            "data/clean/sina_comparison.csv": build_artifact(
                CLEAN / "sina_comparison.csv",
                "CFFEX close vs Sina option daily close",
                "akshare.option_cffex_hs300_daily_sina",
                SINA_COLUMNS,
                len(sina),
            ),
        },
        "interfaces": INTERFACES,
        "csi_source": csi_source,
        "build_stats": build_stats,
        "sample_dates": sample_dates,
        "sample_month_report": month_report,
        "sample_date_report": date_report,
        "assertions": assertions,
    }
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return month_report, date_report


def print_report(month_report, date_report, sina, build_stats, assertions):
    month_table = Table(title="Sample months")
    month_table.add_column("Month")
    month_table.add_column("Trading days", justify="right")
    month_table.add_column("Clean rows", justify="right")
    for item in month_report:
        month_table.add_row(item["month"], str(item["trading_days"]), str(item["rows"]))
    console.print(month_table)

    date_table = Table(title="Sample dates")
    date_table.add_column("Date")
    date_table.add_column("Rows", justify="right")
    date_table.add_column("Expiries", justify="right")
    for item in date_report:
        date_table.add_row(item["date"], str(item["rows"]), str(item["expiries"]))
    console.print(date_table)

    failures = [item for item in INTERFACES if item["status"] in {"failed", "partial"}]
    interface_table = Table(title="Interface failures")
    interface_table.add_column("Interface")
    interface_table.add_column("Status")
    interface_table.add_column("Detail")
    if failures:
        for item in failures:
            interface_table.add_row(item["interface"], item["status"], item["detail"])
    else:
        interface_table.add_row("none", "-", "-")
    console.print(interface_table)

    ok = int((sina["status"] == "ok").sum()) if not sina.empty else 0
    failed = int((sina["status"] == "failed").sum()) if not sina.empty else 0
    no_match = int((sina["status"] == "no_matching_date").sum()) if not sina.empty else 0
    console.print(
        f"Sina comparison: ok={ok} failed={failed} no_matching_date={no_match} total={len(sina)}"
    )

    assertion_table = Table(title="Assertions")
    assertion_table.add_column("Check")
    assertion_table.add_column("Result")
    assertion_table.add_column("Detail")
    for item in assertions:
        assertion_table.add_row(item["name"], "PASS" if item["passed"] else "FAIL", item["detail"])
    console.print(assertion_table)

    validation = build_stats.get("expiry_validation", {})
    if validation.get("rolled") or validation.get("unverified_future"):
        flag_table = Table(title="Expiry validation")
        flag_table.add_column("Nominal")
        flag_table.add_column("Resolved")
        flag_table.add_column("Status")
        for item in validation.get("rolled", []):
            flag_table.add_row(item["nominal"], item["expiry"], "rolled_holiday")
        for expiry in validation.get("unverified_future", []):
            flag_table.add_row(expiry, expiry, "unverified_future")
        console.print(flag_table)


def online():
    RAW.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    import akshare as ak

    session = requests.Session()
    fetch_cffex(session)
    fetch_csi300()
    fetch_ak_csv("Shibor", RAW / "shibor.csv", ak.macro_china_shibor_all)
    fetch_ak_csv("LPR", RAW / "lpr.csv", ak.macro_china_lpr)
    fetch_ak_csv("SSE option current day", RAW / "option_current_day_sse.csv", ak.option_current_day_sse)
    fetch_etf("510050")
    fetch_etf("510300")

    csi, csi_source = load_csi()
    options, build_stats = build_options(csi)
    shibor = build_shibor()
    options.to_csv(CLEAN / "options_io.csv", index=False, encoding="utf-8")
    shibor.to_csv(CLEAN / "shibor.csv", index=False, encoding="utf-8")
    sample_dates = choose_sample_dates(options)
    sina = fetch_sina_comparison(options, sample_dates)
    sina.to_csv(CLEAN / "sina_comparison.csv", index=False, encoding="utf-8")
    assertions = run_assertions(options, shibor, sample_dates, csi)
    month_report, date_report = write_metadata(
        options, shibor, sina, csi_source, build_stats, sample_dates, assertions
    )
    print_report(month_report, date_report, sina, build_stats, assertions)
    failed = [item["name"] for item in assertions if not item["passed"]]
    futures_results, futures_assertions = fetch_futures()
    update_futures_metadata(futures_results, futures_assertions)
    print_futures_report(futures_results, futures_assertions)
    failed += [
        f"{item['dataset']} futures"
        for item in futures_assertions
        if not (
            item["at_least_5_years"]
            and item["no_duplicate_dates"]
            and item["positive_prices"]
        )
    ]
    if failed:
        log.error("assertion failures: %s", failed)
        return 1
    log.info("online fetch complete; all assertions passed")
    return 0


def futures_only():
    RAW.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    futures_results, futures_assertions = fetch_futures()
    update_futures_metadata(futures_results, futures_assertions)
    print_futures_report(futures_results, futures_assertions)
    failed = [
        f"{item['dataset']} futures"
        for item in futures_assertions
        if not (
            item["at_least_5_years"]
            and item["no_duplicate_dates"]
            and item["positive_prices"]
        )
    ]
    if failed:
        log.error("assertion failures: %s", failed)
        return 1
    log.info("futures fetch complete; all assertions passed")
    return 0


def offline():
    if not METADATA.exists():
        log.error("metadata missing: %s", METADATA)
        return 1
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    artifacts = metadata.get("artifacts", {})
    if not artifacts:
        log.error("metadata contains no artifacts")
        return 1
    failed = []
    for relative, info in artifacts.items():
        path = ROOT / relative
        if not path.exists():
            log.error("OFFLINE missing %s", relative)
            failed.append(relative)
            continue
        actual = sha256(path)
        expected = info.get("sha256")
        if actual != expected:
            log.error("OFFLINE checksum mismatch %s expected=%s actual=%s", relative, expected, actual)
            failed.append(relative)
        else:
            log.info("OFFLINE OK %s rows=%s sha256=%s", relative, info.get("rows"), actual)
    if failed:
        return 1
    log.info("offline verification complete; all checksums match")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Q1 option, index and rates data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--online", action="store_true", help="download missing raw data and rebuild clean artifacts")
    group.add_argument("--offline", action="store_true", help="verify clean artifacts against metadata checksums")
    group.add_argument("--futures", action="store_true", help="fetch or refresh only the Q2 futures snapshots")
    group.add_argument("--q3-fetch", action="store_true", help="fetch only the Q3 macro raw snapshots")
    group.add_argument("--q3", action="store_true", help="fetch and build the Q3 macro panel and metadata")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.online:
        return online()
    if args.offline:
        return offline()
    if args.q3_fetch:
        fetch_q3_raw()
        return 0
    if args.q3:
        return q3_run()
    return futures_only()


if __name__ == "__main__":
    sys.exit(main())
