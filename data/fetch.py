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


def rel_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return str(candidate)


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
            "path": rel_path(path),
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
    if failed:
        log.error("assertion failures: %s", failed)
        return 1
    log.info("online fetch complete; all assertions passed")
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.online:
        return online()
    if args.offline:
        return offline()


if __name__ == "__main__":
    raise SystemExit(main())
