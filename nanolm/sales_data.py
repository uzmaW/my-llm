"""Business sales dataset helpers.

Auto-downloads a public *business-sales transaction* CSV (seaborn's ``taxis``:
6,433 real revenue transactions with timestamps, payment type, distance and a
``total`` amount) and exposes:

* :func:`load_sales` -- normalised records,
* :func:`records_to_corpus` -- a compact, char-level **modeling corpus** that
  teaches the nanolm transformer the joint distribution of
  ``amount | month | hour | payment-code | distance`` for each sale,
* :func:`analytics` / :func:`monthly_series` / :func:`next_month_forecast` --
  plain analytics + a transparent (non-learned) monthly revenue forecast.

Line schema (one per sale, newline terminated)::

    <amount>|<day-of-month>|<hour>|<payment_code>|<distance>

e.g. ``13|23|20|1|2`` == a \$13 sale on the 23rd at 20:00 paid by card over 2 mi.
"""

from __future__ import annotations

import csv
import os
import urllib.request
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

DEFAULT_URL = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
)
DEFAULT_PATH = os.path.join("datasets", "sales.csv")

_PAY_CODES: Dict[str, int] = {
    "credit card": 1,
    "cash": 2,
    "dispute": 3,
    "no charge": 4,
    "prcard": 5,
    "prcard ": 5,
    "unknown": 6,
}


# ---------------------------------------------------------------------- #
# Download / load
# ---------------------------------------------------------------------- #
def ensure_sales_csv(path: str = DEFAULT_PATH, url: str = DEFAULT_URL) -> str:
    """Download the dataset if it is missing and return its local path."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _parse_dt(value: str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _to_float(value) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def load_sales(path: str = DEFAULT_PATH) -> List[dict]:
    """Load the CSV into normalised records."""
    records: List[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dt = _parse_dt(row.get("pickup") or row.get("date") or row.get("Order Date") or "")
            amount = _to_float(row.get("total") or row.get("Sales") or row.get("amount"))
            if amount != amount:  # NaN guard
                continue
            distance = _to_float(row.get("distance") or row.get("Quantity") or 0.0)
            if distance != distance:
                distance = 0.0
            payment = (row.get("payment") or row.get("Payment Mode") or "unknown").strip().lower()
            color = (row.get("color") or row.get("Category") or "").strip().lower()
            records.append(
                {
                    "dt": dt,
                    "amount": amount,
                    "distance": distance,
                    "payment": payment or "unknown",
                    "color": color,
                }
            )
    return records


# ---------------------------------------------------------------------- #
# Modeling corpus
# ---------------------------------------------------------------------- #
def _pay_code(payment: str) -> int:
    return _PAY_CODES.get(payment, 6)


def record_to_line(rec: dict) -> str:
    amt = max(1, min(999, int(round(rec["amount"]))))
    day = rec["dt"].day if rec["dt"] else 0
    hour = rec["dt"].hour if rec["dt"] else 0
    dist = max(0, min(99, int(round(rec["distance"]))))
    return f"{amt}|{day}|{hour}|{_pay_code(rec['payment'])}|{dist}\n"


def records_to_corpus(records: List[dict]) -> str:
    return "".join(record_to_line(r) for r in records)


# ---------------------------------------------------------------------- #
# Analytics + (transparent) forecast
# ---------------------------------------------------------------------- #
def daily_series(records: List[dict]) -> List[Tuple[str, float]]:
    agg: Dict[str, float] = defaultdict(float)
    for r in records:
        if r["dt"] is None:
            continue
        agg[r["dt"].strftime("%Y-%m-%d")] += r["amount"]
    return sorted(agg.items())


def hours_of_day(records: List[dict]) -> List[Tuple[int, float]]:
    agg: Dict[int, float] = defaultdict(float)
    for r in records:
        if r["dt"] is None:
            continue
        agg[r["dt"].hour] += r["amount"]
    return [(h, agg.get(h, 0.0)) for h in range(24)]


def analytics(records: List[dict]) -> dict:
    n = len(records)
    total = sum(r["amount"] for r in records)
    by_payment: Dict[str, float] = defaultdict(float)
    for r in records:
        by_payment[r["payment"]] += r["amount"]
    return {
        "n": n,
        "total_revenue": total,
        "avg_ticket": (total / n) if n else 0.0,
        "by_payment": sorted(by_payment.items(), key=lambda kv: -kv[1]),
        "daily": daily_series(records),
        "hours": hours_of_day(records),
    }


def next_period_forecast(records: List[dict]) -> float:
    """OLS linear-trend extrapolation of the daily-revenue series.

    Deliberately simple and fully transparent so the number is explainable --
    it is an *analytic* estimate, not a learned LM prediction.
    """
    series = daily_series(records)
    ys = [v for _, v in series]
    if not ys:
        return 0.0
    if len(ys) < 2:
        return ys[0]
    xs = list(range(len(ys)))
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return b * len(xs) + (my - b * mx)