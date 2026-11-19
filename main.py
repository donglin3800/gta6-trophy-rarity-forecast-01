"""GTA VI week-one trophy rarity forecaster.

Reads one case per run: the house's Exophase reads of the game's PS5 trophy
page before the case checkpoint, and writes five quantiles of each trophy's
earn rate at the case's target, as JSON to the output directory.

Method, per trophy, on the logit scale:
  * x = ln(1 + hours since the release-day start), y = logit(rate).
  * A recency-weighted, outlier-resistant straight-line fit of y on x.  The
    weights favour recent reads and reads taken when many players were
    tracked (early reads with a few hundred players are heavily quantised);
    a Huber pass downweights Exophase refresh steps, which move every
    trophy's rate at once and are not trend.
  * The slope is shrunk toward zero in proportion to its own standard
    error, so a trophy whose history is flat or noisy does not get an
    extrapolated rocket ride, and the forecast is re-anchored on the last
    read, so a zero slope means persistence, not a fall back to the
    historical mean.
  * Growth is damped by how much it has already slowed: the slope of the
    last 24 hours is compared with the slope over the whole history.  A
    trophy still climbing at its old pace keeps its full slope; one whose
    recent slope has fallen to four fifths of the old one keeps less than
    half of it, because launch-week adoption bends over as the first
    cohort finishes the early chapters and new buyers dilute the rates.
    The damping is read off the trophy's own history, so a trophy that
    has not slowed down is not damped at all.
  * Medians are then made coherent along the story chain (Chapter 1 >=
    Chapter 2 >= Chapter 3 >= Chapter 4 >= final mission >= 100%
    completion >= platinum), because earning a later story trophy
    requires the earlier ones, and the platinum requires all of them.
  * The spread is an honest error budget: a random-walk term that grows
    with the horizon, the slope's own uncertainty projected to the
    target, the fit's residual scale, and an allowance for one more
    Exophase refresh between checkpoint and target.

Only the case file is read; nothing depends on the wall clock, the
working directory or the network, so two runs on the same case write
the same bytes.  Standard library only.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE_START = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Story chain: each later trophy's earners are a subset of the earlier
# one's, so medians (and every quantile) must not rise along the chain.
# Ids: 3 Ch1, 4 Ch2, 5 Ch3, 6 Ch4, 7 final mission, 14 100% completion,
# 1 platinum (requires everything, so it sits below every trophy).
CHAIN = [3, 4, 5, 6, 7, 14]
PLATINUM_ID = 1

# Generous ceilings (percent) for the extrapolated median.  They only bind
# when a fit extrapolates far past anything the history supports; the
# story-chain caps do most of the work.
CEILINGS = {1: 5.0, 14: 8.0, 7: 45.0, 3: 95.0, 4: 95.0, 5: 95.0, 6: 95.0}

# Random-walk spread, in logit units per square-root day.
WALK_SIGMA = 0.35
# Allowance, in logit units, for one Exophase refresh step before target.
REFRESH_ALLOWANCE = 0.30
# Recency half-life for read weights, in hours.
RECENCY_TAU_HOURS = 36.0
# Largest logit rise the median may take above the last read's logit.
MAX_LOGIT_RISE = 4.0


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lo):
    if lo >= 20.0:
        return 99.99
    if lo <= -20.0:
        return 0.01
    return clamp_rate(100.0 / (1.0 + math.exp(-lo)))


def round4(x):
    """Round to 4 decimals, half up, like the house baseline does."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e01, 2.209460984245205e02,
         -2.759285984469534e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    seconds = (moment - RELEASE_START).total_seconds()
    return seconds / 3600.0


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, se_b, xbar, n)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (ybar, 0.0, 0.0, xbar, len(xs))
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(len(xs) - 2, 1)
    var = sum(w * r * r for w, r in zip(ws, resid)) / (sw * dof / max(len(xs), 1))
    se_b = math.sqrt(var / sxx) if sxx > 0 else 0.0
    return (a, b, se_b, xbar, len(xs))


def robust_scale(residuals):
    if not residuals:
        return 0.0
    ordered = sorted(abs(r) for r in residuals)
    mid = len(ordered) // 2
    mad = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return 1.4826 * mad


def fit_trophy(points, target_x, horizon_days):
    """points: list of (x, y, weight, t_hours). Returns (median_logit, spread_logit)."""
    if not points:
        return (logit(1.0), 1.2)
    if len(points) == 1:
        spread = max(0.6, WALK_SIGMA * math.sqrt(horizon_days))
        return (points[0][1], spread)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ws = [p[2] for p in points]
    fit = weighted_line(xs, ys, ws)
    if fit is None:
        return (ys[-1], 1.0)
    a, b, se_b, xbar, _n = fit
    # Huber pass: downweight reads the line misses badly (refresh steps).
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    scale = max(robust_scale(resid), 0.08)
    huber = []
    for w, r in zip(ws, resid):
        u = abs(r) / (1.345 * scale)
        huber.append(w / u if u > 1.0 else w)
    fit2 = weighted_line(xs, ys, huber)
    if fit2 is not None:
        a, b, se_b, xbar, _n = fit2
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    # Shrink the slope toward zero by its own uncertainty: a slope the
    # data cannot see is treated as no slope.
    if se_b > 0:
        shrink = (b * b) / (b * b + se_b * se_b)
    else:
        shrink = 1.0 if b != 0.0 else 0.0
    b = max(-2.0, min(4.0, b * shrink))
    # Damping from deceleration: compare the last 24 hours' slope with
    # the whole-history slope.  A ratio near 1 means growth has not
    # slowed, so keep the full slope; a ratio at or below 0.8 means it
    # has bent over, so keep less than half of it.
    damp = 1.0
    if b > 0.2 and len(points) >= 5:
        last_t = points[-1][3]
        recent = [p for p in points if last_t - p[3] <= 24.0]
        if len(recent) >= 3:
            rfit = weighted_line([p[0] for p in recent],
                                 [p[1] for p in recent],
                                 [p[2] for p in recent])
            if rfit is not None and rfit[1] > 0:
                ratio = rfit[1] / b
                if ratio <= 0.8:
                    damp = 0.45
                elif ratio >= 0.95:
                    damp = 1.0
                else:
                    damp = 0.45 + (ratio - 0.8) / 0.15 * 0.55
    resid_scale = min(0.8, max(0.08, robust_scale(resid)))
    x_last = xs[-1]
    y_last = ys[-1]
    median_logit = y_last + b * damp * (target_x - x_last)
    median_logit = min(median_logit, y_last + MAX_LOGIT_RISE)
    slope_var = (se_b * abs(target_x - xbar)) ** 2
    spread = math.sqrt(
        (WALK_SIGMA * math.sqrt(horizon_days)) ** 2
        + slope_var
        + resid_scale ** 2
        + REFRESH_ALLOWANCE ** 2
    )
    spread = min(1.6, max(0.45, spread))
    return (median_logit, spread)


def forecast_case(case):
    history = case.get("history") or []
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    target_x = math.log1p(max(hours_since_release(target), -0.9))
    last_read = history[-1] if history else None
    last_time = parse_time(last_read["at"]) if last_read else target
    last_hours = hours_since_release(last_time)

    medians = {}
    spreads = {}
    for idx, trophy in enumerate(trophies):
        tid = trophy["id"]
        points = []
        for read in history:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            moment = parse_time(read["at"])
            t = hours_since_release(moment)
            x = math.log1p(max(t, -0.9))
            y = logit(rates[idx])
            players = read.get("players_tracked") or 1000
            recency = math.exp(-max(last_hours - t, 0.0) / RECENCY_TAU_HOURS)
            size_weight = players / (players + 2000.0)
            points.append((x, y, recency * size_weight, t))
        if not points and last_read is not None:
            rates = last_read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                points.append((math.log1p(max(last_hours, -0.9)),
                               logit(rates[idx]), 1.0, last_hours))
        median_logit, spread = fit_trophy(points, target_x, horizon_days)
        if points:
            last_rate = None
            for read in reversed(history):
                rates = read.get("rates") or []
                if idx < len(rates) and rates[idx] is not None:
                    last_rate = rates[idx]
                    break
            if last_rate is not None and last_rate >= 99.5:
                # A trophy the page shows at (nearly) 100%: keep it there
                # with a modest spread, not a trend extrapolation.
                median_logit = logit(99.9)
                spread = 0.8
        ceiling = CEILINGS.get(tid)
        if ceiling is not None:
            median_logit = min(median_logit, logit(ceiling))
        medians[tid] = median_logit
        spreads[tid] = spread

    z_values = [norm_ppf(level) for level in case.get("levels") or LEVELS]
    rows = {}
    for trophy in trophies:
        tid = trophy["id"]
        centre = medians[tid]
        spread = spreads[tid]
        rows[tid] = [round4(inv_logit(centre + z * spread)) for z in z_values]
        # Keep each row non-decreasing after rounding.
        for k in range(1, len(rows[tid])):
            if rows[tid][k] < rows[tid][k - 1]:
                rows[tid][k] = rows[tid][k - 1]

    # Story-chain coherence, applied per quantile level: a later story
    # trophy cannot sit above the earlier one at the same quantile.
    for level_pos in range(len(z_values)):
        running = None
        for tid in CHAIN:
            if tid not in rows:
                continue
            if running is None:
                running = rows[tid][level_pos]
            else:
                running = min(running, rows[tid][level_pos])
                rows[tid][level_pos] = running
        if PLATINUM_ID in rows:
            floor_cap = min(rows[t][level_pos] for t in rows)
            rows[PLATINUM_ID][level_pos] = min(rows[PLATINUM_ID][level_pos], floor_cap)

    out_rows = [{"id": trophy["id"], "q": rows[trophy["id"]]} for trophy in trophies]
    return {"trophies": out_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
