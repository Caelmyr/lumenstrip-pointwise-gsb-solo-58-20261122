"""Anchor-line registration between repeat photographs of the same lamp.

The user marks spectral lines that are known not to have changed.  Each anchor
only describes the *local* registration offset between the two camera frames;
it never enters the signal comparison itself, so a line that genuinely changed
width cannot be smoothed away as registration error.

The displacement field is kept deliberately conservative:

* anchors that agree on direction collapse to a single robust constant shift;
* when the end anchors demand materially different shifts a linear field is
  fitted, but it is only trusted *between* the anchor hull (interpolation);
* positions outside that hull have no evidence and are marked pending check.

The unregistered comparison is always retained beside the corrected one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import NamedWindow, Problem, allowed, flag, mapping, plain_text, real, rounded, seq, whole
from .result import Result
from .strip import contiguous, regrid
from .view import AMBER, BLUE, CYAN, GREEN, PINK, Region, Series, palette, plot, strip_overlay

MEASURED = "measured"
PENDING = "pending-check"
UNSUPPORTED = "unsupported source"


@dataclass(frozen=True)
class RegistrationAnchor:
    name: str
    column_a: float
    column_b: float

    @classmethod
    def parse(cls, value, index):
        data = mapping(value, "Registration anchor")
        allowed(data, {"name", "source_1_column", "source_2_column"}, "registration anchor")
        first = real(data.get("source_1_column"), "First image line column", 0.0005, 0.9995)
        second = real(data.get("source_2_column"), "Second image line column", 0.0005, 0.9995)
        name = plain_text(data.get("name", f"Anchor line {index + 1}"), "Anchor name", 58)
        return cls(name, first, second)


def _axis_at_u(trace, column):
    """Axis value of a relative image column under that source's own ruler."""
    if trace.fit is not None:
        values, supported = trace.fit.mapped(np.asarray([column], dtype=float))
        if not bool(supported[0]):
            raise Problem("A marked anchor column lies outside the saved wavelength ruler.")
        return float(values[0])
    return column * trace.plan.width - 0.5


def _local_shift(x, a, b, center, half_window, search, floor_correlation):
    """Best shift of b onto a near one anchor, via windowed cross correlation."""
    step = float(x[1] - x[0])
    i = int(np.argmin(np.abs(x - center)))
    lo, hi = max(0, i - half_window), min(len(x) - 1, i + half_window + 1)
    if hi - lo < 5:
        return None
    ua, ub = a[lo:hi], b[lo:hi]

    def score(grid_shift):
        if grid_shift >= 0:
            moved = ub[:-grid_shift] if grid_shift else ub
            reference = ua[grid_shift:]
        else:
            moved = ub[-grid_shift:]
            reference = ua[:grid_shift]
        keep = np.isfinite(moved) & np.isfinite(reference)
        if int(keep.sum()) < 5:
            return None
        p, q = reference[keep], moved[keep]
        cp, cq = p - p.mean(), q - q.mean()
        denom = np.linalg.norm(cp) * np.linalg.norm(cq)
        if denom <= 1e-12:
            return None
        return float(np.dot(cp, cq) / denom), float(np.mean(np.abs(cp - cq)))

    best = None
    for grid_shift in range(-search, search + 1):
        rated = score(grid_shift)
        if rated is None:
            continue
        correlation, residual = rated
        if best is None or (correlation, -residual) > (best["r"], -best["residual"]):
            best = {"grid_shift": grid_shift, "r": correlation, "residual": residual}
    if best is None or best["r"] < floor_correlation:
        return None
    grid_shift = best["grid_shift"]
    refined = float(grid_shift)
    if -search < grid_shift < search:
        scores = []
        for delta in (-1, 0, 1):
            rated = score(grid_shift + delta)
            scores.append(np.nan if rated is None else rated[0])
        if all(np.isfinite(v) for v in scores):
            y0, y1, y2 = scores
            denominator = y0 - 2 * y1 + y2
            if abs(denominator) > 1e-12:
                offset = 0.5 * (y0 - y2) / denominator
                if -1 < offset < 1:
                    refined = grid_shift + offset
    return {"grid_shift": grid_shift, "subgrid_shift": refined, "axis_shift": refined * step,
            "step": step, "correlation": rounded(best["r"]), "residual": rounded(best["residual"]),
            "window": [rounded(x[lo]), rounded(x[hi - 1])]}


def _fit_field(anchor_rows, x, step, disagreement):
    """Build the conservative displacement field and its trust hull."""
    trusted = [row for row in anchor_rows if row["accepted"]]
    positions = np.asarray([row["anchor_axis"] for row in trusted], dtype=float)
    shifts = np.asarray([row["measured_shift"] for row in trusted], dtype=float)
    hull_lo, hull_hi = float(positions.min()), float(positions.max())
    spread = float(shifts.max() - shifts.min())
    linear = len(trusted) >= 2 and spread > disagreement * step
    if linear:
        design = np.column_stack((positions - positions.mean(), np.ones(len(positions))))
        slope, intercept = np.linalg.lstsq(design, shifts, rcond=None)[0]
        field = intercept + slope * (x - positions.mean())
        model = "linear drift between anchors"
    else:
        slope = 0.0
        field = np.full(len(x), float(np.median(shifts)))
        model = "constant common shift"
    inside = (x >= hull_lo) & (x <= hull_hi)
    return field, {"model": model, "linear": linear, "slope_per_axis": rounded(slope),
                   "hull": [rounded(hull_lo), rounded(hull_hi)],
                   "anchor_shift_spread": rounded(spread), "inside": inside,
                   "disagreement_limit": rounded(disagreement * step)}


def _warp(x, curve, field_grid):
    """Resample the second trace onto x after removing the displacement field."""
    aligned = np.full(len(x), np.nan)
    for run in contiguous(np.isfinite(curve)):
        lo, hi = run.start, run.stop - 1
        for j in range(lo, hi + 1):
            source_index = j - field_grid[j]
            if not (lo <= source_index <= hi):
                continue
            k = int(np.floor(source_index))
            frac = source_index - k
            if k + 1 <= hi and np.isfinite(curve[k]) and np.isfinite(curve[k + 1]):
                aligned[j] = curve[k] * (1 - frac) + curve[k + 1] * frac
            elif np.isfinite(curve[k]):
                aligned[j] = curve[k]
    return aligned


def _watch_rows(windows, x, raw_delta, corrected, states, unit):
    rows = []
    for zone in windows:
        lo, hi = zone.interval.lower, zone.interval.upper
        inside = (x >= lo) & (x <= hi)
        checked = inside & (states == MEASURED)
        sample = corrected[checked]
        raw_sample = raw_delta[checked]
        rows.append({"band": zone.name, "range": zone.interval.public(), "axis_units": unit,
                     "checked_samples": int(checked.sum()),
                     "pending_samples": int((inside & (states == PENDING)).sum()),
                     "raw_abs_median": rounded(np.median(np.abs(raw_sample))) if len(raw_sample) else None,
                     "residual_abs_median": rounded(np.median(np.abs(sample))) if len(sample) else None,
                     "residual_extreme": rounded(np.max(np.abs(sample))) if len(sample) else None})
    return rows


def anchor_registration(bench):
    one, two = bench.require(2, 2)
    bench.same_basis()
    cfg = bench.order.settings
    allowed(cfg, {"anchors", "count", "window", "range", "min_correlation",
                  "disagreement", "per_ms", "bands"}, "anchor registration")
    count = whole(cfg.get("count", 380), "Common grid samples", 80, 1200)
    half_window = whole(cfg.get("window", 14), "Anchor correlation half window", 4, 120)
    search = whole(cfg.get("range", 8), "Shift search either side (samples)", 1, 60)
    floor_correlation = real(cfg.get("min_correlation", 0.55), "Minimum anchor correlation", 0.05, 0.98)
    disagreement = whole(cfg.get("disagreement", 2), "Anchor direction disagreement (samples)", 1, 40)
    per_ms = flag(cfg.get("per_ms", False), "Normalize by exposure")
    raw_anchors = seq(cfg.get("anchors", ()), "Unchanged anchor lines", 2, 4)
    anchors = [RegistrationAnchor.parse(value, i) for i, value in enumerate(raw_anchors)]
    watch = [NamedWindow.read(v, i) for i, v in
             enumerate(seq(cfg.get("bands", []), "Watched change intervals", 0, 8))]

    x, curves = regrid((one.trace, two.trace), count)
    if per_ms:
        curves = [curve / source.exposure for source, curve in zip((one, two), curves, strict=True)]
    a, b = curves
    step = float(x[1] - x[0])
    unit = one.trace.units

    anchor_rows = []
    for anchor in anchors:
        axis_a = _axis_at_u(one.trace, anchor.column_a)
        axis_b = _axis_at_u(two.trace, anchor.column_b)
        center = 0.5 * (axis_a + axis_b)
        if not x[0] - step <= center <= x[-1] + step:
            raise Problem(f"Anchor '{anchor.name}' is outside the common measured axis.")
        measured = _local_shift(x, a, b, center, half_window, search, floor_correlation)
        common = {"name": anchor.name, "source_1_column": anchor.column_a,
                  "source_2_column": anchor.column_b, "source_1_axis": rounded(axis_a),
                  "source_2_axis": rounded(axis_b), "anchor_axis": rounded(center)}
        if measured is None:
            anchor_rows.append({**common, "accepted": False, "measured_shift": 0.0,
                                "correlation": None, "state": "weak evidence - excluded"})
            continue
        anchor_rows.append({**common, "accepted": True, "measured_shift": measured["axis_shift"],
                            "grid_shift": measured["grid_shift"],
                            "subgrid_shift": rounded(measured["subgrid_shift"]),
                            "correlation": measured["correlation"],
                            "match_residual": measured["residual"],
                            "match_window": measured["window"], "state": "used in field"})
    trusted = [row for row in anchor_rows if row["accepted"]]
    if not trusted:
        raise Problem("No anchor line matched strongly enough. Widen the window or lower the correlation floor.")

    field, fit = _fit_field(anchor_rows, x, step, disagreement)
    aligned_b = _warp(x, b, field / step)
    raw_delta = b - a
    corrected = aligned_b - a

    support_a = np.isfinite(a)
    support_aligned = np.isfinite(aligned_b)
    states = np.full(len(x), MEASURED, dtype=object)
    states[~(support_a & support_aligned)] = UNSUPPORTED
    states[(states == MEASURED) & ~fit["inside"]] = PENDING

    rows = []
    for j, value in enumerate(x):
        k1 = one.trace.interval(value)
        k2 = two.trace.interval(value + field[j])
        rows.append({"position": rounded(value), "units": unit, "state": states[j],
                     "applied_shift": rounded(field[j]),
                     "signal_1": rounded(a[j]), "signal_2_raw": rounded(b[j]),
                     "signal_2_aligned": rounded(aligned_b[j]),
                     "raw_difference": rounded(raw_delta[j]),
                     "corrected_residual": rounded(corrected[j]),
                     "source_1_uv": one.trace.row(k1)["source_uv"] if support_a[j] and k1 is not None else None,
                     "source_2_uv": two.trace.row(k2)["source_uv"]
                     if support_aligned[j] and k2 is not None else None})

    measured_mask = states == MEASURED
    pending_count = int((states == PENDING).sum())
    result = Result("anchor-registration", "Anchor-line registration correction",
                    "Unchanged narrow lines measure the local frame displacement. The raw difference is kept; "
                    "only positions inside the anchor hull carry corrected residuals, elsewhere evidence is pending.")
    result.metrics = {"axis": unit, "grid_samples": count, "grid_spacing": rounded(step),
                      "field_model": fit["model"], "anchors_total": len(anchors),
                      "anchors_accepted": len(trusted), "anchors_excluded": len(anchors) - len(trusted),
                      "anchor_hull": fit["hull"], "anchor_shift_spread": fit["anchor_shift_spread"],
                      "disagreement_limit": fit["disagreement_limit"],
                      "pending_check_samples": pending_count,
                      "unsupported_samples": int((states == UNSUPPORTED).sum()),
                      "corrected_samples": int(measured_mask.sum()),
                      "raw_abs_median_inside": rounded(np.median(np.abs(raw_delta[measured_mask])))
                      if measured_mask.any() else None,
                      "residual_abs_median_inside": rounded(np.median(np.abs(corrected[measured_mask])))
                      if measured_mask.any() else None,
                      "exposure_normalized": per_ms,
                      "anchor_evidence": [{k: row[k] for k in
                                           ("name", "accepted", "measured_shift", "correlation", "state")}
                                          for row in anchor_rows]}
    # The browsable table is the per-point comparison; anchor evidence is a CSV.
    result.rows = rows

    residual_graph = plot(
        "Difference after anchor-line correction", unit,
        "Signal / ms difference" if per_ms else "Signal difference",
        [Series("Corrected residual (anchor hull)", x, np.where(states == MEASURED, corrected, np.nan), CYAN),
         Series("Pending check - no anchor evidence", x, np.where(states == PENDING, corrected, np.nan), AMBER),
         Series("Raw unregistered difference", x,
                np.where(support_a & np.isfinite(b), raw_delta, np.nan), PINK)],
        horizontal=0.0, regions=[Region(w.interval.lower, w.interval.upper, w.name, GREEN) for w in watch])
    field_graph = plot(
        "Registration displacement field", unit, f"Shift ({unit})",
        [Series("Applied shift", x, np.where(fit["inside"], field, np.nan), BLUE),
         Series("Pending extrapolation", x, np.where(~fit["inside"], field, np.nan), AMBER),
         Series("Anchor measurements", np.asarray([r["anchor_axis"] for r in trusted]),
                np.asarray([r["measured_shift"] for r in trusted]), CYAN)],
        markers=[(r["anchor_axis"], r["measured_shift"], r["name"][:9]) for r in trusted], horizontal=0.0)
    overlay_graph = plot(
        "Traces on the common axis", unit, "Signal / ms" if per_ms else "Linear signal",
        [Series(one.title, x, a, palette(0)), Series(two.title + " (raw)", x, b, palette(1)),
         Series(two.title + " (aligned)", x, aligned_b, GREEN)])
    # Frozen paired-axis probes: the applied field is stored with the chart so a
    # retained reading resolves both original positions without re-fitting.
    probe_payload = {"kind": "paired-axis", "source_index": 0, "source_index_2": 1,
                     "axis_grid": [rounded(v) for v in x.tolist()],
                     "applied_shift": [rounded(v) for v in field.tolist()]}
    result.chart("registration-residual", "Corrected residual with pending-check region", residual_graph)
    result.chart("registration-field", "Measured anchor shifts and fitted displacement field", field_graph)
    result.chart("registration-overlay", "Source traces before and after alignment", overlay_graph)
    for name in ("registration-residual.png", "registration-field.png", "registration-overlay.png"):
        # The frozen field wins over the chart's generic axis probe metadata.
        result.probes[name] = {**result.probes[name], **probe_payload}
    result.image("source-1-strip", f"{one.title}: native camera strip",
                 strip_overlay(one.camera, one.trace), {"kind": "source", "source_index": 0})
    result.image("source-2-strip", f"{two.title}: native camera strip",
                 strip_overlay(two.camera, two.trace), {"kind": "source", "source_index": 1})
    result.csv("registration-anchors", "Anchor evidence and local shifts", anchor_rows)
    result.csv("registration-points", "Raw and corrected differences with native lookup points", rows)
    if watch:
        watch_rows = _watch_rows(watch, x, raw_delta, corrected, states, unit)
        result.csv("registration-watch", "Residual summary inside watched change intervals", watch_rows)
        result.metrics["watch_bands"] = [r["band"] for r in watch_rows]

    if len(trusted) < len(anchors):
        result.warnings.append(
            "Weak anchor evidence was excluded from the field; widen the correlation window or re-mark that line.")
    if fit["linear"]:
        result.warnings.append(
            "The end anchors imply different shifts: a linear field is interpolated between them and the "
            "regions outside the anchor hull are marked pending check rather than extrapolated.")
    if pending_count:
        result.warnings.append(
            f"{pending_count} grid positions lie outside the anchor hull and are reported as pending check.")
    if any(abs(row["measured_shift"]) > search * step * 0.9 for row in trusted):
        result.warnings.append("An anchor shift sits at the search limit; widen the search range to confirm it.")
    return result
