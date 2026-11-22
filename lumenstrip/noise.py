"""Inspect native sampling variability and real camera support."""

from dataclasses import dataclass

import numpy as np

from .contracts import NamedWindow, Problem, allowed, flag, real, rounded, seq, whole
from .result import Result
from .strip import SampleBlock, contiguous
from .view import AMBER, BLUE, CYAN, GREEN, PINK, Region, Series, plot, strip_overlay


@dataclass(frozen=True)
class SegmentStat:
    total: int
    measured: int
    spread: float | None
    signal: float | None
    uncertainty_proxy: float | None
    largest: float | None
    reads_per_column: float | None

    def public(self):
        return {"total_columns": self.total, "read_columns": self.measured,
                "median_dispersion": rounded(self.spread), "median_signal": rounded(self.signal),
                "median_dispersion_over_signal": rounded(self.uncertainty_proxy),
                "largest_dispersion": rounded(self.largest), "typical_native_pixels": rounded(self.reads_per_column)}


def _measure(t, mask, minimum):
    observed = mask & t.known & np.isfinite(t.deviation) & (t.middle_count >= minimum)
    subset = t.deviation[observed]
    values = t.signal[observed]
    used = t.middle_count[observed]
    if not len(subset):
        return SegmentStat(int(mask.sum()), 0, None, None, None, None, None)
    available = np.abs(values) > .0001
    ratio = subset[available] / np.abs(values[available])
    return SegmentStat(int(mask.sum()), len(subset), float(np.median(subset)), float(np.median(values)),
                       float(np.median(ratio)) if len(ratio) else None,
                       float(np.max(subset)), float(np.median(used)))


def noise_windows(bench):
    s = bench.one()
    t = s.trace
    req = bench.order.settings
    allowed(req, {"bands", "minimum"}, "variability")
    limit = whole(req.get("minimum", 3), "Minimum native pixels", 1, 32)
    axis = t.axis[np.isfinite(t.axis)]
    if len(axis) < 2:
        raise Problem("The trace has fewer than two mapped columns.")
    intervals = req.get("bands") or [{"name": "All measured", "interval": [float(axis.min()), float(axis.max())]}]
    bands = [NamedWindow.read(v, i) for i, v in enumerate(seq(intervals, "Variability windows", 1, 12))]
    rows = []
    for item in bands:
        low, high = item.interval.lower, item.interval.upper
        mask = (t.axis >= low) & (t.axis <= high)
        result = _measure(t, mask, limit)
        rows.append({"band": item.name, "range": item.interval.public(), **result.public()})
    measured = np.isfinite(t.axis) & t.known & (t.middle_count >= limit)
    ratio = np.divide(t.deviation, np.abs(t.signal), out=np.full(len(t.axis), np.nan),
                      where=measured & (np.abs(t.signal) > .0001))
    report = Result("strip-noise", "Spatial variability within the strip",
                    "Within-column deviations expose uneven or insufficient native samples. They are descriptive, not uncertainty guarantees.")
    report.rows = rows
    report.metrics = {"defined_bands": len(rows), "axis_units": t.units,
                      "minimum_native_samples": limit,
                      "columns_meeting_floor": int(measured.sum())}
    report.chart("spatial-noise", "Native dispersion beside the measured light",
                 plot("Strip variability", t.units, "Median absolute deviation",
                      [Series("Native dispersion", t.axis, np.where(measured, t.deviation, np.nan)),
                       Series("Measured light", t.axis, np.where(measured, t.signal, np.nan), BLUE)],
                      regions=[Region(z.interval.lower, z.interval.upper, z.name, AMBER) for z in bands]))
    report.csv("spatial-noise-summary", "Variability over named intervals", rows)
    report.csv("spatial-noise-values", "Column dispersion and number of supported pixels",
               [{"axis": rounded(t.axis[i]), "column": int(t.plan.x[i]), "ratio": rounded(ratio[i]),
                 "spread": rounded(t.deviation[i]) if measured[i] else None, "samples": int(t.middle_count[i])}
                for i in range(len(t.axis))])
    return report


def _side_read(camera, trace):
    side = trace.plan.adjacent
    settings = trace.config
    top, bottom = np.hsplit(side, 2)
    readings = []
    for part in (top, bottom):
        out = SampleBlock.read(camera, trace.plan.x, part, settings)
        valid = out.allowed
        count = valid.sum(axis=1)
        average = np.divide(np.where(valid, out.light, 0.).sum(axis=1), count,
                            out=np.full(len(count), np.nan), where=count > 0)
        readings.append((average, count))
    return readings


def sidebands(bench):
    s = bench.one()
    t = s.trace
    req = bench.order.settings
    allowed(req, {"limit", "minimum", "normalize"}, "sideband agreement")
    guard = real(req.get("limit", .026), "Recommended sideband level difference", .0001, 2)
    floor = whole(req.get("minimum", 2), "Minimum background pixels on each side", 1, 32)
    normalize = flag(req.get("normalize", False), "Divide by exposure time")
    divider = s.exposure if normalize else 1.
    (upper, up_count), (lower, low_count) = _side_read(s.camera, t)
    upper /= divider
    lower /= divider
    matched = (up_count >= floor) & (low_count >= floor) & np.isfinite(t.axis)
    observed = np.where(matched, lower - upper, np.nan)
    clips = matched & (np.abs(observed) >= guard)
    changes = []
    for run in contiguous(clips):
        i, j = run.start, run.stop - 1
        diff = observed[run]
        changes.append({"axis_start": rounded(t.axis[i]), "axis_end": rounded(t.axis[j]),
                        "camera_columns": [int(t.plan.x[i]), int(t.plan.x[j])],
                        "largest_relative_difference": rounded(np.max(np.abs(diff))),
                        "positive_span": int((diff > 0).sum()), "negative_span": int((diff < 0).sum())})
    rows = [{"axis": rounded(v), "camera_column": int(t.plan.x[i]),
             "upper_signal": rounded(upper[i]), "lower_signal": rounded(lower[i]),
             "upper_support": int(up_count[i]), "lower_support": int(low_count[i]),
             "sideband_delta": rounded(observed[i])} for i, v in enumerate(t.axis)]
    result = Result("sideband-check", "Agreement of the background strips",
                    "The upper and lower sidebands are measured separately to expose asymmetric background.")
    result.rows = changes
    result.metrics = {"paired_columns": int(matched.sum()), "above_difference": int(clips.sum()),
                      "unpaired_columns": int(len(t.axis) - matched.sum()), "axis": t.units,
                      "normalized_by_exposure": normalize}
    result.chart("sideband-difference", "Measured background agreement", plot(
        "Sideband agreement", t.units, "Signal / ms" if normalize else "Linear signal",
        [Series("Upper", t.axis, np.where(up_count >= floor, upper, np.nan), BLUE),
         Series("Lower", t.axis, np.where(low_count >= floor, lower, np.nan), GREEN),
         Series("Lower minus upper", t.axis, observed, AMBER)], horizontal=0.))
    result.image("sideband-geometry", "Source locations of the two background strips", strip_overlay(s.camera, t),
                 {"kind": "source", "source_index": 0})
    result.csv("sideband-exceedance", "Runs above the difference guard", changes)
    result.csv("sideband-measurements", "Background evidence by original column", rows)
    return result
