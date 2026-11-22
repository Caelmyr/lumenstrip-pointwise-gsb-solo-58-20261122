"""Measurements for known broad bands, slopes and sampled transition edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import NamedWindow, Problem, Window, allowed, choice, flag, mapping, real, rounded, seq, whole
from .result import Result
from .spectral import area
from .strip import contiguous
from .view import AMBER, BLUE, CYAN, GREEN, PINK, Region, Series, palette, plot


@dataclass(frozen=True)
class BandShape:
    name: str
    interval: Window
    center: float | None
    peak: float | None
    quantiles: tuple
    edge_start: float | None
    edge_stop: float | None
    observed: float

    def row(self):
        return {"name": self.name, "bounds": self.interval.public(),
                "crest_axis": rounded(self.center), "crest_signal": rounded(self.peak),
                "integral_crossings": [rounded(v) for v in self.quantiles],
                "first_supported": rounded(self.edge_start), "last_supported": rounded(self.edge_stop),
                "measured_width": rounded(self.observed),
                "observed_fraction": rounded(self.observed / self.interval.width)}


def _sample_intervals(x, values, valid, clip, *, positive=False):
    spline = []
    for run in contiguous(valid & np.isfinite(values)):
        for i in range(run.start, run.stop - 1):
            u0, u1 = x[i:i + 2]
            a, b = max(u0, clip.lower), min(u1, clip.upper)
            if b <= a:
                continue
            heights = np.interp([a, b], [u0, u1], values[i:i + 2])
            y, z = heights
            if positive and y * z < 0:
                where = a - y / (z - y) * (b - a)
                if y < 0:
                    a, y = where, 0.
                else:
                    b, z = where, 0.
            weight = (b - a) * (y + z) / 2.
            if not positive or weight >= 0:
                spline.append((a, b, y, z, weight))
    return spline


def _integral_crossings(segments, fractions):
    weights = np.asarray([row[-1] for row in segments])
    keys = np.cumsum(weights)
    if not len(keys) or keys[-1] < 1e-12:
        return tuple(None for _ in fractions)
    answers = []
    for fraction in fractions:
        target = keys[-1] * fraction
        index = min(len(keys) - 1, int(np.searchsorted(keys, target, "left")))
        a, b, y, z, _ = segments[index]
        need = target - (keys[index - 1] if index else 0.)
        lo, hi = 0., 1.
        for _ in range(24):
            r = (lo + hi) / 2
            weight = (b - a) * (r * y + .5 * r ** 2 * (z - y))
            if weight > need:
                hi = r
            else:
                lo = r
        answers.append(a + (b - a) * (lo + hi) / 2.)
    return tuple(answers)


def _shape(trace, band, fractions):
    x, y, valid = trace.axis_array()
    intervals = _sample_intervals(x, y, valid, band.interval)
    if not intervals:
        return BandShape(band.name, band.interval, None, None, tuple(None for _ in fractions), None, None, 0.)
    upright = _sample_intervals(x, y, valid, band.interval, positive=True)
    keypoints = [(a, p) for a, _, p, _, _ in intervals] + [(b, q) for _, b, _, q, _ in intervals]
    center, crest = max(keypoints, key=lambda row: (row[1], -row[0]))
    return BandShape(band.name, band.interval, center, crest,
                     _integral_crossings(upright, fractions),
                     intervals[0][0], intervals[-1][1], sum(b - a for a, b, _, _, _ in intervals))


def _fallback_range(trace):
    axis, _, _ = trace.axis_array()
    if len(axis) < 2:
        raise Problem("At least two axis columns are required for this reading.")
    return [float(axis[0]), float(axis[-1])]


def _windows(raw, trace, label):
    items = raw or [{"name": "Whole trace", "interval": _fallback_range(trace)}]
    return [NamedWindow.read(item, i) for i, item in enumerate(seq(items, label, 1, 12))]


def envelopes(bench):
    s = bench.one()
    trace = s.trace
    req = bench.order.settings
    allowed(req, {"bands", "tail"}, "band shape")
    tail = real(req.get("tail", .1), "Area tail", .01, .4)
    bands = _windows(req.get("bands"), trace, "Shape windows")
    fractions = (tail, .5, 1. - tail)
    summary = [_shape(trace, band, fractions) for band in bands]
    rows = [line.row() for line in summary]
    report = Result("band-shape", "Broad spectral band shape",
                    "Area percentiles and crest positions describe the actually measured support of each selected band.")
    report.rows = rows
    report.metrics = {"shape_windows": len(rows), "axis_units": trace.units,
                      "measured_area_fractions": list(fractions),
                      "windows_with_support": sum(r["measured_width"] > 0 for r in rows)}
    report.chart("band-shape", "Crest and specified bands", plot(
        "Camera band shapes", trace.units, "Linear signal",
        [Series("Camera spectrum", trace.axis, np.where(trace.known, trace.signal, np.nan))],
        markers=[(b.center, b.peak, b.name) for b in summary if b.center is not None],
        regions=[Region(b.interval.lower, b.interval.upper, b.name, BLUE) for b in bands]))
    report.csv("band-shape", "Measured crest and area crossing values", rows)
    return report


@dataclass(frozen=True)
class Edge:
    begin: float
    end: float
    height: float
    direction: str
    strong_column: float

    def public(self):
        return {"first_axis": rounded(self.begin), "last_axis": rounded(self.end), "change": rounded(self.height),
                "direction": self.direction, "strongest_step_near": rounded(self.strong_column)}


def _edge_runs(x, y, mask, threshold, side):
    pairs = mask[:-1] & mask[1:] & np.isfinite(y[:-1]) & np.isfinite(y[1:])
    widths = x[1:] - x[:-1]
    gradient = np.divide(y[1:] - y[:-1], widths, out=np.zeros_like(widths), where=pairs & (widths > 0))
    results = []
    for direction, factor in (("rising", 1.), ("falling", -1.)):
        if side not in {"both", direction}:
            continue
        strong = pairs & (factor * gradient >= threshold)
        for sub in contiguous(strong):
            first = sub.start
            last = sub.stop
            strongest = first + int(np.argmax(factor * gradient[sub]))
            edge = Edge(float(x[first]), float(x[last]), float(y[last] - y[first]), direction,
                        float((x[strongest] + x[strongest + 1]) / 2))
            results.append(edge)
    return gradient, pairs, sorted(results, key=lambda e: e.begin)


def slope_edges(bench):
    source = bench.one()
    trace = source.trace
    request = bench.order.settings
    allowed(request, {"gradient", "direction", "window"}, "edge scan")
    threshold = real(request.get("gradient", .015), "Minimum slope per axis unit", .00005, 2)
    direction = choice(request.get("direction", "both"), ("both", "rising", "falling"), "Edge direction")
    bounds = Window.parse(request.get("window") or _fallback_range(trace))
    axis, samples, known = trace.axis_array()
    valid = known & (axis >= bounds.lower) & (axis <= bounds.upper)
    grad, support, candidates = _edge_runs(axis, samples, valid, threshold, direction)
    rows = [candidate.public() for candidate in candidates]
    center = (axis[1:] + axis[:-1]) / 2
    stat = Result("edge-scan", "Spectral transitions",
                  "Local changes are reported only between contiguous measured samples in the selected axis interval.")
    stat.rows = rows
    stat.metrics = {"total_transitions": len(rows), "rising": sum(e["direction"] == "rising" for e in rows),
                    "falling": sum(e["direction"] == "falling" for e in rows), "axis_units": trace.units,
                    "slope_guard": threshold, "covered_intervals": int(support.sum())}
    stat.chart("trace-gradients", "Inclines on the saved spectrum", plot(
        "Local spectral slope", trace.units, "Signal per axis unit",
        [Series("Measured derivative", center, np.where(support, grad, np.nan))],
        horizontal=0., regions=[Region(e.begin, e.end, e.direction,
                                       GREEN if e.direction == "rising" else PINK) for e in candidates]))
    stat.csv("trace-gradients", "Gradient threshold crossings", rows)
    stat.csv("derivative-values", "Adjacent supported slopes",
             [{"midpoint": rounded(mid), "incline": rounded(grad[i]) if support[i] else None,
               "measured": bool(support[i])} for i, mid in enumerate(center)])
    return stat


@dataclass(frozen=True)
class FlatSurvey:
    first: float
    last: float
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    measured: float
    limit: float

    def row(self):
        span = self.maximum - self.minimum if self.maximum is not None else None
        if self.median is not None and self.median > 1e-6:
            fraction = span / self.median
        else:
            fraction = None
        return {"lower_axis": rounded(self.first), "upper_axis": rounded(self.last),
                "supported_width": rounded(self.measured), "mean_level": rounded(self.mean),
                "middle_level": rounded(self.median), "lowest_level": rounded(self.minimum),
                "highest_level": rounded(self.maximum), "span_fraction_of_median": rounded(fraction),
                "within_limit": fraction <= self.limit if fraction is not None else None}


def _flat_intervals(trace, bounds: Window, pieces: int, limit: float):
    x, y, valid = trace.axis_array()
    limits = np.linspace(bounds.lower, bounds.upper, pieces + 1)
    for a, b in zip(limits[:-1], limits[1:], strict=True):
        interval = _sample_intervals(x, y, valid, Window(a, b))
        if not interval:
            yield FlatSurvey(a, b, None, None, None, None, 0., limit)
            continue
        raw = np.concatenate([np.asarray([y0, y1]) for _, _, y0, y1, _ in interval])
        widths = sum(hi - lo for lo, hi, _, _, _ in interval)
        integral = sum(total for _, _, _, _, total in interval)
        yield FlatSurvey(a, b, integral / widths, float(np.median(raw)),
                         float(np.min(raw)), float(np.max(raw)), widths, limit)


def flatness(bench):
    source = bench.one()
    trace = source.trace
    cfg = bench.order.settings
    allowed(cfg, {"window", "pieces", "limit"}, "flatness check")
    pieces = whole(cfg.get("pieces", 8), "Number of bands", 2, 60)
    maximum = real(cfg.get("limit", .22), "Maximum signal span / median", .002, 3.)
    capture = Window.parse(cfg.get("window") or _fallback_range(trace))
    output = list(_flat_intervals(trace, capture, pieces, maximum))
    rows = [item.row() for item in output]
    stat = Result("continuum-flatness", "Continuum uniformity",
                  "The chart shows local range and midpoint around the recorded continuum. Empty bands remain unresolved.")
    stat.metrics = {"bands": len(rows), "passed": sum(r["within_limit"] is True for r in rows),
                    "failed": sum(r["within_limit"] is False for r in rows),
                    "unresolved": sum(r["within_limit"] is None for r in rows), "limit": maximum,
                    "axis": trace.units}
    stat.rows = rows
    center = np.asarray([.5 * (f.last + f.first) for f in output])
    layers = []
    for key, title, color in (("mean", "Mean", CYAN), ("minimum", "Minimum", BLUE), ("maximum", "Maximum", AMBER)):
        values = np.asarray([getattr(item, key) if getattr(item, key) is not None else np.nan for item in output])
        layers.append(Series(title, center, values, color))
    stat.chart("continuum-flatness", "Bandwise observed continuum values",
                 plot("Continuum uniformity", trace.units, "Linear light", layers,
                      x_bounds=(capture.lower, capture.upper)))
    stat.csv("continuum-flatness", "Native support and continuum summary", rows)
    return stat
