"""Measure supported pass/stop properties of a real camera filter comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .comparison import _input, _resampled, _source_views, _transmitted
from .contracts import NamedWindow, Window, allowed, choice, flag, mapping, real, rounded, seq, whole
from .result import Result
from .spectral import area
from .strip import contiguous
from .view import AMBER, CYAN, GREEN, PINK, Region, Series, plot


@dataclass(frozen=True)
class Requirement:
    label: str
    interval: Window
    relation: str
    limit: float

    @classmethod
    def parse(cls, value, i):
        data = mapping(value, "Filter band")
        allowed(data, {"name", "interval", "relation", "limit"}, "filter band")
        zone = NamedWindow.read({k: v for k, v in data.items() if k in {"name", "interval"}}, i)
        mode = choice(data.get("relation", "above"), ("above", "below"), "Pass relation")
        limit = real(data.get("limit", .45), "Transmission threshold", 0, 3.)
        return cls(zone.name, zone.interval, mode, limit)

    def clause(self, value):
        return value >= self.limit if self.relation == "above" else value <= self.limit

    def threshold_cross(self, a, b, p, q):
        if p == q or self.clause(q) == self.clause(p):
            return None
        t = (self.limit - p) / (q - p)
        return float(a + np.clip(t, 0, 1) * (b - a))


DEFAULT = [
    {"name": "Green pass band", "interval": [500, 559], "relation": "above", "limit": .44},
    {"name": "Short-wave stop", "interval": [410, 465], "relation": "below", "limit": .36},
]


@dataclass(frozen=True)
class Compliance:
    covered: float
    meets: float
    below: float
    segments: int
    observations: int
    crossings: tuple

    def detail(self, requirement: Requirement):
        width = requirement.interval.width
        remaining = max(0., width - self.covered)
        coverage = np.clip(self.covered / width, 0, 1)
        state = ("partly measured" if remaining > max(1e-6, .00001 * width)
                 else "measured")
        return {"name": requirement.label, "range": requirement.interval.public(),
                "required_comparison": requirement.relation, "required_level": requirement.limit,
                "observed_axis_width": rounded(self.covered), "certified_width": rounded(self.meets),
                "width_outside_limit": rounded(self.below), "unobserved_axis_width": rounded(remaining),
                "measured_fraction": rounded(coverage),
                "fraction_passing_of_observed": rounded(self.meets / self.covered) if self.covered > 1e-10 else None,
                "known_sections": self.segments, "native_samples": self.observations,
                "state": state if self.covered > 1e-10 else "unobserved"}


def _clip_axis(start, stop, x0, x1, y0, y1):
    left, right = max(start, x0), min(stop, x1)
    if left >= right:
        return None
    a, b = np.interp([left, right], [x0, x1], [y0, y1])
    return left, right, a, b


def _threshold_sections(x, y, valid, requirement):
    lo, hi = requirement.interval.lower, requirement.interval.upper
    covered = passing = failure = 0.
    branches = 0
    candidates = []
    enough = valid & np.isfinite(y)
    observations = int(np.sum(enough & (x >= lo) & (x <= hi)))
    for track in contiguous(enough):
        if track.stop - track.start < 2:
            continue
        if x[track.start] >= hi or x[track.stop - 1] <= lo:
            continue
        branches += 1
        for i in range(track.start, track.stop - 1):
            clipped = _clip_axis(lo, hi, x[i], x[i + 1], y[i], y[i + 1])
            if clipped is None:
                continue
            a, b, p, q = clipped
            length = b - a
            covered += length
            crossing = requirement.threshold_cross(a, b, p, q)
            if crossing is None:
                passed = length if requirement.clause((p + q) / 2) else 0.
            else:
                passed = crossing - a if requirement.clause(p) else b - crossing
                candidates.append({"window": requirement.label, "axis": rounded(crossing),
                                   "left_level": rounded(p), "right_level": rounded(q),
                                   "direction": "rising" if q > p else "falling"})
            passing += passed
            failure += length - passed
    return Compliance(covered, passing, failure, branches, observations, tuple(candidates))


def _default(axis, units):
    if units == "nm":
        return DEFAULT
    low, high = float(axis[0]), float(axis[-1])
    middle = (low + high) / 2
    return [{"name": "Leading interval", "interval": [low, middle], "relation": "above", "limit": .4},
            {"name": "Trailing interval", "interval": [middle, high], "relation": "below", "limit": .5}]


def filter_requirements(bench):
    sources = _input(bench, 2, 3)
    req = bench.order.settings
    allowed(req, {"bands", "floor", "count"}, "filter qualification")
    count = whole(req.get("count", 480), "Comparison grid samples", 60, 1350)
    floor = real(req.get("floor", .00008), "Minimum open signal per ms", 0., 2)
    x, raw = _resampled(sources, count, True)
    calc = _transmitted(x, raw, floor)
    units = sources[0].trace.units
    checks = [Requirement.parse(v, i) for i, v in enumerate(seq(req.get("bands") or _default(x, units),
                                                                 "Filter bands", 1, 14))]
    rows = []
    crossings = []
    for target in checks:
        measured = _threshold_sections(x, calc.ratio, calc.valid, target)
        rows.append(measured.detail(target))
        crossings.extend(measured.crossings)
    stat = Result("filter-specs", "Measured filter window qualification",
                  "A selected filter is measured against its supplied window thresholds, with unknown parts shown separately.")
    stat.metrics = {"requested_bands": len(rows), "measured_bands": sum(r["state"] == "measured" for r in rows),
                    "bands_with_outside_width": sum(r["width_outside_limit"] > .00001 for r in rows),
                    "interval_unit": units, "measured_samples": int(calc.valid.sum()),
                    "threshold_crossings": len(crossings)}
    stat.rows = rows
    markers = []
    levels = {test.label: test.limit for test in checks}
    for entry in crossings:
        markers.append((entry["axis"], levels[entry["window"]], "edge"))
    stat.chart("filter-thresholds", "Measured transmission and required windows",
               plot("Filter window qualification", units, "Transmission",
                    [Series("Measured filter", x, calc.ratio, CYAN)], markers=markers,
                    regions=[Region(item.interval.lower, item.interval.upper, item.label,
                                    GREEN if item.relation == "above" else AMBER) for item in checks]))
    stat.csv("filter-window-report", "Threshold compliance and observed width", rows)
    stat.csv("filter-window-crossings", "Measured threshold crossing details", crossings)
    _source_views(stat, sources)
    return stat


def _normalized_angle(y, major):
    peak = np.nanmax(np.abs(y)) if np.any(np.isfinite(y)) else 0.
    if major and peak > 1e-10:
        return y / peak
    return y


def filter_contrast(bench):
    sources = _input(bench, 2, 3)
    req = bench.order.settings
    allowed(req, {"bands", "floor", "count", "normalize"}, "filter attenuation")
    count = whole(req.get("count", 400), "Comparison grid samples", 60, 1350)
    floor = real(req.get("floor", .00008), "Open floor per ms", 0., 2)
    normalize = flag(req.get("normalize", False), "Normalize chart amplitudes")
    x, raw = _resampled(sources, count, True)
    calc = _transmitted(x, raw, floor)
    u = sources[0].trace.units
    if u == "nm":
        defaults = [{"name": "Reference short wave", "interval": [412., 478.]},
                    {"name": "Reference long wave", "interval": [566., 650.]}]
    else:
        defaults = [{"name": "Full camera interval", "interval": [float(x[0]), float(x[-1])]}]
    windows = [NamedWindow.read(v, i) for i, v in enumerate(seq(req.get("bands") or defaults, "Attenuation bands", 1, 12))]
    results = []
    for zone in windows:
        lo, hi = zone.interval.lower, zone.interval.upper
        after = area(x, calc.sample_level, calc.valid, lo, hi)
        before = area(x, calc.open_level, calc.valid, lo, hi)
        direct = area(x, calc.ratio, calc.valid, lo, hi)
        recovery = after.positive / before.positive if before.positive > 1e-10 else None
        arithmetic = direct.positive / direct.covered if direct.covered > 1e-10 else None
        results.append({"window": zone.name, "range": zone.interval.public(),
                        "background_subtracted_material_area": rounded(after.positive) if after.covered else None,
                        "background_subtracted_open_area": rounded(before.positive) if before.covered else None,
                        "light_ratio": rounded(recovery), "average_spectral_ratio": rounded(arithmetic),
                        "known_width": rounded(direct.covered),
                        "missing_width": rounded(max(0., zone.interval.width - direct.covered))})
    data = Result("filter-attenuation", "Total transmission vs spectral mean",
                  "Compare the fraction of measured light with average per-wavelength transmission in each band.")
    data.rows = results
    data.metrics = {"measured_support": rounded(np.mean(calc.valid)), "bands": len(windows),
                    "units": u, "chart_uses_normalized_amplitudes": normalize}
    first = np.where(calc.valid, calc.sample_level, np.nan)
    second = np.where(calc.valid, calc.open_level, np.nan)
    data.chart("filter-light-removal", "Open and filtered light on matching support", plot(
        "Measured filter contrast", u, "Normalized corrected light" if normalize else "Corrected light / ms",
        [Series("Filtered", x, _normalized_angle(first, normalize), CYAN),
         Series("Open", x, _normalized_angle(second, normalize), PINK)],
        regions=[Region(w.interval.lower, w.interval.upper, w.name) for w in windows]))
    data.csv("filter-attenuation", "Side-by-side estimated attenuation measures", results)
    return data
