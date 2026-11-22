"""Line, pass band and relative visible response measurements for one trace."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .contracts import NamedWindow, Problem, allowed, choice, mapping, real, rounded, seq, whole
from .result import Result
from .strip import Trace, contiguous
from .view import AMBER, BLUE, CYAN, GREEN, PINK, Region, Series, plot, strip_overlay


MIN_KNOWN_SPAN = 1e-8


@dataclass(frozen=True)
class Line:
    position: float
    crest: float
    floor: float
    low_shoulder: float | None
    high_shoulder: float | None
    local_index: int

    @property
    def contrast(self):
        return self.crest - self.floor

    def row(self, number, trace: Trace):
        point = trace.row(self.local_index)
        origin = ([{"source": c["source"], "source_index": c["source_index"],
                    "native_column": c["native_column"], "source_uv": c["source_uv"],
                    "share": c["share"], "exposure_ms": c["exposure_ms"]}
                   for c in point["contributors"]]
                  if point.get("contributors") else None)
        return {"line": number, "position": rounded(self.position), "axis_units": trace.units,
                "crest": rounded(self.crest), "local_contrast": rounded(self.contrast),
                "half_height_low": rounded(self.low_shoulder), "half_height_high": rounded(self.high_shoulder),
                "fwhm": rounded(self.high_shoulder - self.low_shoulder)
                if self.low_shoulder is not None and self.high_shoulder is not None else None,
                "source_uv": point["source_uv"], "native_column": int(point["column_px"]),
                "source_index": point.get("source_index"), "origin": origin}


def _cross_x(a, b, p, q, threshold):
    if p == q:
        return float((a + b) / 2)
    value = (threshold - p) / (q - p)
    return float(a + np.clip(value, 0, 1) * (b - a))


def _crest(x, y, i):
    a, b, c = x[i - 1:i + 2]
    fa, fb, fc = y[i - 1:i + 2]
    slope = (fc - fb) / (c - b)
    other = (fb - fa) / (b - a)
    quadratic = (slope - other) / (c - a)
    if quadratic >= -1e-12:
        return float(b), float(fb)
    linear_term = other + quadratic * (b - a)
    offset = -linear_term / (2 * quadratic)
    if not a < b + offset < c:
        return float(b), float(fb)
    return float(b + offset), float(fb + linear_term * offset + quadratic * offset ** 2)


def _crossing(x, y, k, count, half, distance):
    end = int(np.clip(k + count * distance, 0, len(y) - 1))
    for i in range(k + count, end + count, count):
        if y[i] <= half:
            return _cross_x(x[i], x[i - count], y[i], y[i - count], half)
    return None


def local_lines(t: Trace, window: int, contrast: float, spacing: float, maximum: int):
    axis_ok = np.isfinite(t.axis)
    selected = np.flatnonzero(axis_ok)
    selected = selected[np.argsort(t.axis[selected])]
    x, y = t.axis[selected], t.signal[selected]
    known = t.known[selected]
    candidates = []
    for section in contiguous(known & np.isfinite(y)):
        u, v = x[section], y[section]
        start = section.start
        for i in range(1, len(v) - 1):
            if not (v[i - 1] < v[i] >= v[i + 1]):
                continue
            left = v[max(0, i - window):i]
            right = v[i + 1:i + window + 1]
            floor = float(max(left.min(), right.min()))
            peak, crest = _crest(u, v, i)
            if crest - floor < contrast:
                continue
            half = .5 * (crest + floor)
            candidates.append(Line(peak, crest, floor, _crossing(u, v, i, -1, half, window),
                                   _crossing(u, v, i, 1, half, window), int(selected[start + i])))
    ordered = sorted(candidates, key=lambda line: (-line.contrast, line.position))
    kept = []
    for c in ordered:
        if all(abs(c.position - peak.position) >= spacing for peak in kept):
            kept.append(c)
        if len(kept) == maximum:
            break
    return sorted(kept, key=lambda peak: peak.position)


def crest_report(bench):
    bench.require(1, 4)
    merged = len(bench.sources) > 1
    t = bench.trace()
    data = bench.order.settings
    allowed(data, {"shoulder", "contrast", "spacing", "maximum", "match_gain"}, "line survey")
    span = whole(data.get("shoulder", 16), "Shoulder window (columns)", 3, 180)
    threshold = real(data.get("contrast", .035), "Minimum line contrast", 0, 2)
    distance = real(data.get("spacing", 4), "Minimum line spacing", .02, 230)
    maximum = whole(data.get("maximum", 30), "Maximum recorded lines", 1, 70)
    lines = local_lines(t, span, threshold, distance, maximum)
    rows = [p.row(i + 1, t) for i, p in enumerate(lines)]
    result = Result("emission-lines", "Local emission line register",
                    "Line crests are associated with camera coordinates and reported with local half-height shoulders where available."
                    + (" Each line records the exposures that built the fused trace." if merged else ""))
    result.metrics = {"axis": t.units, "lines": len(lines), "merged": merged,
                      "with_two_shoulders": sum(r["fwhm"] is not None for r in rows),
                      "strongest_position": rounded(max(lines, key=lambda a: a.contrast).position) if lines else None,
                      "minimum_contrast": threshold, "spacing": distance}
    if merged:
        result.metrics["merged_exposures"] = [source.title for source in bench.sources]
    result.rows = rows
    value = np.where(t.known, t.signal, np.nan)
    chart = plot("Emission line register", t.units, "Linear light", [Series("Camera spectrum", t.axis, value)],
                 markers=[(p.position, p.crest, str(i + 1)) for i, p in enumerate(lines)])
    result.image("detected-lines", "Annotated light crests", chart.image,
                 {"kind": "merged-axis", **chart.public()} if merged
                 else {"kind": "axis", **chart.public()})
    if merged:
        for i, source in enumerate(bench.sources):
            result.image(f"line-strip-{i + 1}", f"{source.title}: native strip under the crest chart",
                         strip_overlay(source.camera, source.trace), {"kind": "source", "source_index": i})
    else:
        result.image("line-strip", "Original strip under the crest chart", strip_overlay(bench.sources[0].camera, t),
                     {"kind": "source", "source_index": 0})
    result.csv("emission-line-register", "Positions and observed widths", rows)
    return result


@dataclass(frozen=True)
class Area:
    signed: float
    positive: float
    negative: float
    covered: float
    center_weight: float
    segments: int

    def public(self, width):
        measured = self.covered > MIN_KNOWN_SPAN
        return {"signed_area": rounded(self.signed) if measured else None,
                "positive_area": rounded(self.positive) if measured else None,
                "negative_area": rounded(self.negative) if measured else None,
                "observed_width": rounded(self.covered), "unobserved_width": rounded(max(0, width - self.covered)),
                "supported_fraction": rounded(min(1, self.covered / width)),
                "signal_weighted_center": rounded(self.center_weight / self.positive)
                if self.positive > MIN_KNOWN_SPAN else None, "segments": self.segments}


def _area_segment(a, b, y, z):
    if y >= 0 and z >= 0:
        area = (y + z) / 2 * (b - a)
        x_moment = (b - a) / 6 * ((2 * a + b) * y + (a + 2 * b) * z)
        return area, 0., x_moment
    if y <= 0 and z <= 0:
        return 0., -(y + z) / 2 * (b - a), 0.
    mid = _cross_x(a, b, y, z, 0.)
    one, two, three = _area_segment(a, mid, y, 0.)
    p, n, m = _area_segment(mid, b, 0., z)
    return one + p, two + n, three + m


def area(x, values, valid, lo, hi, scale=1.) -> Area:
    plus = minus = covered = moment = 0.
    segments = 0
    usable = valid & np.isfinite(values)
    for run in contiguous(usable):
        if run.stop - run.start < 2 or x[run.stop - 1] <= lo or x[run.start] >= hi:
            continue
        segments += 1
        for i in range(run.start, run.stop - 1):
            x0, x1 = float(x[i]), float(x[i + 1])
            a, b = max(lo, x0), min(hi, x1)
            if b <= a:
                continue
            fa, fb = np.interp([a, b], [x0, x1], values[i:i + 2]) * scale
            positive, negative, weight = _area_segment(a, b, fa, fb)
            plus += positive
            minus += negative
            moment += weight
            covered += b - a
    return Area(plus - minus, plus, minus, covered, moment, segments)


def default_windows(axes, units):
    if units == "nm":
        return [{"name": "Blue-green", "interval": [450, 520]},
                {"name": "Red", "interval": [610, 665]}]
    lo, hi = float(axes[0]), float(axes[-1])
    step = (hi - lo) / 4
    return [{"name": "Left region", "interval": [lo, lo + 2 * step]},
            {"name": "Right region", "interval": [lo + 2 * step, hi]}]


def bands(bench):
    s = bench.one()
    t = s.trace
    data = bench.order.settings
    allowed(data, {"bands", "per_ms"}, "band reading")
    axes, values, valid = t.axis_array()
    if len(axes) < 2:
        raise Problem("At least two supported axis positions are needed to measure a band.")
    bands = seq(data.get("bands") or default_windows(axes, t.units), "Band intervals", 1, 12)
    refs = [NamedWindow.read(v, i) for i, v in enumerate(bands)]
    per_ms = data.get("per_ms", False)
    multiplier = 1. / s.exposure if per_ms is True else 1.
    rows = []
    for zone in refs:
        lo, hi = zone.interval.lower, zone.interval.upper
        measure = area(axes, values, valid, lo, hi, multiplier)
        rows.append({"band": zone.name, "lower": lo, "upper": hi, "axis_units": t.units,
                     **measure.public(zone.interval.width)})
    stat = Result("band-area", "Integrated light in defined pass bands",
                  "The integration uses only contiguous measured spans. It reports their width and does not bridge gaps.")
    stat.metrics = {"windows": len(rows), "units": t.units,
                    "rate_normalized": per_ms is True, "exposure_ms": s.exposure,
                    "fully_read_windows": sum(row["supported_fraction"] >= .999999 for row in rows)}
    stat.rows = rows
    layers = [Region(item.interval.lower, item.interval.upper, item.name, CYAN if i % 2 else AMBER)
              for i, item in enumerate(refs)]
    stat.chart("band-integration", "Measured signal and integration bounds",
               plot("Pass band areas", t.units, "Signal / ms" if per_ms is True else "Linear signal",
                    [Series("Camera spectrum", axes, np.where(valid, values * multiplier, np.nan))],
                    regions=layers))
    stat.csv("band-area-table", "Numerical window integration and evidence support", rows)
    return stat


def _cue_curve(x, group):
    definitions = {"violet": (420., 26.), "green": (535., 44.), "red": (635., 49.)}
    center, width = definitions[group]
    return np.exp(-.5 * ((x - center) / width) ** 2)


def visible_balance(bench):
    s = bench.one()
    t = s.trace
    if t.units != "nm":
        raise Problem("Visible response requires a wavelength calibration.")
    options = bench.order.settings
    allowed(options, {"target"}, "visible response")
    indicator = choice(options.get("target", "all"), ("all", "violet", "green", "red"), "Chart emphasis")
    axis, values, good = t.axis_array()
    if len(axis) < 2:
        raise Problem("Calibrate more than one sample in the visible range.")
    rows = []
    shapes = []
    summed = 0.
    for name, colour in zip(("violet", "green", "red"), (BLUE, GREEN, PINK), strict=True):
        weights = _cue_curve(axis, name)
        actual = area(axis, values * weights, good, 390, 700)
        reference = area(axis, weights, good, 390, 700)
        expected_axis = np.linspace(390, 700, 311)
        expected_weight = float(np.sum((_cue_curve(expected_axis, name)[1:] +
                                       _cue_curve(expected_axis, name)[:-1]) * .5))
        weight = actual.positive
        rows.append({"cue": name, "positive_weighted_light": rounded(weight) if actual.covered else None,
                     "support_of_visible_weight": rounded(reference.positive / expected_weight),
                     "measured_nm": rounded(actual.covered)})
        if indicator in {"all", name}:
            shapes.append(Series(name, axis, np.where(good, values * weights, np.nan), colour))
        summed += weight
    for line in rows:
        raw = line["positive_weighted_light"]
        line["share_of_weighted_light"] = rounded(raw / summed) if summed > MIN_KNOWN_SPAN and raw is not None else None
    report = Result("visible-balance", "Relative camera response in the visible region",
                    "The three broad weights are consistent comparison cues for this studio, with no claim of human color matching.")
    report.rows = rows
    report.metrics = {"cue_types": len(rows), "range_nm": [390, 700], "relative_total": rounded(summed)}
    report.chart("visible-response", "Camera signal under broad comparison weights",
                 plot("Relative visible response", "Wavelength (nm)", "Weighted linear light", shapes,
                      x_bounds=(390., 700.)))
    report.csv("visible-response", "Response and evidence coverage", rows)
    return report
