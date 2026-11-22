"""Quantify wavelength overlap and gaps on several real camera observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .comparison import _input, _source_views
from .contracts import NamedWindow, Problem, allowed, flag, rounded, seq, whole
from .result import Result
from .strip import contiguous, regrid
from .view import AMBER, BLUE, CYAN, GREEN, PINK, Region, Series, palette, plot


@dataclass(frozen=True)
class Extent:
    low: float
    high: float
    samples: int
    covered_width: float
    name: str

    def row(self):
        return {"name": self.name, "first_axis": rounded(self.low), "last_axis": rounded(self.high),
                "coverage_width": rounded(self.covered_width), "read_samples": self.samples}


def _width(axis, support, low, high):
    pairs = support[:-1] & support[1:]
    widths = np.maximum(0., np.minimum(high, axis[1:]) - np.maximum(low, axis[:-1]))
    return float(np.dot(pairs.astype(float), widths))


def _edges(axis, support):
    p = np.flatnonzero(support)
    return (float(axis[p[0]]), float(axis[p[-1]])) if len(p) else (np.nan, np.nan)


def _availability_rows(axis, flags, sources):
    low, high = axis[0], axis[-1]
    response = []
    for i, s in enumerate(sources):
        support = flags[i]
        start, end = _edges(axis, support)
        response.append(Extent(start, end, int(np.sum(support)), _width(axis, support, low, high), s.title).row())
    return response


def _runs(axis, observed):
    changes = np.flatnonzero(np.diff(observed, prepend=-1, append=-2))
    outputs = []
    for a, b in zip(changes[:-1], changes[1:], strict=True):
        if b <= a:
            continue
        codes = observed[a:b]
        states = int(codes[0])
        outputs.append({"source_count": states, "from_axis": rounded(axis[a]),
                        "to_axis": rounded(axis[b - 1]), "samples": int(b - a)})
    return outputs


def _comparison(bench, count):
    sources = _input(bench, 2, 4)
    t = [s.trace for s in sources]
    axis, values = regrid(tuple(t), count)
    measured = np.isfinite(np.asarray(values))
    return sources, axis, measured


def common_coverage(bench):
    cfg = bench.order.settings
    allowed(cfg, {"samples", "bands", "show_individual"}, "wavelength coverage")
    size = whole(cfg.get("samples", 380), "Common axis sample count", 40, 1200)
    per_capture = flag(cfg.get("show_individual", True), "Show source-specific support")
    sources, axis, good = _comparison(bench, size)
    all_types = np.sum(good, axis=0, dtype=int)
    collective = np.any(good, axis=0)
    agreement = np.all(good, axis=0)
    plain = [{"name": "Full axis", "interval": [float(axis[0]), float(axis[-1])]}]
    zones = [NamedWindow.read(v, i) for i, v in enumerate(seq(cfg.get("bands") or plain, "Coverage windows", 1, 18))]
    readings = []
    for choice in zones:
        span = choice.interval
        observed = _width(axis, collective, span.lower, span.upper)
        shared = _width(axis, agreement, span.lower, span.upper)
        rows = {"band": choice.name, "bounds": span.public(), "union_width": rounded(observed),
                "shared_width": rounded(shared), "no_observation_width": rounded(max(0, span.width - observed))}
        for i, item in enumerate(good, 1):
            rows[f"source_{i}_covered"] = rounded(_width(axis, item, span.lower, span.upper))
        readings.append(rows)
    unit = sources[0].trace.units
    draw = [Series("Support count", axis, all_types.astype(float), CYAN)]
    if per_capture:
        for i, line in enumerate(good):
            displayed = np.where(line, i + .3, np.nan)
            draw.append(Series(sources[i].title, axis, displayed, palette(i + 1)))
    report = Result("source-coverage", "Wavelength support shared by observations",
                    "Show where several detector observations have actual calibrated data before comparing measured signal.")
    report.rows = readings
    report.metrics = {"selected_observations": len(sources), "axis_units": unit,
                      "sampled_grid": size, "all_sources": int(np.sum(agreement)),
                      "any_source": int(np.sum(collective)), "measured_intervals": len(_runs(axis, all_types))}
    report.chart("wavelength-coverage", "Measured support from selected sources", plot(
        "Wavelength availability", unit, "Number of usable observations", draw,
        regions=[Region(z.interval.lower, z.interval.upper, z.name) for z in zones],
        y_bounds=(-.1, len(sources) + .15)))
    report.csv("available-band-widths", "Coverage counts and band widths", readings)
    report.csv("source-available-extents", "Extent and width of available wavelengths", _availability_rows(axis, good, sources))
    report.csv("coverage-runs", "Runs with the same number of supporting observations", _runs(axis, all_types))
    _source_views(report, sources)
    return report


def _grid_ranges(axis, selected, minimum, states):
    for span in contiguous(selected):
        n = span.stop - span.start
        if n >= minimum:
            i, j = span.start, span.stop - 1
            yield {"first_axis": rounded(axis[i]), "last_axis": rounded(axis[j]),
                   "samples": n, "matching_state": states,
                   "approximate_width": rounded(axis[j] - axis[i])}


def discontinuities(bench):
    cfg = bench.order.settings
    allowed(cfg, {"samples", "minimum", "consider"}, "support gap")
    n = whole(cfg.get("samples", 420), "Common axis samples", 50, 1200)
    minimum = whole(cfg.get("minimum", 5), "Minimum run length", 1, n)
    category = cfg.get("consider", "any")
    if category not in {"any", "shared"}:
        raise Problem("Select shared or any coverage.")
    sources, axis, observed = _comparison(bench, n)
    full_grid = np.any(observed, axis=0) if category == "any" else np.all(observed, axis=0)
    rows = list(_grid_ranges(axis, ~full_grid, minimum, "missing support"))
    output = []
    for i, mask in enumerate(observed):
        details = list(_grid_ranges(axis, ~mask, minimum, f"{sources[i].title}: unmeasured"))
        output.extend(details)
    doc = Result("coverage-gaps", "Gaps between recorded spectral intervals",
                 "Distinct gap ranges are retained rather than joined through an empty interval.")
    doc.rows = rows
    doc.metrics = {"support_rule": category, "reported_ranges": len(rows), "missing_samples": int(np.sum(~full_grid)),
                   "sources": len(observed), "grid_samples": n, "source_gaps": len(output)}
    doc.chart("coverage-gaps", "Unmeasured spectral positions", plot(
        "Coverage gaps", sources[0].trace.units, "Measurable at position",
        [Series("Available", axis, full_grid.astype(float))], y_bounds=(-.04, 1.05),
        regions=[Region(r["first_axis"], r["last_axis"], "gap", PINK) for r in rows]))
    doc.csv("common-coverage-gaps", "Gaps for the chosen support rule", rows)
    doc.csv("individual-coverage-gaps", "Distinct gaps on each source", output)
    return doc
