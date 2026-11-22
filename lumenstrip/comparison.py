"""Compare recorded light, blank transmission and relative repeat agreement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import NamedWindow, Problem, allowed, choice, flag, real, rounded, seq, whole
from .result import Result
from .spectral import area
from .strip import contiguous, regrid
from .view import AMBER, BLUE, CYAN, GREEN, PINK, Region, Series, palette, plot, strip_overlay


def _input(bench, low=2, high=2):
    bench.require(low, high)
    return bench.same_basis()


def _resampled(sources, count, per_ms=False):
    x, curves = regrid(tuple(source.trace for source in sources), count)
    if per_ms:
        curves = [curve / s.exposure for s, curve in zip(sources, curves, strict=True)]
    return x, curves


def _row_values(x, curves, sources, units):
    return [{"position": rounded(v), "units": units,
             **{f"signal_{i + 1}": rounded(curve[j]) for i, curve in enumerate(curves)},
             **{f"source_{i + 1}_uv": (s.trace.row(k)["source_uv"] if np.isfinite(curves[i][j])
                                      and (k := s.trace.interval(v)) is not None else None)
                for i, s in enumerate(sources)}} for j, v in enumerate(x)]


def _state_counts(support):
    all_count = np.sum(support, axis=0)
    return {"common_positions": int(np.sum(all_count == len(support))),
            "partly_read_positions": int(np.sum((all_count > 0) & (all_count < len(support)))),
            "no_source_positions": int(np.sum(all_count == 0))}


def _source_views(result, sources):
    for index, s in enumerate(sources):
        result.image(f"source-{index + 1}-strip", f"{s.title}: native camera strip",
                     strip_overlay(s.camera, s.trace), {"kind": "source", "source_index": index})


def overlay(bench):
    sources = _input(bench, 2, 4)
    data = bench.order.settings
    allowed(data, {"count", "per_ms", "normalize"}, "comparison")
    count = whole(data.get("count", 360), "Common grid size", 40, 1400)
    normalize = choice(data.get("normalize", "none"), ("none", "max"), "Curve normalization")
    per_ms = flag(data.get("per_ms", False), "Normalize by exposure")
    axes, raw = _resampled(sources, count, per_ms)
    scales = []
    curves = []
    for profile in raw:
        valid = profile[np.isfinite(profile)]
        maximum = float(valid.max()) if len(valid) else 0.
        scale = maximum if normalize == "max" and maximum > 0 else 1.
        scales.append(scale)
        curves.append(profile / scale)
    support = np.asarray([np.isfinite(line) for line in curves])
    rows = _row_values(axes, curves, sources, sources[0].trace.units)
    stats = Result("repeat-overlay", "Aligned camera traces",
                   "Every selected observation remains separate. The common grid does not interpolate across unsupported gaps.")
    stats.metrics = {"units": sources[0].trace.units, "profile_count": len(sources),
                     "samples_on_grid": count, "applied_divisors": [rounded(s) for s in scales],
                     "exposure_normalized": per_ms, **_state_counts(support)}
    stats.rows = rows
    stats.chart("aligned-spectra", "Independent curves on one axis", plot(
        "Aligned camera traces", sources[0].trace.units,
        "Peak-relative signal" if normalize == "max" else "Signal / ms" if per_ms else "Linear signal",
        [Series(s.title, axes, curve, palette(i)) for i, (s, curve) in enumerate(zip(sources, curves, strict=True))]))
    _source_views(stats, sources)
    stats.csv("aligned-spectra", "Separate source curves and native lookup points", rows)
    return stats


@dataclass(frozen=True)
class Transmission:
    axis: np.ndarray
    sample: np.ndarray
    reference: np.ndarray
    dark: np.ndarray
    sample_level: np.ndarray
    open_level: np.ndarray
    ratio: np.ndarray
    density: np.ndarray
    states: np.ndarray

    def table(self, sources, units):
        return [{**row, "transmitted": rounded(self.ratio[i]),
                 "density": rounded(self.density[i]), "dark_signal": rounded(self.dark[i]),
                 "corrected_sample_signal": rounded(self.sample_level[i]),
                 "corrected_open_signal": rounded(self.open_level[i]), "state": self.states[i]}
                for i, row in enumerate(_row_values(self.axis, [self.sample, self.reference, self.dark][:len(sources)], sources, units))]

    @property
    def valid(self):
        return self.states == "measured"


def _transmitted(axes, channels, floor):
    camera, reference = channels[:2]
    dark = channels[2] if len(channels) == 3 else np.zeros(len(axes))
    a, b = camera - dark, reference - dark
    present = np.isfinite(camera) & np.isfinite(reference) & np.isfinite(dark)
    supported = present & (b > floor) & (a >= 0)
    ratio = np.divide(a, b, where=supported, out=np.full(len(a), np.nan))
    density = np.full(len(ratio), np.nan)
    density[ratio > 0] = -np.log10(ratio[ratio > 0])
    states = np.full(len(axes), "measured", dtype="<U32")
    states[present & (a < 0)] = "signal below dark"
    states[present & (b <= floor)] = "reference below floor"
    states[~present] = "unsupported source"
    return Transmission(axes, camera, reference, dark, a, b, ratio, density, states)


def _transmission_stats(calc, unit):
    flags, counts = np.unique(calc.states, return_counts=True)
    measurements = calc.ratio[calc.valid]
    return {"axis": unit, "states": {str(k): int(v) for k, v in zip(flags, counts, strict=True)},
            "known_share": rounded(np.mean(calc.valid)),
            "minimum_transmission": rounded(measurements.min()) if len(measurements) else None,
            "median_transmission": rounded(np.median(measurements)) if len(measurements) else None,
            "maximum_transmission": rounded(measurements.max()) if len(measurements) else None,
            "above_nominal_open": int(np.sum(measurements > 1.))}


def transmission(bench):
    views = _input(bench, 2, 3)
    request = bench.order.settings
    allowed(request, {"count", "floor", "mode"}, "transmission")
    count = whole(request.get("count", 420), "Wavelength samples", 48, 1400)
    floor = real(request.get("floor", .00008), "Open signal floor / ms", 0., 2.)
    mode = choice(request.get("mode", "transmission"), ("transmission", "density"), "Reading mode")
    x, curves = _resampled(views, count, per_ms=True)
    unit = views[0].trace.units
    result_data = _transmitted(x, curves, floor)
    result = Result("transmission", "Filter transmission and optical density",
                    "The first image is the material, the second the open reference and the optional third is dark.")
    measures = _transmission_stats(result_data, unit)
    result.metrics = {**measures, "display_mode": mode,
                      "source_roles": [s.role for s in views],
                      "optional_dark_reference_used": len(views) == 3, "open_floor_per_ms": floor}
    result.rows = result_data.table(views, unit)
    values = result_data.density if mode == "density" else result_data.ratio
    result.chart("filter-reading", "Corrected filter measurement",
                 plot("Filter measurement", unit, mode.title(), [Series(mode.title(), x, values)],
                      horizontal=0. if mode == "density" else 1.))
    result.chart("filter-ambient", "Signal before dividing on the common axis",
                 plot("Exposure-normalized source evidence", unit, "Signal / ms",
                      [Series(s.title, x, curve, palette(i)) for i, (s, curve) in
                       enumerate(zip(views, curves, strict=True))]))
    _source_views(result, views)
    result.csv("filter-reading", "Transmission and density by wavelength", result.rows)
    if measures["above_nominal_open"]:
        result.warnings.append("Some readings exceed the open reference. Check capture and exposure comparability.")
    if views[1].role != "open-reference":
        result.warnings.append("The second item is not labelled an open reference.")
    if len(views) == 3 and views[2].role != "dark-reference":
        result.warnings.append("The third item is not labelled a dark reference.")
    return result


def _runs_table(index, flags, x, values, cutoff):
    rows = []
    for run in contiguous(flags):
        a = run.start
        b = run.stop - 1
        section = values[a:b + 1]
        rows.append({"window": index, "start": rounded(x[a]), "end": rounded(x[b]),
                     "sample_count": len(section), "state": "above limit",
                     "maximum_error": rounded(max(section)), "exceedance": rounded(max(section) - cutoff)})
    return rows


def repeatability(bench):
    sources = _input(bench, 2, 4)
    data = bench.order.settings
    allowed(data, {"count", "limit", "per_ms", "bands"}, "repeat spread")
    count = whole(data.get("count", 360), "Common grid samples", 40, 1100)
    maximum = real(data.get("limit", .04), "Absolute repeat difference", .000002, 2.)
    per_ms = flag(data.get("per_ms", False), "Normalize by exposure")
    unit = sources[0].trace.units
    x, curves = _resampled(sources, count, per_ms)
    signals = np.asarray(curves)
    support = np.sum(np.isfinite(signals), axis=0)
    extrema = np.max(np.where(np.isfinite(signals), signals, -np.inf), axis=0) - (
        np.min(np.where(np.isfinite(signals), signals, np.inf), axis=0)
    )
    observed = support >= 2
    extrema = np.where(observed, extrema, np.nan)
    exceeded = observed & (extrema > maximum)
    intervals = data.get("bands") or [{"name": "Entire available axis",
                                       "interval": [float(max(0, x[0])), float(x[-1])]}]
    windows = [NamedWindow.read(v, i) for i, v in enumerate(seq(intervals, "Repeatability windows", 1, 12))]
    rows = []
    details = []
    for i, w in enumerate(windows):
        inside = (x >= w.interval.lower) & (x <= w.interval.upper)
        confirmed = inside & observed
        n = int(confirmed.sum())
        present = extrema[confirmed]
        rows.append({"band": w.name, "window": w.interval.public(), "sampled": int(inside.sum()),
                     "compared": n, "above_limit": int((inside & exceeded).sum()),
                     "median_spread": rounded(np.median(present)) if n else None,
                     "max_spread": rounded(np.max(present)) if n else None})
        details.extend(_runs_table(i + 1, exceeded & inside, x, extrema, maximum))
    report = Result("repeat-spread", "Repeat observation spread",
                    "Disagreement is measured only where at least two traces can be read. Unsupported wavelengths remain blank.")
    report.metrics = {"exposure_normalized": per_ms, "threshold": maximum, "units": unit,
                      "compared_samples": int(observed.sum()), "above_limit": int(exceeded.sum())}
    report.rows = rows
    report.chart("repeat-spread", "Observed difference relative to limit", plot(
        "Repeat measurement range", unit, "Light spread / ms" if per_ms else "Light spread",
        [Series("Highest minus lowest", x, extrema)], horizontal=maximum,
        regions=[Region(w.interval.lower, w.interval.upper, w.name, BLUE) for w in windows]))
    rows_raw = _row_values(x, curves, sources, unit)
    report.csv("repeat-spread", "Spread within selected ranges", rows)
    report.csv("repeat-exceedances", "Contiguous above-limit ranges", details)
    report.csv("repeat-input-points", "Independent signal and evidence coordinates", rows_raw)
    _source_views(report, sources)
    return report


def _delay(x, a, b, shifts, minimum):
    results = []
    step = x[1] - x[0]
    for shift in range(-shifts, shifts + 1):
        offset = np.full(len(b), np.nan)
        if shift == 0:
            offset = b.copy()
        elif shift > 0:
            offset[shift:] = b[:-shift]
        else:
            offset[:shift] = b[-shift:]
        keep = np.isfinite(a) & np.isfinite(offset)
        count = int(keep.sum())
        if count >= minimum:
            u, v = a[keep], offset[keep]
            centered, moving = u - np.mean(u), v - np.mean(v)
            scale = np.linalg.norm(centered) * np.linalg.norm(moving)
            correlation = float(np.dot(centered, moving) / scale) if scale > 1e-12 else np.nan
            residual = float(np.mean(np.abs(centered - moving)))
        else:
            correlation = residual = np.nan
        results.append({"x_shift": rounded(shift * step), "grid_shift": shift, "common_samples": count,
                        "r": rounded(correlation), "mean_centered_signal_difference": rounded(residual)})
    return results


def drift(bench):
    one, two = _input(bench)
    cfg = bench.order.settings
    allowed(cfg, {"range", "min_common", "count"}, "axis comparison")
    size = whole(cfg.get("count", 320), "Common grid size", 80, 900)
    search = whole(cfg.get("range", 12), "Offset range (common samples)", 1, min(80, size // 3))
    min_common = whole(cfg.get("min_common", 40), "Minimum shared support", 8, size)
    x, (a, b) = _resampled((one, two), size, True)
    shifts = _delay(x, a, b, search, min_common)
    supported = [h for h in shifts if h["r"] is not None]
    winner = max(supported, key=lambda s: (s["r"], -abs(s["grid_shift"]))) if supported else None
    result = Result("axis-drift", "Compare wavelength registration drift",
                    "Correlation is calculated for candidate offsets on their actual common support. Raw traces stay attached to their original axes.")
    result.rows = shifts
    result.metrics = {"axis": one.trace.units, "evaluated_offsets": len(shifts),
                      "usable_offsets": len(supported), "strongest_common_offset": winner,
                      "grid_spacing": rounded(x[1] - x[0])}
    result.image("axis-drift", "Correlation by candidate offset", plot(
        "Axis offset check", f"Offset ({one.trace.units})", "Centered correlation",
        [Series("Match", np.asarray([h["x_shift"] for h in shifts]),
                np.asarray([h["r"] if h["r"] is not None else np.nan for h in shifts]))],
        y_bounds=(-1., 1.)).image)
    result.chart("axis-drift-inputs", "Quick comparison of the source spectra",
                 plot("Unshifted source curves", one.trace.units, "Signal / ms",
                      [Series(one.title, x, a), Series(two.title, x, b, AMBER)]))
    result.csv("axis-drift", "Read evidence behind candidate offsets", shifts)
    _source_views(result, (one, two))
    return result


def band_comparison(bench):
    sources = _input(bench, 2, 4)
    data = bench.order.settings
    allowed(data, {"bands", "count", "per_ms"}, "shared band reading")
    target_count = whole(data.get("count", 440), "Comparison grid samples", 44, 1100)
    check_rate = flag(data.get("per_ms", False), "Normalize by exposure")
    x, curves = _resampled(sources, target_count, check_rate)
    unit = sources[0].trace.units
    intervals = data.get("bands") or [{"name": "Shared range",
                                      "interval": [float(max(0, x[0])), float(x[-1])]}]
    bands = [NamedWindow.read(v, i) for i, v in enumerate(seq(intervals, "Band windows", 1, 12))]
    common = np.all(np.isfinite(np.asarray(curves)), axis=0)
    rows = []
    for zone in bands:
        lo, hi = zone.interval.lower, zone.interval.upper
        for i, (s, curve) in enumerate(zip(sources, curves, strict=True)):
            own = area(x, curve, np.isfinite(curve), lo, hi)
            shared = area(x, curve, common, lo, hi)
            rows.append({"band": zone.name, "source": s.title, "range": zone.interval.public(),
                         "own_area": own.public(hi - lo), "shared_area": shared.public(hi - lo)})
    report = Result("shared-windows", "Light area on own or shared support",
                    "Separate and shared-support integrations help distinguish a light difference from a coverage difference.")
    report.rows = rows
    report.metrics = {"bands": len(bands), "sources": len(sources), "units": unit,
                      "exposure_normalized": check_rate, "common_positions": int(common.sum())}
    report.chart("shared-windows", "Band data limited to common support", plot(
        "Shared band windows", unit, "Signal / ms" if check_rate else "Linear signal",
        [Series(s.title, x, np.where(common, values, np.nan), palette(i))
         for i, (s, values) in enumerate(zip(sources, curves, strict=True))],
        regions=[Region(w.interval.lower, w.interval.upper, w.name) for w in bands]))
    report.csv("shared-window-area", "Native and common interval light areas", rows)
    report.csv("shared-window-evidence", "Source values with native positions", _row_values(x, curves, sources, unit))
    _source_views(report, sources)
    return report
