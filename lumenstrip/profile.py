"""Report the value distribution and axial energy of single recorded lines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .continuum import _fallback_range, _sample_intervals, _integral_crossings
from .contracts import NamedWindow, allowed, flag, real, rounded, seq, whole
from .result import Result
from .spectral import area
from .view import AMBER, BLUE, CYAN, GREEN, PINK, Region, Series, plot


@dataclass(frozen=True)
class Distribution:
    qlo: float | None
    center: float | None
    qhi: float | None
    minimum: float | None
    maximum: float | None
    accepted: int

    @classmethod
    def read(cls, signal, fraction):
        if not len(signal):
            return cls(None, None, None, None, None, 0)
        qlo, center, qhi = np.quantile(signal, [fraction, .5, 1 - fraction])
        return cls(float(qlo), float(center), float(qhi), float(np.min(signal)), float(np.max(signal)),
                   len(signal))

    def public(self):
        return {"lower_quantile": rounded(self.qlo), "median": rounded(self.center),
                "upper_quantile": rounded(self.qhi), "minimum": rounded(self.minimum),
                "maximum": rounded(self.maximum), "measured": self.accepted}


def _selections(t, raw):
    items = raw or [{"name": "Complete axis", "interval": _fallback_range(t)}]
    return [NamedWindow.read(item, i) for i, item in enumerate(seq(items, "Profile bands", 1, 12))]


def _distribution_rows(t, bands, tail):
    for b in bands:
        permitted = (t.axis >= b.interval.lower) & (t.axis <= b.interval.upper) & t.known
        measurements = t.signal[permitted]
        compare = Distribution.read(measurements, tail)
        yield {"band": b.name, "range": b.interval.public(), **compare.public()}, measurements


def distributions(bench):
    source = bench.one()
    t = source.trace
    cfg = bench.order.settings
    allowed(cfg, {"bands", "bins", "tail"}, "spectral distribution")
    n = whole(cfg.get("bins", 35), "Signal histogram bins", 8, 100)
    tail = real(cfg.get("tail", .1), "Quantile tail", .001, .49)
    bands = _selections(t, cfg.get("bands"))
    evidence = list(_distribution_rows(t, bands, tail))
    rows = [summary for summary, _ in evidence]
    kept = [sub for _, sub in evidence if len(sub)]
    bounds = (float(min(v.min() for v in kept)), float(max(v.max() for v in kept))) if kept else (0., 1.)
    lower, upper = bounds
    if upper - lower <= 1e-10:
        lower -= 1e-4
        upper += 1e-4
    bucket_edges = np.linspace(lower, upper, n + 1)
    centers = .5 * (bucket_edges[:-1] + bucket_edges[1:])
    series = []
    bins = []
    for i, (summary, values) in enumerate(evidence):
        sample_ct = len(values)
        count, _ = np.histogram(values, bins=bucket_edges)
        density = count / sample_ct if sample_ct else np.zeros_like(count)
        colour = (CYAN, BLUE, PINK, GREEN, AMBER)[i % 5]
        series.append(Series(summary["band"], centers, density, colour))
        for j in range(n):
            bins.append({"band": summary["band"], "minimum": rounded(bucket_edges[j]),
                         "maximum": rounded(bucket_edges[j + 1]), "columns": int(count[j]),
                         "fraction": rounded(density[j]) if sample_ct else None})
    output = Result("signal-distribution", "Measured spectrum value distribution",
                    "Value histograms are derived only from native readings in the selected wavelength bands.")
    output.rows = rows
    output.metrics = {"bands": len(rows), "bins_per_band": n, "tail_fraction": tail,
                      "scope_units": t.units, "profile_setting": t.config.channel}
    output.image("spectral-distributions", "Signal histogram of selected wavelengths",
                 plot("Spectral value distribution", "Linear signal", "Fraction of measured columns", series).image)
    output.csv("spectral-distributions", "Value range and observed signal quantiles", rows)
    output.csv("spectral-value-bins", "Per-band signal histogram", bins)
    return output


def _energy(t, band, tail, exposure):
    x, raw, valid = t.axis_array()
    y = raw / exposure
    b = band.interval
    segments = _sample_intervals(x, y, valid, b, positive=True)
    limits = _integral_crossings(segments, (tail, .5, 1 - tail))
    stats = area(x, y, valid, b.lower, b.upper)
    widths = (limits[2] - limits[0]) if all(k is not None for k in limits) else None
    return {"band": band.name, "requested_range": b.public(),
            "shape_center": rounded(limits[1]), "lower_energy_crossing": rounded(limits[0]),
            "upper_energy_crossing": rounded(limits[2]), "included_width": rounded(widths),
            **stats.public(b.width)}, segments


def energy_profile(bench):
    s = bench.one()
    t = s.trace
    req = bench.order.settings
    allowed(req, {"bands", "tail", "per_ms"}, "band energy profile")
    tail = real(req.get("tail", .1), "Tail fraction", .01, .4)
    normalized = flag(req.get("per_ms", False), "Divide by exposure time")
    scale = s.exposure if normalized else 1.
    bands = _selections(t, req.get("bands"))
    candidates = [_energy(t, band, tail, scale) for band in bands]
    results = [r for r, _ in candidates]
    observations = []
    curves = []
    for i, (band, pieces) in enumerate(candidates):
        if not pieces:
            continue
        locus = np.asarray([p[1] for p in pieces])
        weights = np.asarray([p[-1] for p in pieces])
        total = np.sum(weights)
        energy = np.cumsum(weights) / total if total > .0000001 else np.full(len(weights), np.nan)
        curves.append(Series(band["band"], locus, energy, (CYAN, BLUE, PINK, AMBER, GREEN)[i % 5]))
        observations.extend({"band": band["band"], "axis": rounded(x), "fraction": rounded(energy[j]),
                             "segment_energy": rounded(weights[j])} for j, x in enumerate(locus))
    result = Result("band-energy", "Measured energy widths",
                    "Positive light inside a marked band estimates where its available energy lies. Missing support stays explicit.")
    result.rows = results
    result.metrics = {"band_count": len(results), "calibrated_axis": t.units, "tail_fraction": tail,
                      "normalized_by_exposure": normalized}
    result.image("cumulative-band-energy", "Cumulative energy by wavelength",
                 plot("Measured band energy", t.units, "Observed energy fraction", curves, y_bounds=(0, 1.)).image)
    result.csv("band-energy-widths", "Observed energy limits and coverage", results)
    result.csv("band-energy-evidence", "Positive light accumulation per measured interval", observations)
    return result
