"""Fuse several exposures of the same dispersion into one trace for line finding.

Each position on the fused axis keeps the contributing camera originals, their
native columns and the rows actually used.  Brightness disagreement in the
overlap is matched before blending, so a change of source is never mistaken for
a spectral line.  Positions without any trustworthy exposure stay blank.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .contracts import Problem, allowed, flag, rounded
from .result import Result
from .strip import Trace, contiguous, resample
from .view import CANVAS, CYAN, Series, annotate_loupe, font, palette, plot, strip_overlay


MAX_GRID = 5000
# A sample built from mostly clipped rows underestimates the real crest, so it is
# not allowed to erase a narrow line that a shorter exposure saw cleanly.
CLIP_LIMIT = .5
CLIP_PENALTY = 4.


def _clip_fraction(trace: Trace, index: int | None) -> float:
    if index is None:
        return 1.
    observed = int(trace.middle_count[index]) + int(trace.clipped_count[index]) + int(trace.dark_count[index])
    if observed <= 0:
        return 1.
    return float(np.clip(int(trace.clipped_count[index]) / observed, 0., 1.))


def _common_axis(traces: tuple[Trace, ...]) -> np.ndarray:
    units = {trace.units for trace in traces}
    if len(units) != 1:
        raise Problem("Merged exposures must share an axis. Calibrate all of them, or all in native columns.")
    if units == {"px"} and len({trace.plan.width for trace in traces}) > 1:
        raise Problem("Calibrate images of differing native widths before merging them.")
    values = [trace.axis[np.isfinite(trace.axis)] for trace in traces]
    if any(len(v) < 2 for v in values):
        raise Problem("Every merged exposure needs at least two measured axis positions.")
    joined = np.sort(np.concatenate(values))
    tolerance = max(1e-12, 1e-9 * max(1., abs(float(joined[-1] - joined[0]))))
    axis = joined[np.r_[True, np.diff(joined) > tolerance]]
    if len(axis) > MAX_GRID:
        axis = np.linspace(float(axis[0]), float(axis[-1]), MAX_GRID)
    return axis


@dataclass(frozen=True)
class Contribution:
    source_index: int
    trace_index: int
    title: str
    source_id: str
    exposure: float
    weight: float
    share: float
    value: float
    raw_value: float
    gain: float
    clip_fraction: float
    native_column: int
    source_uv: list[float]

    def public(self):
        return {"source_index": self.source_index, "source": self.title,
                "exposure_ms": rounded(self.exposure), "native_column": self.native_column,
                "source_uv": self.source_uv, "per_ms_light": rounded(self.value),
                "measured_light": rounded(self.raw_value), "weight": rounded(self.weight),
                "share": rounded(self.share), "applied_gain": rounded(self.gain),
                "clipped_row_fraction": rounded(self.clip_fraction)}


@dataclass(frozen=True)
class FusedTrace:
    units: str
    axis: np.ndarray
    signal: np.ndarray
    known: np.ndarray
    weight: np.ndarray
    contributions: tuple[tuple[Contribution, ...], ...]
    seams: tuple[int, ...]
    seam_steps: tuple[float, ...]
    sources: tuple
    traces: tuple
    match_gain: bool
    gains: tuple[float, ...]
    reference_exposure: float

    def axis_array(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.isfinite(self.axis)
        x = self.axis[mask]
        order = np.argsort(x)
        first = np.flatnonzero(mask)[order]
        return x[order], self.signal[first], self.known[first]

    def interval(self, target: float) -> int | None:
        points = np.flatnonzero(np.isfinite(self.axis))
        if not len(points):
            return None
        axis = self.axis[points]
        edge = abs(axis[-1] - axis[0]) / max(1, len(axis) - 1) / 2
        if target < np.min(axis) - edge or target > np.max(axis) + edge:
            return None
        return int(points[np.argmin(np.abs(axis - target))])

    @property
    def plan(self):
        return self.sources[0].trace.plan

    def _dominant(self, index: int) -> Contribution | None:
        group = self.contributions[index]
        return max(group, key=lambda item: item.weight) if group else None
    def row(self, index: int, exact: bool = False) -> dict:
        lead = self._dominant(index)
        found = self.contributions[index]
        result = {
            "x_axis": rounded(self.axis[index]), "axis_units": self.units,
            "linear_signal": rounded(self.signal[index]),
            "state": "merged" if self.known[index] else "no trustworthy exposure",
            "contributing_sources": len(found),
            "source_uv": lead.source_uv if lead else None,
            "column_px": lead.native_column if lead else None,
            "native_column": lead.native_column if lead else None,
            "source": lead.title if lead else None,
            "source_index": lead.source_index if lead else None,
            "exposures_ms": [rounded(item.exposure) for item in found],
            "contributors": [item.public() for item in found],
            "exposure_normalized_signal": rounded(self.signal[index] / self.reference_exposure)
            if self.known[index] and self.reference_exposure > 0 else None,
        }
        if exact:
            result["used_strip_rows"] = [self.traces[item.source_index].row(item.trace_index, exact=True)
                                         ["used_strip_rows"] for item in found]
            result["source_columns"] = [item.native_column for item in found]
        return result

    def summary(self) -> dict:
        finite = self.known & np.isfinite(self.signal)
        return {"axis": self.units, "columns": len(self.axis), "merged": True,
                "measured_fraction": rounded(self.known.sum() / len(self.axis)),
                "unmeasured_columns": int(np.sum(~self.known)),
                "minimum_signal": rounded(np.min(self.signal[finite])) if np.any(finite) else None,
                "maximum_signal": rounded(np.max(self.signal[finite])) if np.any(finite) else None,
                "median_signal": rounded(np.median(self.signal[finite])) if np.any(finite) else None,
                "reference_exposure_ms": rounded(self.reference_exposure),
                "brightness_matched": self.match_gain,
                "applied_gains": [rounded(g) for g in self.gains],
                "largest_seam_step": rounded(max((abs(s) for s in self.seam_steps), default=0.)),
                "seam_count": len(self.seams)}

    def contributing_titles(self, position: float) -> list[str]:
        index = self.interval(position)
        if index is None:
            return []
        return [item.title for item in self.contributions[index]]


def _match(curve, reference, good, reference_good, clipped, reference_clipped):
    """Match one exposure to the reference on their shared, unclipped overlap.

    A robust scale and additive term are chosen so that the two exposures agree
    where both can be read.  That keeps the hand-over from one exposure to the
    other level, rather than turning a brightness difference into a false crest.
    """
    overlap = good & reference_good & (clipped <= 0) & (reference_clipped <= 0)
    overlap &= np.isfinite(curve) & np.isfinite(reference) & (curve > 1e-9)
    if int(np.sum(overlap)) < max(5, int(.01 * len(curve))):
        return 1., 0., 0
    ratio = reference[overlap] / curve[overlap]
    gain = float(np.clip(np.median(ratio), .2, 5.))
    offset = float(np.median(reference[overlap] - gain * curve[overlap]))
    scale = float(np.median(np.abs(reference[overlap])))
    offset = float(np.clip(offset, -max(1e-6, .1 * scale), max(1e-6, .1 * scale)))
    return gain, offset, int(np.sum(overlap))


def _ordered(trace: Trace):
    mask = np.isfinite(trace.axis)
    absolute = np.flatnonzero(mask)
    absolute = absolute[np.argsort(trace.axis[absolute])]
    return trace.axis[absolute], np.where(trace.known, trace.signal, np.nan)[absolute], absolute


def fuse(sources, settings: dict | None = None) -> FusedTrace:
    data = settings or {}
    match_gain = flag(data["match_gain"], "Match overlap brightness") if "match_gain" in data else True
    traces = tuple(source.trace for source in sources)
    axis = _common_axis(traces)
    count = len(axis)
    exposures = np.asarray([source.exposure for source in sources], dtype=np.float64)
    reference = int(np.argmax(exposures))
    series = [_ordered(t) for t in traces]
    curves = np.asarray([resample(axis, xs, ys, trace.known[absolute]) / e
                         for (xs, ys, absolute), trace, e in zip(series, traces, exposures, strict=True)],
                        dtype=np.float64)
    indexes = [[t.interval(float(x)) for x in axis] for t in traces]
    clip = np.asarray([[_clip_fraction(t, j) for j in row] for t, row in zip(traces, indexes, strict=True)],
                      dtype=np.float64)
    good = np.asarray([np.isfinite(curve) for curve in curves])
    for i, trace in enumerate(traces):
        for j, index in enumerate(indexes[i]):
            good[i, j] = good[i, j] and index is not None and bool(trace.known[index])
    gains = np.ones(len(sources))
    offsets = np.zeros(len(sources))
    if match_gain:
        for i in range(len(sources)):
            gains[i], offsets[i], _ = _match(curves[i], curves[reference], good[i], good[reference],
                                             clip[i], clip[reference])
    gains[reference] = 1.
    offsets[reference] = 0.
    matched = curves * gains[:, None] + offsets[:, None]
    # Exposures are weighted by their own length, but clipped samples fade out
    # smoothly and vanish before half their rows are clipped.  A bright apex that
    # only the long exposure clips therefore hands over to the short exposure
    # instead of flattening a real crest; a position clipped everywhere stays blank.
    penalty = np.clip(1. - clip / CLIP_LIMIT, 0., 1.) ** CLIP_PENALTY
    weights = exposures[:, None] * penalty * good
    total = np.sum(weights, axis=0)
    weighted = np.where(weights > 0, weights * np.where(np.isfinite(matched), matched, 0.), 0.)
    fused = np.divide(np.sum(weighted, axis=0), total,
                      out=np.full(count, np.nan), where=total > 0)
    known = total > 0
    contributions = []
    dominant = np.full(count, -1, dtype=np.int32)
    for j in range(count):
        found = []
        for i in range(len(sources)):
            weight = float(weights[i, j])
            if weight <= 0 or not good[i, j]:
                continue
            index = int(indexes[i][j])
            point = traces[i].row(index)
            found.append(Contribution(
                i, index, sources[i].title, sources[i].id, float(exposures[i]), weight,
                weight / float(total[j]) if total[j] > 0 else 0., float(matched[i, j]),
                float(curves[i, j]), float(gains[i]), float(clip[i, j]),
                int(point["column_px"]), list(point["source_uv"])))
        if found:
            dominant[j] = max(found, key=lambda item: item.weight).source_index
        contributions.append(tuple(found))
    seams = tuple(int(j) for j in range(1, count)
                  if known[j] and known[j - 1] and dominant[j] >= 0 and dominant[j - 1] >= 0
                  and dominant[j] != dominant[j - 1])
    seam_steps = tuple(float(fused[j] - fused[j - 1]) for j in seams)
    return FusedTrace(traces[0].units, axis, fused * exposures[reference], known, total,
                      tuple(contributions), seams, seam_steps, tuple(sources), traces, match_gain,
                      tuple(float(v) for v in gains), float(exposures[reference]))


def merged_witness(sources, settings: dict | None, position: float, radius: int):
    """Resolve a fused axis position to the originals and rows that built it."""
    fused = fuse(sources, settings)
    index = fused.interval(float(position))
    if index is None:
        return {"state": "outside merged trace", "requested_axis": rounded(position),
                "merged": True, "source": {"meta": {"title": "Merged exposures"}}}, None
    found = fused.contributions[index]
    lead = fused._dominant(index)
    total = sum(item.weight for item in found)
    canvases = []
    details = []
    for item in found:
        source = sources[item.source_index]
        image, reading = annotate_loupe(source.camera, source.trace, item.trace_index, radius)
        details.append({**item.public(), "requested_axis": rounded(position),
                        "source_state": source.trace.row(item.trace_index)["state"],
                        "used_strip_rows": reading["column_result"]["used_strip_rows"],
                        "used_sideband_rows": reading["column_result"]["used_sideband_rows"],
                        "loupe_pixel": reading["pixel"]})
        share = item.weight / total if total > 1e-12 else 0.
        canvases.append((f"{source.title} · col {item.native_column} · {share:.0%}", image))
    if not canvases:
        return {"state": "no trustworthy exposure", "requested_axis": rounded(position),
                "merged": True, "source": {"meta": {"title": "Merged exposures"}}}, None
    witness = _stack(canvases)
    result = {"state": fused.row(index)["state"], "requested_axis": rounded(position),
              "sampled_axis": rounded(fused.axis[index]), "merged": True,
              "fused_light": rounded(fused.signal[index]), "contributing_sources": len(found),
              "leading_source": lead.title if lead else None,
              "contributors": details, "source": {"meta": {"title": "Merged exposures"}}}
    return result, witness


def _stack(panels):
    width = max(image.width for _, image in panels)
    label_height = 20
    height = sum(image.height + label_height for _, image in panels)
    page = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(page)
    top = 0
    for label, image in panels:
        draw.text((4, top + 4), label[:70], fill=CYAN, font=font(12))
        page.paste(image, (0, top + label_height))
        top += image.height + label_height
    return page


def merge_report(bench):
    sources = bench.require(2, 4)
    data = bench.order.settings
    allowed(data, {"match_gain", "show_sources"}, "exposure merge")
    if len({s.basis for s in sources}) != 1:
        raise Problem("Use the same signal channel and axis type for all merged images.")
    match_gain = flag(data.get("match_gain", True), "Match overlap brightness")
    show_sources = flag(data.get("show_sources", True), "Show individual exposures")
    fused = fuse(sources, {"match_gain": match_gain})
    rows = []
    for j in range(len(fused.axis)):
        item = fused.row(j)
        rows.append({"position": item["x_axis"], "units": item["axis_units"],
                     "fused_light": item["linear_signal"], "state": item["state"],
                     "contributing_sources": item["contributing_sources"],
                     "dominant_source": item["source"], "dominant_source_index": item["source_index"],
                     "dominant_column": item["native_column"],
                     "dominant_uv": item["source_uv"], "contributors": item["contributors"]})
    report = Result("merged-spectrum", "Exposure-merged camera spectrum",
                    "Overlapping exposures are brightness-matched and fused into one trace. "
                    "Every sample keeps the originals, native columns and rows behind it; "
                    "positions without a trustworthy exposure stay blank.")
    report.metrics = {**fused.summary(), "observations": len(sources),
                      "source_roles": [s.role for s in sources],
                      "overlap_samples": int(np.sum(np.asarray([len(g) for g in fused.contributions]) > 1))}
    report.rows = rows
    lines = []
    if show_sources:
        for i, source in enumerate(sources):
            xs, ys, absolute = _ordered(source.trace)
            rate = fused.reference_exposure / source.exposure if source.exposure > 0 else 1.
            single = resample(fused.axis, xs, ys * fused.gains[i] * rate, source.trace.known[absolute])
            lines.append(Series(source.title, fused.axis, single, palette(i)))
    lines.append(Series("Merged light", fused.axis, np.where(fused.known, fused.signal, np.nan)))
    chart = plot("Merged exposures", fused.units, "Linear light (reference exposure)", lines)
    report.chart("merged-spectrum", "One trace assembled from the selected exposures", chart)
    report.probes["merged-spectrum.png"] = {**report.probes["merged-spectrum.png"],
                                            "kind": "merged-axis", "source_index": 0}
    for i, source in enumerate(sources):
        report.image(f"source-{i + 1}-strip", f"{source.title}: native camera strip",
                     strip_overlay(source.camera, source.trace), {"kind": "source", "source_index": i})
    report.csv("merged-spectrum", "Fused light and contributing evidence", rows)
    contributors = [{"position": item["position"], **part} for item in rows for part in item["contributors"]]
    report.csv("merge-contributors", "Each original column and weight behind the fused trace", contributors)
    gaps = [{"first_axis": rounded(fused.axis[a]), "last_axis": rounded(fused.axis[b - 1]), "samples": b - a}
            for a, b in ((run.start, run.stop) for run in contiguous(~fused.known))]
    report.csv("merge-gaps", "Positions with no trustworthy exposure", gaps)
    if gaps:
        report.warnings.append(f"{sum(g['samples'] for g in gaps)} merged positions have no trustworthy exposure and remain blank.")
    if match_gain and any(abs(g - 1) > .05 for g in fused.gains):
        report.warnings.append("The overlapping exposures needed brightness matching; compare their capture settings.")
    if fused.seams:
        report.warnings.append(f"The leading exposure changes at {len(fused.seams)} positions; the fused trace is level-matched there.")
    return report
