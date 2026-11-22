"""Compare actual RGB channel support, relative signal and chromatic variation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .contracts import NamedWindow, Problem, allowed, flag, real, rounded, seq, whole
from .result import Result
from .spectral import area
from .strip import DARK, CLIPPED, NO_PIXELS, Trace
from .view import AMBER, BLUE, CYAN, GREEN, PINK, Region, Series, plot, strip_overlay

NAMES = ("red", "green", "blue")
COLOURS = (PINK, GREEN, BLUE)


def _capture_rgb(s):
    return [Trace.read(s.camera, replace(s.strip, channel=channel), s.fit) for channel in NAMES]


def _common_axes(traces):
    return traces[0].axis, np.asarray([np.where(t.known, t.signal, np.nan) for t in traces])


def _fit_rows(traces):
    red, green, blue = traces
    input_counts = [np.count_nonzero(t.known) for t in traces]
    all_measured = red.known & green.known & blue.known
    return {"axis": red.units, "measured_column_union": int((red.known | green.known | blue.known).sum()),
            "all_channels_measured": int(all_measured.sum()),
            "known_columns": dict(zip(NAMES, (int(n) for n in input_counts), strict=True)),
            "clipped_columns": {name: int(np.sum(t.statuses == CLIPPED))
                                for name, t in zip(NAMES, traces, strict=True)}}


def channel_balance(bench):
    s = bench.one()
    cfg = bench.order.settings
    allowed(cfg, {"show_shares", "signal_floor"}, "separate channels")
    share = flag(cfg.get("show_shares", False), "Compare channel fraction")
    floor = real(cfg.get("signal_floor", .002), "Signal floor", 1e-7, .5)
    traces = _capture_rgb(s)
    x, values = _common_axes(traces)
    total = np.sum(values, axis=0)
    good = np.all(np.isfinite(values) & (values >= 0), axis=0) & (total >= floor)
    ratios = np.divide(values, total, where=good, out=np.full_like(values, np.nan))
    output = ratios if share else values
    result = Result("rgb-traces", "Separate detector channel spectra",
                    "Each colour channel retains its own camera support. Fractions require all three measured signals.")
    result.metrics = {**_fit_rows(traces), "sum_above_signal_floor": int(good.sum()),
                      "display_shares": share, "signal_floor": floor}
    rows = []
    for j, axis in enumerate(x):
        r = {"axis": rounded(axis), "column": int(traces[0].plan.x[j]),
             "uv": traces[0].row(j)["source_uv"]}
        for i, name in enumerate(NAMES):
            r["linear_" + name] = rounded(values[i, j])
            r["fraction_" + name] = rounded(ratios[i, j])
            r["state_" + name] = traces[i].row(j)["state"]
        rows.append(r)
    result.rows = rows
    result.chart("channel-curves", "Native detector channels", plot(
        "Separate detector channels", traces[0].units, "Fraction" if share else "Linear signal",
        [Series(name.title(), x, output[i], COLOURS[i]) for i, name in enumerate(NAMES)],
        y_bounds=(0., 1.) if share else None))
    result.image("rgb-strip", "Native camera sampling strip", strip_overlay(s.camera, s.trace),
                 {"kind": "source", "source_index": 0})
    result.csv("channel-line-readings", "RGB light and availability", rows)
    return result


def color_bands(bench):
    s = bench.one()
    traces = _capture_rgb(s)
    req = bench.order.settings
    allowed(req, {"bands", "shared"}, "colour intervals")
    same = flag(req.get("shared", True), "Use common channel support")
    x, y = _common_axes(traces)
    mask = np.isfinite(x)
    if np.count_nonzero(mask) < 2:
        raise Problem("At least two axis samples must be available for a band.")
    indices = np.flatnonzero(mask)
    indices = indices[np.argsort(x[indices])]
    x = x[indices]
    y = y[:, indices]
    common = np.all(np.isfinite(y), axis=0)
    raw = req.get("bands") or [{"name": "Visible strip", "interval": [float(x[0]), float(x[-1])]}]
    windows = [NamedWindow.read(v, i) for i, v in enumerate(seq(raw, "RGB band windows", 1, 12))]
    rows = []
    for interval in windows:
        width = interval.interval.width
        lo, hi = interval.interval.lower, interval.interval.upper
        measures = [area(x, item, common if same else np.isfinite(item), lo, hi).public(width) for item in y]
        summary = {"band": interval.name, "bounds": interval.interval.public()}
        total = sum((m["positive_area"] or 0) for m in measures)
        for label, measure in zip(NAMES, measures, strict=True):
            summary[label] = {**measure, "positive_fraction": rounded(measure["positive_area"] / total)
                              if total > 0. and measure["positive_area"] is not None else None}
        rows.append(summary)
    report = Result("channel-bands", "Per-channel band energy",
                    "Compare the exposed detector channels separately on the selected bands.")
    report.metrics = {"axis_units": traces[0].units, "windows": len(rows), "common_required": same,
                      "jointly_measured_columns": int(common.sum())}
    report.rows = rows
    report.chart("channel-band-curves", "Native bands and camera RGB signals",
                 plot("RGB band evidence", traces[0].units, "Linear signal",
                      [Series(name.title(), x, np.where(common, y[i], np.nan) if same else y[i], COLOURS[i])
                       for i, name in enumerate(NAMES)],
                      regions=[Region(w.interval.lower, w.interval.upper, w.name, AMBER) for w in windows]))
    report.csv("channel-window-areas", "Per-channel windows and support measures", rows)
    return report


def _fraction_by_index(traces, blocks):
    x, y = _common_axes(traces)
    idx = np.flatnonzero(np.isfinite(x))
    idx = idx[np.argsort(x[idx])]
    for start in np.array_split(idx, min(len(idx), blocks)):
        counts = []
        for t in traces:
            kinds = t.statuses[start]
            counts.append((np.count_nonzero(kinds == CLIPPED),
                           np.count_nonzero(kinds == DARK),
                           np.count_nonzero(kinds == NO_PIXELS)))
        yield start, counts


def channel_guard(bench):
    s = bench.one()
    req = bench.order.settings
    allowed(req, {"intervals"}, "channel clipping")
    blocks = whole(req.get("intervals", 12), "Intervals", 4, 44)
    traces = _capture_rgb(s)
    rows = []
    if np.any(np.isfinite(traces[0].axis)):
        for axis, values in _fraction_by_index(traces, blocks):
            row = {"start": rounded(traces[0].axis[axis[0]]), "end": rounded(traces[0].axis[axis[-1]]),
                   "read_columns": len(axis)}
            for name, (clip, dark, missing) in zip(NAMES, values, strict=True):
                row[name + "_clip"] = int(clip)
                row[name + "_dark"] = int(dark)
                row[name + "_missing"] = int(missing)
            rows.append(row)
    report = Result("channel-guard", "Detector clipping by channel",
                    "Red, green and blue guard outcomes are recorded independently in the same axes.")
    report.rows = rows
    report.metrics = {**_fit_rows(traces), "axis_bins": len(rows)}
    domain = np.asarray([.5 * (r["start"] + r["end"]) for r in rows])
    graph = [Series(name.title(), domain, np.asarray([r[name + "_clip"] / r["read_columns"] for r in rows]),
                    COLOURS[i]) for i, name in enumerate(NAMES)]
    report.chart("channel-guard", "Share with clipping in each interval",
                 plot("Channel clipping share", traces[0].units, "Fraction of sampled columns", graph,
                      y_bounds=(0., 1.)))
    report.csv("channel-guard", "Dark, clipped and missing columns by channel", rows)
    return report
