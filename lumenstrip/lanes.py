"""Read independent light bands photographed together on one detector."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw

from .camera import channel_values
from .contracts import ImageCrop, Problem, allowed, choice, flag, mapping, plain_text, points, rounded, seq, whole
from .result import Result
from .strip import Trace, regrid
from .view import AMBER, BLUE, CANVAS, CYAN, GREEN, PINK, TEXT, Series, font, palette, plot


@dataclass(frozen=True)
class Lane:
    name: str
    endpoints: tuple[tuple[float, float], tuple[float, float]]
    radius: int

    @classmethod
    def read(cls, value, index):
        data = mapping(value, "Parallel strip")
        allowed(data, {"name", "points", "radius"}, "parallel strip")
        p, q = points(data.get("points"), "Center points", 2, 2)
        if p[0] > q[0]:
            p, q = q, p
        if q[0] - p[0] < .05:
            raise Problem("Each parallel strip needs at least 5% of the camera width.")
        return cls(plain_text(data.get("name", f"Strip {index + 1}"), "Lane name", 45),
                   (p, q), whole(data.get("radius", 4), "Native half-height", 0, 16))

    def public(self):
        return {"name": self.name, "points": [list(p) for p in self.endpoints], "radius": self.radius}

    def trace(self, source):
        settings = replace(source.strip, first=self.endpoints[0], last=self.endpoints[1], half_height=self.radius)
        return Trace.read(source.camera, settings, source.fit)


DEFAULT = [
    {"name": "Upper strip", "points": [[.09, .43], [.91, .43]], "radius": 4},
    {"name": "Lower strip", "points": [[.09, .56], [.91, .56]], "radius": 4},
]


def _unpack(source, raw):
    choices = [Lane.read(v, i) for i, v in enumerate(seq(raw, "Parallel strips", 2, 6))]
    names = [one.name for one in choices]
    if len(names) != len(set(names)):
        raise Problem("Give each compared strip a distinct name.")
    return choices, tuple(one.trace(source) for one in choices)


def _combined_limit(curve, normalize):
    measured = curve[np.isfinite(curve)]
    if normalize == "none" or not len(measured):
        return curve, 1.
    maximum = float(np.max(measured))
    if maximum <= 0.:
        return curve, 1.
    return curve / maximum, maximum


def _row(phase, axis, values, traces):
    result = {"position": rounded(axis)}
    for i, (values_here, trace) in enumerate(zip(values, traces, strict=True)):
        sample = values_here[phase]
        point = trace.interval(axis) if np.isfinite(sample) else None
        result[f"trace_{i + 1}_level"] = rounded(sample)
        result[f"trace_{i + 1}_source_uv"] = trace.row(point)["source_uv"] if point is not None else None
    return result


def _overlay(source, choices, traces):
    p = source.camera.image(1100)
    canvas = Image.new("RGB", p.size, CANVAS)
    canvas.paste(p, (0, 0), p)
    draw = ImageDraw.Draw(canvas)
    camera_width, camera_height = source.camera.size
    for i, (name, t) in enumerate(zip(choices, traces, strict=True)):
        plan = t.plan
        x = canvas.width * (plan.x + .5) / camera_width
        upper = canvas.height * (plan.middle[:, 0] + .5) / camera_height
        lower = canvas.height * (plan.middle[:, -1] + .5) / camera_height
        colour = palette(i)
        draw.line(list(zip(x, upper, strict=True)), fill=colour, width=2)
        draw.line(list(zip(x, lower, strict=True)), fill=colour, width=2)
        y = int(np.clip(upper[0] - 13, 1, canvas.height - 15))
        draw.text((max(4, x[0] + 3), y), name.name, font=font(11), fill=colour)
    return canvas


def _lane_metrics(lane, trace, divisor, snapshots):
    allowed = trace.known
    available = snapshots[np.isfinite(snapshots)]
    return {"strip": lane.name, "sample_count": len(trace.plan.x),
            "measured_columns": int(allowed.sum()), "normalized_divisor": rounded(divisor),
            "min_measured": rounded(available.min()) if len(available) else None,
            "max_measured": rounded(available.max()) if len(available) else None,
            "camera_scope": lane.public()}


def parallel(bench):
    s = bench.one()
    settings = bench.order.settings
    allowed(settings, {"lanes", "resample", "normalize"}, "parallel traces")
    count = whole(settings.get("resample", 360), "Grid samples", 40, 1100)
    normalized = choice(settings.get("normalize", "none"), ("none", "peak"), "Display normalization")
    definitions, traces = _unpack(s, settings.get("lanes", DEFAULT))
    x, raw = regrid(traces, count)
    adjusted = [_combined_limit(line, normalized) for line in raw]
    curves = [one for one, _ in adjusted]
    common = np.logical_and.reduce([np.isfinite(line) for line in curves])
    summary = [_lane_metrics(item, trace, scale, vals) for item, trace, (vals, scale)
               in zip(definitions, traces, adjusted, strict=True)]
    result = Result("parallel-strips", "Side-by-side detector strips",
                    "Read multiple independent strips from the same camera image, with each strip retaining its geometry.")
    result.metrics = {"compared_strips": len(traces), "grid_samples": len(x),
                      "common_samples": int(common.sum()), "coordinate_units": traces[0].units,
                      "curve_normalization": normalized}
    result.rows = summary
    result.image("parallel-camera", "Native detector strip positions", _overlay(s, definitions, traces),
                 {"kind": "source", "source_index": 0})
    graph = plot("Parallel detector strips", traces[0].units,
                 "Peak-relative signal" if normalized == "peak" else "Linear signal",
                 [Series(lane.name, x, curve, palette(i)) for i, (lane, curve)
                  in enumerate(zip(definitions, curves, strict=True))])
    result.image("parallel-strip-readings", "Common-axis parallel strip comparison", graph.image)
    result.csv("parallel-strip-metrics", "Independent strip geometry and support", summary)
    result.csv("parallel-profile-readings", "Strip readings and actual camera coordinates",
               [_row(i, axis, curves, traces) for i, axis in enumerate(x)])
    return result


def _axis_bands(signal, support, split, axis, offset, threshold):
    slices = np.array_split(np.arange(signal.shape[axis]), split)
    result = []
    for choice in slices:
        if not len(choice):
            continue
        j = int(choice[0])
        k = int(choice[-1] + 1)
        piece = signal[j:k, :] if axis == 0 else signal[:, j:k]
        valid = support[j:k, :] if axis == 0 else support[:, j:k]
        read = piece[valid]
        n = len(read)
        p = float(np.mean(read >= threshold)) if n else None
        result.append({"start_native_px": j + offset, "end_native_px_exclusive": k + offset,
                       "native_pixels": int(valid.size), "supported_pixels": n,
                       "supported_fraction": rounded(n / max(1, valid.size)),
                       "mean": rounded(np.mean(read)) if n else None,
                       "median": rounded(np.median(read)) if n else None,
                       "low": rounded(np.quantile(read, .1)) if n else None,
                       "high": rounded(np.quantile(read, .9)) if n else None,
                       "bright_pixel_fraction": rounded(p)})
    return result


def projection(bench):
    s = bench.one()
    req = bench.order.settings
    allowed(req, {"window", "direction", "samples", "threshold"}, "detector projection")
    patch = ImageCrop.parse(req.get("window", [[.20, .31], [.80, .72]]))
    axis = choice(req.get("direction", "horizontal"), ("horizontal", "vertical"), "Projection direction")
    bins = whole(req.get("samples", 48), "Projection intervals", 4, 180)
    from .contracts import real
    limit = real(req.get("threshold", .07), "Bright-pixel threshold", 0., 1.)
    bounds = patch.pixels(*s.camera.size)
    rgb, alpha, offsets = s.camera.crop(bounds)
    channel = channel_values(rgb, s.strip.channel)
    measured = alpha >= .995
    rows = _axis_bands(channel, measured, bins, 1 if axis == "horizontal" else 0,
                       offsets[0] if axis == "horizontal" else offsets[1], limit)
    x = np.asarray([.5 * (r["start_native_px"] + r["end_native_px_exclusive"] - 1) for r in rows])
    levels = [("median", "Median", CYAN), ("high", "90% signal", GREEN), ("low", "10% signal", BLUE)]
    graph = [Series(label, x, np.asarray([r[key] if r[key] is not None else np.nan for r in rows]), colour)
             for key, label, colour in levels]
    report = Result("camera-projection", "Camera patch projection",
                    "Project a selected native camera area across rows or columns to inspect light distribution and available support.")
    report.metrics = {"axis": axis, "native_bounds": list(bounds), "requested_bins": bins,
                      "occupied_bins": len(rows), "camera_pixels": int(measured.sum()),
                      "bright_threshold": limit, "signal_channel": s.strip.channel}
    report.rows = rows
    report.image("camera-projection", "Projected light inside the native patch",
                 plot("Camera patch projection", f"Native {axis} position (px)", "Linear signal", graph).image)
    report.csv("camera-projection", "Native projection values and support by interval", rows)
    return report
