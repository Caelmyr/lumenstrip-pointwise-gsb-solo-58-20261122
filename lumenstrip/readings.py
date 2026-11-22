"""Routine image inspections and the native spectrum viewer."""

from __future__ import annotations

import numpy as np

from .camera import input_histogram, rgb_stats, saturation, signal_overlay, channel_values, native_grid
from .contracts import ImageCrop, Problem, allowed, choice, flag, real, rounded, whole
from .result import Result
from .strip import SampleBlock, contiguous
from .suggest import influence as anchor_influence
from .view import (AMBER, BLUE, CYAN, GREEN, PINK, Series, cropped_camera, palette, plot,
                   plot_anchor_view, strip_heat, strip_overlay, tile_view)

SOURCE_PROBE = {"kind": "source", "source_index": 0}


def diagnostics(bench):
    source = bench.one()
    cfg = bench.order.settings
    allowed(cfg, {"low", "high", "bins"}, "camera reading")
    low = real(cfg.get("low", .02), "Dark code", .0, .8)
    high = real(cfg.get("high", .985), "Clipped code", .2, 1)
    bins = whole(cfg.get("bins", 28), "Histogram bins", 12, 128)
    if low >= high:
        raise Problem("The dark code must be lower than the clipped code.")
    c = source.camera
    axes, counts = input_histogram(c.rgb, c.alpha, bins)
    centers = (axes[1:] + axes[:-1]) / 2
    n = max(1, int(c.support.sum()))
    frequencies = [v / n for v in counts]
    rows = [{"code_minimum": rounded(axes[i]), "code_maximum": rounded(axes[i + 1]),
             "red_pixels": int(counts[0][i]), "green_pixels": int(counts[1][i]), "blue_pixels": int(counts[2][i])}
            for i in range(bins)]
    result = Result("camera-signal", "Camera codes and unsampled pixels",
                    "False-color camera highlights show clipping, shadow and missing support.")
    result.metrics = {**saturation(c.rgb, c.alpha, low, high), "image": c.public(),
                      "code_summary": rgb_stats(c.rgb[c.support]), "clip_guard": high, "dark_guard": low}
    result.rows = rows
    result.image("camera-signal", "Original camera codes with guards", signal_overlay(c, low, high), SOURCE_PROBE)
    chart = plot("RGB code distribution", "Encoded camera level", "Fraction in bin",
                 [Series(name, centers, value, color)
                  for name, value, color in zip(("Red", "Green", "Blue"), frequencies, (PINK, GREEN, BLUE),
                                                strict=True)], x_bounds=(0., 1.))
    result.image("camera-histogram", "Camera code frequency", chart.image)
    result.csv("camera-code-histogram", "Per-channel camera code histogram", rows)
    return result


def tiles(bench):
    source = bench.one()
    data = bench.order.settings
    allowed(data, {"columns", "rows", "low", "high"}, "camera tile")
    cols = whole(data.get("columns", 8), "Tiles across", 2, 16)
    rows = whole(data.get("rows", 4), "Tiles down", 2, 12)
    low = real(data.get("low", .02), "Dark code", 0, .8)
    high = real(data.get("high", .985), "Clipped code", .2, 1)
    if low >= high:
        raise Problem("The dark code must be lower than the clipped code.")
    diag = native_grid(source.camera, cols, rows, low, high)
    stat = Result("signal-grid", "Camera support by position",
                  "Each cell measures actual native support and channel clipping in that portion of the camera image.")
    stat.metrics = {"tile_count": len(diag), "tiles_with_transparency": sum(x["missing"] > 0 for x in diag),
                    "tiles_with_clipping": sum(x["clip"] > 0 for x in diag),
                    "highest_clip_fraction": rounded(max(x["clip_fraction_of_supported"] or 0 for x in diag))}
    stat.rows = diag
    stat.image("camera-tile-footprint", "Native sampling tiles", tile_view(source.camera, diag), SOURCE_PROBE)
    stat.csv("camera-tile-report", "Tile support by native bounds", diag)
    return stat


def aperture(bench):
    source = bench.one()
    data = bench.order.settings
    allowed(data, {"window", "channel", "low", "high"}, "camera patch")
    crop = ImageCrop.parse(data.get("window", [[.28, .37], [.70, .62]]))
    channel = choice(data.get("channel", "linear-luma"), ("linear-luma", "red", "green", "blue"), "Channel")
    low = real(data.get("low", .02), "Dark code", 0, .8)
    high = real(data.get("high", .985), "Clipped code", .2, 1)
    if low >= high:
        raise Problem("The dark code must be lower than the clipped code.")
    camera = source.camera
    n = crop.pixels(*camera.size)
    rgb, alpha, offset = camera.crop(n)
    p = channel_values(rgb, channel)
    sa = saturation(rgb, alpha, low, high)
    all_pixels = p[alpha >= .995]
    rows = []
    height, width = p.shape
    for i, interval in enumerate(np.array_split(np.arange(width), min(width, 24))):
        subset = p[:, interval]
        good = alpha[:, interval] >= .995
        sampled = subset[good]
        rows.append({"stripe": i + 1, "first_source_column": int(interval[0]) + offset[0],
                     "last_source_column": int(interval[-1]) + offset[0], "samples": int(good.sum()),
                     "linear_signal": rounded(np.mean(sampled)) if len(sampled) else None,
                     "spatial_sd": rounded(np.std(sampled)) if len(sampled) else None})
    stat = Result("camera-patch", "Native patch readout",
                  "The selected camera block is shown at native sampling and summarized one strip of columns at a time.")
    stat.metrics = {"crop_bounds": list(n), "height": height, "width": width, **sa,
                    "mean_linear_" + channel: rounded(np.mean(all_pixels)) if len(all_pixels) else None,
                    "p90_linear_" + channel: rounded(np.quantile(all_pixels, .9)) if len(all_pixels) else None}
    stat.rows = rows
    stat.image("camera-patch", "Selected camera block", cropped_camera(camera, n),
               {"kind": "source-crop", "source_index": 0, "bounds": list(n)})
    stat.csv("patch-signal", "Native columns inside camera block", rows)
    return stat


def spectrum(bench):
    s = bench.one()
    allowed(bench.order.settings, {"show_native", "show_background"}, "trace")
    t = s.trace
    options = bench.order.settings
    lines = [Series("Signal", t.axis, np.where(t.known, t.signal, np.nan))]
    show_native = flag(options.get("show_native", True), "Display native light")
    show_background = flag(options.get("show_background", False), "Display background")
    if show_native:
        lines.append(Series("Strip level", t.axis, np.where(t.known, t.native, np.nan), BLUE))
    if show_background and t.config.subtract_sides:
        lines.append(Series("Sidebands", t.axis, np.where(t.known, t.background, np.nan), AMBER))
    report = Result("spectrum", "Camera spectrum",
                    "The calibrated or native trace uses only the pixels in the recorded strip and sidebands.")
    report.metrics = {**t.summary(), "source": s.title, "channel": t.config.channel,
                      "wavelength_calibration": s.fit.title if s.fit else None}
    report.rows = t.rows()
    report.chart("camera-spectrum", "Linear light along the strip",
                 plot("Camera spectrum", t.units, "Linear signal", lines))
    report.image("strip-origin", "Native strip and sideband positions", strip_overlay(s.camera, t), SOURCE_PROBE)
    report.image("source-rows", "Expanded camera rows", strip_heat(s.camera, t))
    report.csv("spectrum-reading", "Camera strip readings and source coordinates", report.rows)
    return report


def calibration_check(bench):
    s = bench.one()
    allowed(bench.order.settings, {"columns"}, "calibration check")
    if s.fit is None:
        raise Problem("Retain a wavelength calibration before checking it.")
    t = s.trace
    columns = whole(bench.order.settings.get("columns", 65), "Axis inspection samples", 16, 160)
    rows = s.fit.coordinates_rows(s.camera.size[0], columns)
    residual = s.fit.description(s.camera.size[0])
    influence = anchor_influence(s.fit, s.camera.size[0])
    xs = np.asarray([r["relative_column"] for r in rows])
    ys = np.asarray([r["wavelength_nm"] if r["wavelength_nm"] is not None else np.nan for r in rows])
    stat = Result("calibration-check", "Wavelength ruler",
                  "The known anchors, mapped detector columns and excluded outer columns are shown together.")
    stat.metrics = {**residual, "unmeasured_intervals": len(s.fit.excluded),
                    "most_influential_anchor": (influence[0]["name"] if influence else None),
                    "most_influential_change_nm": (influence[0]["largest_reading_change_nm"]
                                                   if influence else None)}
    stat.rows = rows
    chart = plot("Wavelength by camera column", "Relative camera column", "Wavelength (nm)",
                 [Series("Included wavelength support", xs, ys)], x_bounds=(0, 1),
                 markers=[(p.column, p.wavelength, p.name) for p in s.fit.anchors])
    stat.image("wavelength-map", "Saved axis and its anchors", chart.image)
    stat.image("wavelength-anchors", "Anchor columns on the source", plot_anchor_view(s.camera, t, s.fit.anchors),
               SOURCE_PROBE)
    stat.csv("wavelength-columns", "Columns inside and outside wavelength support", rows)
    stat.csv("calibration-residuals", "Anchor fit residuals", residual["anchors"])
    if influence:
        stat.csv("anchor-influence", "Reading change if each anchor is dropped", influence)
    if s.fit.excluded:
        stat.csv("unmeasured-intervals", "Intervals the ruler will not measure",
                 [{"first_relative_column": rounded(low), "last_relative_column": rounded(high),
                   "first_native_px": rounded(low * s.camera.size[0] - .5, 3),
                   "last_native_px": rounded(high * s.camera.size[0] - .5, 3),
                   "first_nm": rounded(float(s.fit._raw(np.asarray([low]))[0])),
                   "last_nm": rounded(float(s.fit._raw(np.asarray([high]))[0]))}
                  for low, high in s.fit.excluded])
    return stat


def _column_summary(sample, rows, axis, known, q):
    p = np.where(sample.allowed, sample.light, np.nan)
    result = []
    for i, index in enumerate(rows):
        row = p[:, i]
        domain = np.isfinite(row) & known
        drawn = row[domain]
        total = len(drawn)
        result.append({"row_offset_px": index, "supported_columns": total,
                       "median_linear": rounded(np.median(drawn)) if total else None,
                       "lower_quantile": rounded(np.quantile(drawn, q)) if total else None,
                       "upper_quantile": rounded(np.quantile(drawn, 1 - q)) if total else None,
                       "first_measured_axis": rounded(axis[np.flatnonzero(domain)[0]]) if total else None,
                       "last_measured_axis": rounded(axis[np.flatnonzero(domain)[-1]]) if total else None})
    return result


def height_scan(bench):
    s = bench.one()
    data = bench.order.settings
    allowed(data, {"radius", "quantile"}, "height scan")
    radius = whole(data.get("radius", 13), "Cross-strip radius", 2, 32)
    q = real(data.get("quantile", .2), "Outside quantile", .01, .49)
    t = s.trace
    indexes = list(range(-radius, radius + 1))
    y = t.plan.centers[:, None] + np.asarray(indexes)
    samples = SampleBlock.read(s.camera, t.plan.x, y, t.config)
    survey = _column_summary(samples, indexes, t.axis, np.isfinite(t.axis), q)
    def values(field):
        return np.asarray([r[field] if r[field] is not None else np.nan for r in survey])
    x = np.asarray(indexes)
    stat = Result("strip-height", "Cross-strip response",
                  "Vertical row offsets have separate support and signal statistics. This can expose a badly placed strip.")
    stat.metrics = {"cross_section_radius": radius, "offset_with_most_support":
                    max(survey, key=lambda row: row["supported_columns"])["row_offset_px"],
                    "measured_columns": len(t.axis), "center": survey[radius],
                    "configured_half_height": t.config.half_height}
    stat.rows = survey
    chart = plot("Per-row cross section", "Vertical offset (native pixels)", "Linear signal", [
        Series("Upper quantile", x, values("upper_quantile"), BLUE),
        Series("Median", x, values("median_linear")),
        Series("Lower quantile", x, values("lower_quantile"), AMBER)])
    stat.image("strip-height", "Vertical trace geometry and signal", chart.image)
    stat.image("placement", "Actual strip placements", strip_overlay(s.camera, t), SOURCE_PROBE)
    stat.csv("vertical-line-profile", "Signal away from strip center", survey)
    return stat


def support_audit(bench):
    s = bench.one()
    allowed(bench.order.settings, {"minimum"}, "support scan")
    minimum = whole(bench.order.settings.get("minimum", 3), "Minimum run columns", 1, 160)
    t = s.trace
    bad = ~t.known
    runs = contiguous(bad)
    data = []
    for run in runs:
        length = run.stop - run.start
        if length < minimum:
            continue
        i = run.start
        j = run.stop - 1
        data.append({"first_source_column": int(t.plan.x[i]), "last_source_column": int(t.plan.x[j]),
                     "first_axis": rounded(t.axis[i]), "last_axis": rounded(t.axis[j]),
                     "declared_columns": length, "start_uv": t.row(i)["source_uv"],
                     "end_uv": t.row(j)["source_uv"],
                     "states": sorted({t.row(x)["state"] for x in range(i, j + 1)})})
    rows = t.rows()
    stat = Result("support-runs", "Unsupported runs in the trace",
                  "Missing image, clipping, dark columns and axis limits remain identifiable runs.")
    stat.metrics = {**t.summary(), "unsupported_runs": len(runs), "listed_runs": len(data),
                    "minimum_run_columns": minimum}
    stat.rows = data
    stat.chart("support-runs", "Measured vs unsupported columns", plot(
        "Strip support", t.units, "Support flag", [Series("Measured", t.axis, t.known.astype(float))],
        y_bounds=(-.08, 1.12)))
    stat.image("support-strip", "Runs on the native camera strip", strip_overlay(s.camera, t), SOURCE_PROBE)
    stat.csv("unsupported-runs", "Unsupported run boundaries", data)
    stat.csv("states-by-column", "All recorded camera positions", rows)
    return stat
