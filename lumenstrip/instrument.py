"""Detector repeat, exposure response and strip placement checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import channel_values
from .comparison import _resampled
from .contracts import ImageCrop, NamedWindow, Problem, allowed, choice, real, rounded, seq, whole
from .result import Result
from .strip import SampleBlock
from .view import AMBER, BLUE, CYAN, GREEN, PINK, Series, cropped_camera, palette, plot, strip_overlay


@dataclass(frozen=True)
class DarkBlock:
    row: int
    col: int
    bounds: tuple[int, int, int, int]
    count: int
    level: float | None
    noise: float | None
    hot: int
    hot_points: tuple

    def public(self):
        return {"block": f"{self.row + 1}/{self.col + 1}", "native_bounds": list(self.bounds),
                "measured_pixels": self.count, "median_linear_level": rounded(self.level),
                "median_deviation": rounded(self.noise), "high_outliers": self.hot}


def _dark_tiles(values, support, bounds, tiles, outlier):
    xmin, ymin, xmax, ymax = bounds
    across, down = tiles
    width = xmax - xmin
    height = ymax - ymin
    gx = np.linspace(0, width, across + 1).astype(int)
    gy = np.linspace(0, height, down + 1).astype(int)
    for row in range(down):
        for col in range(across):
            a, b = gx[col:col + 2]
            c, d = gy[row:row + 2]
            patch = values[c:d, a:b]
            pixels = support[c:d, a:b]
            read = patch[pixels]
            if len(read) > 0:
                median = float(np.median(read))
                deviation = float(np.median(np.abs(read - median)))
                threshold = median + max(1e-5, outlier * deviation)
                strong = pixels & (patch > threshold)
                n = int(strong.sum())
                coordinates = np.column_stack(np.nonzero(strong))
                if n > 8:
                    brightness = patch[strong]
                    coordinates = coordinates[np.argsort(brightness)[-8:]]
                native = tuple((int(x + a + xmin), int(y + c + ymin)) for y, x in coordinates)
            else:
                median = deviation = None
                n = 0
                native = ()
            yield DarkBlock(row, col, (int(a + xmin), int(c + ymin), int(b + xmin), int(d + ymin)),
                            len(read), median, deviation, n, native)


def dark_field(bench):
    s = bench.one()
    req = bench.order.settings
    allowed(req, {"window", "across", "down", "outlier"}, "dark field")
    crop = ImageCrop.parse(req.get("window", [[.18, .2], [.84, .78]]))
    across = whole(req.get("across", 5), "Dark tiles across", 2, 12)
    down = whole(req.get("down", 3), "Dark tiles down", 2, 12)
    outlier = real(req.get("outlier", 7), "Outlier deviations", 2, 30)
    bounds = crop.pixels(*s.camera.size)
    encoded, alpha, _ = s.camera.crop(bounds)
    valid = alpha >= .995
    values = channel_values(encoded, s.strip.channel)
    blocks = list(_dark_tiles(values, valid, bounds, (across, down), outlier))
    rows = [tile.public() for tile in blocks]
    hot_pixels = [{"block": f"{tile.row + 1}/{tile.col + 1}", "x": x, "y": y,
                   "linear_value": rounded(values[y - bounds[1], x - bounds[0]])}
                  for tile in blocks for x, y in tile.hot_points]
    chosen = sorted(hot_pixels, key=lambda point: -point["linear_value"])[:30]
    drawn = cropped_camera(s.camera, bounds, marks=[(k["x"], k["y"], "") for k in chosen])
    medians = [tile.level for tile in blocks if tile.level is not None]
    stat = Result("dark-field", "Dark frame and hot pixels",
                  "The selected sample is split into blocks to report the local floor, spatial noise and unusually active pixels.")
    stat.metrics = {"camera_role": s.role, "examined_tiles": len(blocks), "support": int(valid.sum()),
                    "local_floor_span": [rounded(min(medians)), rounded(max(medians))] if medians else None,
                    "hot_pixel_candidates": int(sum(t.hot for t in blocks)),
                    "marked_strongest": len(chosen)}
    stat.rows = rows
    stat.image("dark-field-points", "Strong local outliers on the source", drawn,
               {"kind": "source-crop", "source_index": 0, "bounds": list(bounds)})
    stat.csv("dark-field-blocks", "Median noise and high outlier totals", rows)
    stat.csv("dark-field-pixels", "Native positions of strongest local outliers", hot_pixels)
    if s.role != "dark-reference":
        stat.warnings.append("The current image is not labelled a dark reference.")
    return stat


def _exposure_rows(exposures, estimates):
    observed = np.isfinite(estimates)
    fitted = np.full_like(estimates, np.nan)
    count = int(observed.sum())
    if count >= 2 and float(np.ptp(exposures[observed])) > .001:
        e = exposures[observed]
        v = estimates[observed]
        mean_x, mean_y = np.mean(e), np.mean(v)
        gain = np.dot(e - mean_x, v - mean_y) / np.dot(e - mean_x, e - mean_x)
        intercept = mean_y - gain * mean_x
        fitted = exposures * gain + intercept
    else:
        gain = intercept = np.nan
    residual = np.where(observed, estimates - fitted, np.nan)
    return count, fitted, residual, gain, intercept


def exposure_response(bench):
    bench.require(2, 4)
    sources = tuple(sorted(bench.same_basis(), key=lambda s: (s.exposure, s.id)))
    cfg = bench.order.settings
    allowed(cfg, {"bands", "count", "support"}, "exposure response")
    count = whole(cfg.get("count", 320), "Axis grid size", 40, 1100)
    required_support = real(cfg.get("support", .4), "Minimum compared band fraction", .05, 1)
    x, curves = _resampled(sources, count)
    intervals = cfg.get("bands") or [{"name": "Common exposure region", "interval": [float(max(0, x[0])), float(x[-1])]}]
    windows = [NamedWindow.read(v, k) for k, v in enumerate(seq(intervals, "Response bands", 1, 8))]
    exposure = np.asarray([source.exposure for source in sources])
    values = np.asarray(curves)
    covered = np.all(np.isfinite(values), axis=0)
    rows = []
    table = []
    graph = []
    for index, zone in enumerate(windows):
        inside = (x >= zone.interval.lower) & (x <= zone.interval.upper)
        selected = inside & covered
        p = np.count_nonzero(selected) / max(1, np.count_nonzero(inside))
        estimates = np.median(values[:, selected], axis=1) if p >= required_support else np.full(len(sources), np.nan)
        num, fitted, residual, gain, intercept = _exposure_rows(exposure, estimates)
        rows.append({"band": zone.name, "range": zone.interval.public(), "compared_fraction": rounded(p),
                     "camera_samples": num, "slope_per_millisecond": rounded(gain),
                     "fitted_offset": rounded(intercept),
                     "residual_rmse": rounded(np.sqrt(np.mean(residual[np.isfinite(residual)] ** 2)))
                     if np.any(np.isfinite(residual)) else None})
        graph.append(Series(zone.name + " observed", exposure, estimates, palette(index)))
        graph.append(Series(zone.name + " fit", exposure, fitted, palette(index + 3)))
        for i, source in enumerate(sources):
            table.append({"band": zone.name, "source": source.title, "ms": source.exposure,
                          "observed_median": rounded(estimates[i]), "fitted_signal": rounded(fitted[i]),
                          "residual": rounded(residual[i])})
    report = Result("exposure-response", "Response over known exposures",
                    "The camera's exposure relation is checked from matched spectral bands. Scaled traces are not composited.")
    report.rows = rows
    report.metrics = {"observations": len(sources), "distinct_exposure_times": len(set(exposure)),
                      "bands": len(windows), "minimum_required_support": required_support}
    report.image("exposure-response", "Observed band medians and linear comparison",
                 plot("Exposure response", "Exposure time (ms)", "Median linear signal", graph).image)
    report.csv("exposure-fit", "Exposure fit and common support", rows)
    report.csv("exposure-samples", "Source medians by exposure", table)
    return report


def _strip_sections(s, bands, count, offset, width):
    t = s.trace
    x = t.axis
    vicinity = np.arange(-width, width + 1)
    heights = (t.plan.centers[:, None] + np.arange(-offset, offset + 1)[:, None, None]
               + vicinity[None, None, :])
    for region in bands:
        start, stop = region.interval.lower, region.interval.upper
        chosen = np.flatnonzero(np.isfinite(x) & (x >= start) & (x <= stop))
        if len(chosen) > count:
            rounded_indices = np.rint(np.linspace(0, len(chosen) - 1, count)).astype(int)
            chosen = chosen[rounded_indices]
        if not len(chosen):
            yield region, []
            continue
        output = []
        for j, support in enumerate(heights[:, chosen, :]):
            subset = SampleBlock.read(s.camera, t.plan.x[chosen], support, t.config)
            known = subset.allowed
            good = subset.light[known]
            level = np.median(good) if len(good) else None
            level_mad = np.median(np.abs(good - level)) if level is not None else None
            output.append({"band": region.name, "offset_px": j - offset,
                           "linear_level": rounded(level), "median_deviation": rounded(level_mad),
                           "valid_native_samples": int(known.sum()),
                           "requested_samples": int(known.size),
                           "valid_fraction": rounded(known.sum() / known.size)})
        yield region, output


def trace_placement(bench):
    s = bench.one()
    t = s.trace
    cfg = bench.order.settings
    allowed(cfg, {"bands", "samples", "offset", "width"}, "strip placement")
    count = whole(cfg.get("samples", 100), "Maximum columns per band", 12, 500)
    offset = whole(cfg.get("offset", 16), "Read offset on each side", 4, 36)
    width = whole(cfg.get("width", 1), "Smoothing half-height", 0, 6)
    axis, _, _ = t.axis_array()
    if len(axis) < 2:
        raise Problem("A selected observation has less than two axis columns.")
    intervals = cfg.get("bands") or [{"name": "Whole strip", "interval": [float(axis[0]), float(axis[-1])]}]
    regions = [NamedWindow.read(v, i) for i, v in enumerate(seq(intervals, "Check bands", 1, 8))]
    series = []
    details = []
    rows = []
    for i, (region, read) in enumerate(_strip_sections(s, regions, count, offset, width)):
        measured = [r for r in read if r["linear_level"] is not None]
        best = max(measured, key=lambda r: (r["linear_level"], -abs(r["offset_px"]))) if measured else None
        rows.append({"band": region.name, "limits": region.interval.public(), "measured_offsets": len(measured),
                     "brightest_observed_offset": best["offset_px"] if best else None,
                     "center_signal": read[offset]["linear_level"] if read else None,
                     "brightest_signal": best["linear_level"] if best else None,
                     "winner_pixel_support": best["valid_fraction"] if best else None})
        if read:
            series.append(Series(region.name, np.asarray([r["offset_px"] for r in read]),
                                 np.asarray([r["linear_level"] if r["linear_level"] is not None else np.nan
                                             for r in read]), palette(i)))
        details.extend(read)
    result = Result("trace-placement", "Strip placement at selected wavelengths",
                    "Different wavelength regions can have different visible height peaks. The saved strip geometry does not move.")
    result.rows = rows
    result.metrics = {"requested_windows": len(regions), "scan_half_height": offset, "checked_axis": t.units}
    result.image("placement-history", "Vertical placement by band",
                 plot("Region signal across strip height", "Offset from the saved strip (px)",
                      "Median linear light", series, x_bounds=(-float(offset), float(offset))).image)
    result.image("placement-source", "Saved strip geometry", strip_overlay(s.camera, t),
                 {"kind": "source", "source_index": 0})
    result.csv("placement-summary", "Brightest observed offset by spectral band", rows)
    result.csv("placement-observations", "Vertical band samples", details)
    return result
