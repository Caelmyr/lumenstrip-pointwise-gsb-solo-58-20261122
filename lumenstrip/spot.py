"""Measure a selected real detector spot, its spread and dynamic range."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .camera import channel_values
from .contracts import ImageCrop, Problem, allowed, choice, flag, real, rounded, whole
from .result import Result
from .view import AMBER, BLUE, CANVAS, CYAN, GREEN, PINK, Series, cropped_camera, font, plot


@dataclass(frozen=True)
class Spot:
    rgb: np.ndarray
    alpha: np.ndarray
    light: np.ndarray
    good: np.ndarray
    background: float
    energy: np.ndarray
    offset: tuple[int, int]
    pool: np.ndarray

    @classmethod
    def read(cls, source, crop, bg_edge, clamp):
        camera = source.camera
        bounds = crop.pixels(*camera.size)
        rgb, alpha, offset = camera.crop(bounds)
        values = channel_values(rgb, source.strip.channel)
        if min(values.shape) < 7:
            raise Problem("Choose a detector region at least seven pixels tall and wide.")
        yy, xx = np.indices(values.shape)
        border = ((yy < bg_edge) | (xx < bg_edge) |
                  (xx >= values.shape[1] - bg_edge) | (yy >= values.shape[0] - bg_edge))
        good = alpha >= .995
        clip = np.max(rgb, axis=-1) >= clamp
        valid = good & ~clip
        controls = values[border & valid]
        if len(controls) < 8:
            raise Problem("There are too few measured border pixels to estimate a floor.")
        background = float(np.median(controls))
        residual = np.where(valid, np.maximum(0., values - background), 0.)
        return cls(rgb, alpha, values, valid, background, residual, offset, clip & good)

    @property
    def weight(self):
        return float(np.sum(self.energy))

    @property
    def xy(self):
        y, x = np.indices(self.light.shape, dtype=np.float64)
        return x + self.offset[0], y + self.offset[1]

    def centroid(self):
        if self.weight < 1e-7:
            return None
        return tuple(float(np.sum(axis * self.energy) / self.weight) for axis in self.xy)

    def moments(self):
        center = self.centroid()
        if center is None:
            return None
        x, y = self.xy
        u = x - center[0]
        v = y - center[1]
        cov = np.asarray([[np.sum(u * u * self.energy), np.sum(u * v * self.energy)],
                          [np.sum(u * v * self.energy), np.sum(v * v * self.energy)]]) / self.weight
        eigen, basis = np.linalg.eigh(cov)
        axis = basis[:, 1]
        phase = np.degrees(np.arctan2(axis[1], axis[0])) % 180
        return {"sensor_x_sd": rounded(np.sqrt(max(0, cov[0, 0]))),
                "sensor_y_sd": rounded(np.sqrt(max(0, cov[1, 1]))),
                "short_sd": rounded(np.sqrt(max(0, eigen[0]))),
                "long_sd": rounded(np.sqrt(max(0, eigen[1]))),
                "orientation_deg": rounded(phase)}
    
    def radii(self):
        mean = self.centroid()
        if mean is None:
            return np.empty(0)
        x, y = self.xy
        return np.hypot(x - mean[0], y - mean[1])

    def radius_rows(self, bins):
        radii = self.radii()
        extent = float(np.max(radii)) if radii.size else 1.
        edges = np.linspace(0, extent, bins + 1)
        power = np.histogram(radii, bins=edges, weights=self.energy.reshape(radii.shape) if radii.size else None)[0]
        support = np.histogram(radii, bins=edges, weights=self.good.reshape(radii.shape).astype(np.int64) if radii.size else None)[0]
        certain = np.cumsum(power) / self.weight if self.weight > 1e-7 else np.zeros_like(power)
        return [{"radius_from_px": rounded(edges[i]), "radius_to_px": rounded(edges[i + 1]),
                 "observed_light": rounded(power[i]), "known_pixels": int(support[i]),
                 "share_inside": rounded(certain[i]) if self.weight else None} for i in range(bins)]

    def projection(self):
        x, y = self.xy
        horizontal = self.energy.sum(axis=0)
        vertical = self.energy.sum(axis=1)
        return [(x[0], horizontal, self.good.sum(axis=0), "Columns"),
                (y[:, 0], vertical, self.good.sum(axis=1), "Rows")]


def _annotate(s, crop, object_measure, label):
    camera = s.camera
    x, y, w, h = crop.pixels(*camera.size)
    anchor = object_measure.centroid()
    marked = [] if anchor is None else [(anchor[0], anchor[1], label)]
    return cropped_camera(camera, (x, y, w, h), marks=marked)


def _known_share(spot):
    good = float(np.mean(spot.good))
    clipped = int(np.count_nonzero(spot.pool))
    missing = int(np.count_nonzero(spot.alpha < .995))
    return {"usable_fraction": rounded(good), "clipped_native_samples": clipped,
            "unsampled_pixels": missing, "background_floor": rounded(spot.background)}


def energy_spot(bench):
    s = bench.one()
    cfg = bench.order.settings
    allowed(cfg, {"window", "border", "clip", "bins"}, "detector spot")
    crop = ImageCrop.parse(cfg.get("window", [[.42, .4], [.58, .66]]))
    border = whole(cfg.get("border", 2), "Background border width", 1, 7)
    clip = real(cfg.get("clip", .99), "Clipped camera code", .7, 1.)
    bins = whole(cfg.get("bins", 26), "Radius sample bins", 6, 55)
    spot = Spot.read(s, crop, border, clip)
    anchor = spot.centroid()
    rows = spot.radius_rows(bins)
    p = np.asarray([r["radius_to_px"] for r in rows])
    q = np.asarray([r["share_inside"] if r["share_inside"] is not None else np.nan for r in rows])
    k = np.asarray([r["observed_light"] for r in rows])
    stat = Result("spot-energy", "Detector spot and energy spread",
                  "Measure positive light above the local border floor while retaining the observed detector origin.")
    stat.metrics = {**_known_share(spot), "positive_light": rounded(spot.weight),
                    "centroid_px": [rounded(v) for v in anchor] if anchor is not None else None,
                    "spread": spot.moments(),
                    "radius_median_bin_px": next((r["radius_to_px"] for r in rows
                                                  if r["share_inside"] is not None and r["share_inside"] >= .5), None)}
    stat.rows = rows
    stat.image("spot-witness", "Selected region and observed center", _annotate(s, crop, spot, "Spot center"),
               {"kind": "source-crop", "source_index": 0, "bounds": list(crop.pixels(*s.camera.size))})
    stat.image("spot-energy", "Observed energy by radius", plot(
        "Detector spot", "Distance from observed center (px)", "Accumulated light share",
        [Series("Observed share", p, q)], y_bounds=(0., 1.)).image)
    stat.csv("spot-radial-energy", "Radial energy and native support", rows)
    return stat


def spot_balance(bench):
    source = bench.one()
    cfg = bench.order.settings
    allowed(cfg, {"window", "border", "clip", "normalize"}, "spot projection")
    crop = ImageCrop.parse(cfg.get("window", [[.44, .39], [.62, .72]]))
    border = whole(cfg.get("border", 2), "Background border width", 1, 7)
    clip = real(cfg.get("clip", .99), "Clipped code", .7, 1.)
    normalization = flag(cfg.get("normalize", False), "Normalize projected shares")
    spot = Spot.read(source, crop, border, clip)
    rows = []
    plots = []
    for i, (axis, summed, votes, label) in enumerate(spot.projection()):
        reduced = summed / spot.weight if normalization and spot.weight else summed
        colour = CYAN if i == 0 else AMBER
        graph = plot("Spot " + label.lower(), "Native pixel", "Share" if normalization else "Linear light sum",
                     [Series(label, axis, reduced, colour)])
        plots.append(graph)
        rows.extend({"direction": label, "native_px": int(at), "measured": int(votes[j]),
                     "observed": rounded(reduced[j])} for j, at in enumerate(axis))
    result = Result("spot-projection", "Native spot projections",
                    "Compare the horizontal and vertical measured spread without inventing signal at clipped or missing pixels.")
    result.metrics = {**_known_share(spot), "using_shares": normalization, "described_light": rounded(spot.weight)}
    result.rows = rows
    result.image("spot-projection-x", "Horizontal spot signal", plots[0].image)
    result.image("spot-projection-y", "Vertical spot signal", plots[1].image)
    result.csv("native-spot-projection", "Spot energy projected across native rows and columns", rows)
    return result
