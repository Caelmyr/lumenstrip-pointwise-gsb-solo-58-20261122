"""Native row selection, transparent gaps and axis-aware trace readers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import Fit
from .camera import Camera, channel_values
from .contracts import Problem, Strip, rounded

OK = 0
NO_PIXELS = 1
DARK = 2
CLIPPED = 3
NO_BACKGROUND = 4
NO_AXIS = 5
STATUSES = ("measured", "no pixels", "below guard", "clipped", "missing sidebands", "unmapped")


@dataclass(frozen=True)
class RegionPlan:
    x: np.ndarray
    centers: np.ndarray
    middle: np.ndarray
    adjacent: np.ndarray
    width: int
    height: int

    @classmethod
    def make(cls, camera: Camera, settings: Strip) -> RegionPlan:
        w, h = camera.size
        first, last = settings.first, settings.last
        start = max(0, min(w - 1, int(np.ceil(first[0] * w - .5))))
        stop = max(0, min(w - 1, int(np.floor(last[0] * w - .5))))
        if stop - start < 5:
            raise Problem("The strip anchors span fewer than six native columns.")
        x = np.arange(start, stop + 1, settings.step, dtype=np.int32)
        ux = (x + .5) / w
        center = (np.interp(ux, [first[0], last[0]], [first[1], last[1]]) * h -.5)
        center = np.rint(center).astype(np.int32)
        mid = center[:, None] + np.arange(-settings.half_height, settings.half_height + 1)
        offset = settings.half_height + settings.side_spacing
        near = np.arange(offset, offset + settings.side_height)
        sides = center[:, None] + np.concatenate((-near[::-1], near))
        return cls(x, center, mid, sides, w, h)

    @property
    def uv(self) -> np.ndarray:
        return np.column_stack(((self.x + .5) / self.width, np.clip((self.centers + .5) / self.height, 0, 1)))

    def traces(self) -> dict[str, np.ndarray]:
        return {"central": self.centers, "upper": self.middle[:, 0], "lower": self.middle[:, -1],
                "side_upper_top": self.adjacent[:, 0], "side_upper_bottom": self.adjacent[:, self.adjacent.shape[1] // 2 - 1],
                "side_lower_top": self.adjacent[:, self.adjacent.shape[1] // 2],
                "side_lower_bottom": self.adjacent[:, -1]}

    def nearest_uv_column(self, uv: float) -> int | None:
        xp = uv * self.width - .5
        if xp < self.x[0] - .501 or xp > self.x[-1] + .501:
            return None
        return int(np.argmin(np.abs(self.x - xp)))


@dataclass(frozen=True)
class SampleBlock:
    code: np.ndarray
    light: np.ndarray
    allowed: np.ndarray
    observed: np.ndarray
    clip: np.ndarray
    dark: np.ndarray

    @classmethod
    def read(cls, camera: Camera, x, y, settings: Strip) -> SampleBlock:
        inside = (y >= 0) & (y < camera.size[1])
        clipped_y = np.clip(y, 0, camera.size[1] - 1)
        code = camera.rgb[clipped_y, x[:, None], :]
        observed = inside & (camera.alpha[clipped_y, x[:, None]] >= .995)
        if settings.channel in {"red", "blue", "green"}:
            maximum = code[..., ["red", "green", "blue"].index(settings.channel)]
        else:
            maximum = np.max(code, axis=-1)
        clip = observed & (maximum >= settings.high_guard)
        dark = observed & (maximum <= settings.low_guard)
        return cls(code, channel_values(code, settings.channel), observed & ~clip & ~dark,
                   observed, clip, dark)


def _location(samples: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        numerator = np.sum(np.where(mask, samples, 0), axis=1)
        count = mask.sum(axis=1)
        return np.divide(numerator, count, out=np.full(len(count), np.nan), where=count > 0)
    values = np.sort(np.where(mask, samples, np.inf), axis=1)
    n = mask.sum(axis=1)
    low = np.maximum(0, (n - 1) // 2)
    high = np.maximum(0, n // 2)
    rows = np.arange(len(n))
    median = (values[rows, low] + values[rows, high]) / 2
    return np.where(n > 0, median, np.nan)


def _dispersion(samples: np.ndarray, mask: np.ndarray) -> np.ndarray:
    center = _location(samples, mask, "median")
    magnitude = np.abs(samples - center[:, None])
    return _location(magnitude, mask, "median") * 1.4826


@dataclass(frozen=True)
class Trace:
    plan: RegionPlan
    config: Strip
    fit: Fit | None
    axis: np.ndarray
    units: str
    native: np.ndarray
    background: np.ndarray
    signal: np.ndarray
    deviation: np.ndarray
    statuses: np.ndarray
    middle_count: np.ndarray
    side_count: np.ndarray
    clipped_count: np.ndarray
    dark_count: np.ndarray
    middle_support: np.ndarray
    side_support: np.ndarray

    @classmethod
    def read(cls, camera: Camera, settings: Strip, fit: Fit | None) -> Trace:
        plan = RegionPlan.make(camera, settings)
        primary = SampleBlock.read(camera, plan.x, plan.middle, settings)
        sides = SampleBlock.read(camera, plan.x, plan.adjacent, settings)
        signal, status, raw, background = _separate(settings, primary, sides)
        if fit:
            axis, in_axis = fit.mapped(plan.uv[:, 0])
            status = np.where((status == OK) & ~in_axis, NO_AXIS, status)
            units = "nm"
        else:
            axis = plan.x.astype(np.float64)
            units = "px"
        return cls(plan, settings, fit, axis, units, raw, background, signal,
                   _dispersion(primary.light, primary.allowed), status,
                   primary.allowed.sum(axis=1), sides.allowed.sum(axis=1),
                   primary.clip.sum(axis=1), primary.dark.sum(axis=1), primary.allowed, sides.allowed)

    @property
    def known(self) -> np.ndarray:
        return self.statuses == OK

    def axis_array(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.isfinite(self.axis)
        x = self.axis[mask]
        order = np.argsort(x)
        absolute_indexes = np.flatnonzero(mask)[order]
        return x[order], self.signal[absolute_indexes], self.known[absolute_indexes]

    def interval(self, target: float) -> int | None:
        points = np.flatnonzero(np.isfinite(self.axis))
        if not len(points):
            return None
        axis = self.axis[points]
        edge = abs(axis[-1] - axis[0]) / max(1, len(axis) - 1) / 2
        if target < np.min(axis) - edge or target > np.max(axis) + edge:
            return None
        return int(points[np.argmin(np.abs(axis - target))])

    def row(self, i: int, exact: bool = False) -> dict:
        plan = self.plan
        result = {"column_px": int(plan.x[i]), "center_row_px": int(plan.centers[i]),
                  "source_uv": [rounded(q, 8) for q in plan.uv[i]], "x_axis": rounded(self.axis[i]),
                  "axis_units": self.units, "native_signal": rounded(self.native[i]),
                  "sideband_signal": rounded(self.background[i]), "linear_signal": rounded(self.signal[i]),
                  "within_strip_mad": rounded(self.deviation[i]), "state": STATUSES[self.statuses[i]],
                  "support_pixels": int(self.middle_count[i]), "sideband_pixels": int(self.side_count[i]),
                  "clip_pixels": int(self.clipped_count[i]), "dark_pixels": int(self.dark_count[i])}
        if exact:
            result["used_strip_rows"] = plan.middle[i][self.middle_support[i]].astype(int).tolist()
            result["used_sideband_rows"] = (
                plan.adjacent[i][self.side_support[i]].astype(int).tolist() if self.config.subtract_sides else []
            )
            result["requested_strip_rows"] = plan.middle[i].astype(int).tolist()
        return result

    def rows(self):
        return [self.row(i) for i in range(len(self.axis))]

    def summary(self) -> dict:
        finite = self.known & np.isfinite(self.signal)
        stats = np.bincount(self.statuses, minlength=len(STATUSES))
        return {"axis": self.units, "columns": len(self.axis),
                "states": {name: int(n) for name, n in zip(STATUSES, stats, strict=True)},
                "measured_fraction": rounded(stats[OK] / len(self.axis)),
                "minimum_signal": rounded(np.min(self.signal[finite])) if np.any(finite) else None,
                "maximum_signal": rounded(np.max(self.signal[finite])) if np.any(finite) else None,
                "median_signal": rounded(np.median(self.signal[finite])) if np.any(finite) else None,
                "sideband_subtraction": self.config.subtract_sides,
                "strip_height": 2 * self.config.half_height + 1}


def _separate(settings: Strip, block: SampleBlock, side: SampleBlock):
    observed = block.observed.sum(axis=1)
    clipped = block.clip.sum(axis=1)
    dark = block.dark.sum(axis=1)
    usable = block.allowed.sum(axis=1)
    side_support = side.allowed.sum(axis=1)
    raw = _location(block.light, block.allowed, settings.level)
    baseline = _location(side.light, side.allowed, settings.level)
    status = np.full(len(raw), OK, dtype=np.uint8)
    status[observed == 0] = NO_PIXELS
    status[(observed > 0) & (usable == 0) & (dark >= clipped)] = DARK
    status[(observed > 0) & (usable == 0) & (clipped > dark)] = CLIPPED
    status[(usable > 0) & (clipped * 2 >= observed)] = CLIPPED
    if settings.subtract_sides:
        status[(status == OK) & (side_support == 0)] = NO_BACKGROUND
        adjusted = raw - baseline
    else:
        adjusted = raw.copy()
    return np.where(status == OK, adjusted, np.nan), status, raw, baseline


def contiguous(known: np.ndarray) -> list[slice]:
    boundaries = np.flatnonzero(np.diff(np.r_[False, known, False].astype(np.int8)))
    return [slice(int(a), int(b)) for a, b in boundaries.reshape(-1, 2)]


def resample(x_new: np.ndarray, x: np.ndarray, values: np.ndarray, known: np.ndarray) -> np.ndarray:
    result = np.full(len(x_new), np.nan)
    for run in contiguous(known & np.isfinite(values)):
        axes, v = x[run], values[run]
        if len(axes) == 1:
            at = np.isclose(x_new, axes[0], rtol=0, atol=1e-8)
            result[at] = v[0]
        elif len(axes):
            support = (x_new >= axes[0]) & (x_new <= axes[-1])
            result[support] = np.interp(x_new[support], axes, v)
    return result


def regrid(traces: tuple[Trace, ...], count: int) -> tuple[np.ndarray, list[np.ndarray]]:
    units = {trace.units for trace in traces}
    if len(units) != 1:
        raise Problem("Compared images must use the same axis type. Calibrate both sources, or both in native columns.")
    if units == {"px"} and len({trace.plan.width for trace in traces}) > 1:
        raise Problem("Calibrate images of differing native widths before comparing them.")
    inputs = [trace.axis_array() for trace in traces]
    if any(len(a) < 2 for a, _, _ in inputs):
        raise Problem("A selected source has fewer than two calibrated columns in this strip.")
    lower = min(values[0][0] for values in inputs)
    upper = max(values[0][-1] for values in inputs)
    targets = np.linspace(lower, upper, count)
    return targets, [resample(targets, a, v, k) for a, v, k in inputs]
