"""Wavelength calibration from user-measured image and reference coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import Problem, allowed, choice, mapping, plain_text, real, rounded, seq

MODES = ("linear", "interpolate")


@dataclass(frozen=True)
class Anchor:
    column: float
    wavelength: float
    name: str
    weight: float

    @classmethod
    def parse(cls, value, index: int) -> Anchor:
        data = mapping(value, "Wavelength anchor")
        allowed(data, {"column", "nm", "name", "weight"}, "anchor")
        return cls(real(data.get("column"), "Relative column", 0, 1),
                   real(data.get("nm"), "Wavelength (nm)", 320, 1200),
                   plain_text(data.get("name", f"Line {index + 1}"), "Anchor name", 58),
                   real(data.get("weight", 1.), "Anchor weight", .05, 50))

    def public(self):
        return {"column": self.column, "nm": self.wavelength, "name": self.name, "weight": self.weight}


@dataclass(frozen=True)
class Fit:
    method: str
    anchors: tuple[Anchor, ...]
    limit: str
    title: str
    slope: float
    intercept: float
    excluded: tuple[tuple[float, float], ...] = ()

    @classmethod
    def parse(cls, value: dict) -> Fit:
        data = mapping(value, "Wavelength calibration")
        allowed(data, {"method", "anchors", "limit", "title", "excluded"}, "calibration")
        method = choice(data.get("method", "linear"), MODES, "Calibration method")
        limit = choice(data.get("limit", "anchors"), ("anchors", "frame"), "Axis support")
        name = plain_text(data.get("title", "Wavelength calibration"), "Calibration name", 86)
        anchors = sorted((Anchor.parse(v, i) for i, v in
                          enumerate(seq(data.get("anchors"), "Known wavelengths", 2, 28))),
                         key=lambda a: a.column)
        excluded = []
        for i, pair in enumerate(seq(data.get("excluded") or [], "Unmeasured intervals", 0, 12)):
            low, high = seq(pair, f"Unmeasured interval {i + 1}", 2, 2)
            a = real(low, f"Unmeasured interval {i + 1} start", 0, 1)
            b = real(high, f"Unmeasured interval {i + 1} end", 0, 1)
            if b - a < 1e-6:
                raise Problem("An unmeasured interval must be wider than one millionth of the image.")
            excluded.append((a, b))
        excluded.sort()
        if any(b > c for (_, b), (c, _) in zip(excluded, excluded[1:])):
            raise Problem("Unmeasured intervals must not overlap.")
        u = np.asarray([a.column for a in anchors])
        w = np.asarray([a.wavelength for a in anchors])
        weights = np.asarray([a.weight for a in anchors])
        if min(np.diff(u)) < .0005 or u[-1] - u[0] < .035:
            raise Problem("Calibration anchors must be distinct and cover at least 3.5% of the image width.")
        delta = np.diff(w)
        if not (np.all(delta > .02) or np.all(delta < -.02)):
            raise Problem("Wavelength anchors must move in one direction with the corresponding image columns.")
        mean_u = np.average(u, weights=weights)
        mean_w = np.average(w, weights=weights)
        slope = np.dot(weights * (u - mean_u), w - mean_w) / np.dot(weights, (u - mean_u) ** 2)
        return cls(method, tuple(anchors), limit, name, float(slope), float(mean_w - slope * mean_u),
                   tuple(excluded))

    @property
    def columns(self) -> tuple[float, float]:
        return self.anchors[0].column, self.anchors[-1].column

    @property
    def wavelength_extent(self) -> tuple[float, float]:
        lower, upper = self.columns if self.limit == "anchors" else (0., 1.)
        values, supported = self.mapped(np.asarray([lower, upper]))
        if not supported.any():
            return float(sorted(values)[0]), float(sorted(values)[-1])
        return tuple(sorted(float(v) for v in values[supported]))

    def _raw(self, columns: np.ndarray) -> np.ndarray:
        if self.method == "linear":
            return self.intercept + columns * self.slope
        controls = np.asarray([a.column for a in self.anchors])
        wavelengths = np.asarray([a.wavelength for a in self.anchors])
        mapped = np.interp(columns, controls, wavelengths)
        for index in (0, -2):
            boundary = 0 if index == 0 else -1
            face = columns < controls[0] if index == 0 else columns > controls[-1]
            slope = (wavelengths[index + 1] - wavelengths[index]) / (controls[index + 1] - controls[index])
            mapped = np.where(face, wavelengths[boundary] + (columns - controls[boundary]) * slope, mapped)
        return mapped

    def mapped(self, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        columns = np.asarray(columns, np.float64)
        lower, upper = self.columns if self.limit == "anchors" else (0., 1.)
        supported = (columns >= lower - 1e-12) & (columns <= upper + 1e-12)
        for a, b in self.excluded:
            supported &= ~((columns >= a - 1e-12) & (columns <= b + 1e-12))
        mapped = self._raw(columns)
        return np.where(supported, mapped, np.nan), supported

    def inverse(self, nm: float) -> float | None:
        """Detector column for a wavelength, or None when that wavelength is not measured."""
        source = np.asarray([a.column for a in self.anchors] if self.limit == "anchors" else
                            [0., *[a.column for a in self.anchors if 0 < a.column < 1], 1.])
        mapped = self._raw(source)
        order = np.argsort(mapped)
        source, mapped = source[order], mapped[order]
        if nm < mapped[0] or nm > mapped[-1]:
            return None
        column = float(np.interp(nm, mapped, source))
        if any(a - 1e-12 <= column <= b + 1e-12 for a, b in self.excluded):
            return None
        return column

    def public(self) -> dict:
        return {"title": self.title, "method": self.method, "limit": self.limit,
                "anchors": [a.public() for a in self.anchors],
                "excluded": [list(pair) for pair in self.excluded]}

    def description(self, width: int) -> dict:
        u = np.asarray([a.column for a in self.anchors])
        observed = np.asarray([a.wavelength for a in self.anchors])
        fitted = self._raw(u)
        residual = observed - fitted
        details = [{"name": a.name, "relative_column": a.column,
                    "x_pixel": rounded(a.column * width - .5, 3), "anchor_nm": a.wavelength,
                    "mapped_nm": rounded(fitted[i]), "residual_nm": rounded(residual[i]), "weight": a.weight}
                   for i, a in enumerate(self.anchors)]
        lower, upper = self.wavelength_extent
        return {"title": self.title, "method": self.method, "support": self.limit,
                "anchor_rmse_nm": rounded(np.sqrt(np.mean(residual ** 2))),
                "maximum_anchor_error_nm": rounded(np.max(np.abs(residual))),
                "nm_per_pixel_linear_fit": rounded(self.slope / width), "reversed": self.slope < 0,
                "mapped_extent_nm": [rounded(lower), rounded(upper)],
                "unmeasured_intervals": len(self.excluded),
                "anchors": details}

    def coordinates_rows(self, width: int, count: int = 65) -> list[dict]:
        columns = np.linspace(0, 1, count)
        values, mask = self.mapped(columns)
        edges = self._raw(columns)
        return [{"relative_column": rounded(column), "x_from_left_px": rounded(column * width, 3),
                 "wavelength_nm": rounded(values[i]), "inside_calibrated_extent": bool(mask[i]),
                 "local_bandwidth_nm_per_px": rounded((edges[min(i + 1, count - 1)] - edges[max(i - 1, 0)])
                                                       / (columns[min(i + 1, count - 1)] - columns[max(i - 1, 0)])
                                                       / width)}
                for i, column in enumerate(columns)]
