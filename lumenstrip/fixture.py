"""Generate inspectable, explicitly synthetic camera images for bench setup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import encode_png, transfer
from .contracts import Capture, ImageCrop, Problem, Strip, allowed, choice, flag, mapping, plain_text, point, real, seq, whole


@dataclass(frozen=True)
class SyntheticLine:
    name: str
    u: float
    height: float
    width: float
    color: str

    @classmethod
    def parse(cls, value, index):
        data = mapping(value, "Generated line")
        allowed(data, {"name", "column", "height", "width", "color"}, "synthetic line")
        return cls(plain_text(data.get("name", f"Line {index + 1}"), "Line title", 46),
                   real(data.get("column", .5), "Line column", .01, .99),
                   real(data.get("height", .4), "Line intensity", .001, 2.),
                   real(data.get("width", 3), "Line sigma (px)", .4, 120),
                   choice(data.get("color", "white"), ("white", "red", "green", "blue", "amber"), "Line color"))

    def public(self):
        return {"name": self.name, "column": self.u, "height": self.height, "width": self.width, "color": self.color}


@dataclass(frozen=True)
class Pattern:
    title: str
    width: int
    height: int
    start: tuple[float, float]
    end: tuple[float, float]
    sigma: float
    continuum: float
    pedestal: float
    noise: float
    exposure: float
    flipped: bool
    seed: int
    lines: tuple[SyntheticLine, ...]
    holes: tuple[ImageCrop, ...]

    @classmethod
    def parse(cls, value):
        doc = mapping(value, "Synthetic detector")
        allowed(doc, {"title", "size", "start", "end", "sigma", "continuum",
                      "pedestal", "noise", "exposure", "lines", "holes", "seed", "flipped"}, "synthetic detector")
        name = plain_text(doc.get("title", "Synthetic calibration strip"), "Camera name", 85)
        w, h = seq(doc.get("size", [780, 256]), "Canvas dimensions", 2, 2)
        width = whole(w, "Sensor width", 160, 1800)
        height = whole(h, "Sensor height", 100, 650)
        start = point(doc.get("start", [.05, .48]), "Left slit position")
        end = point(doc.get("end", [.95, .52]), "Right slit position")
        if end[0] - start[0] < .15:
            raise Problem("Separate the left and right slit positions by at least 15% of the image.")
        sigma = real(doc.get("sigma", 3), "Vertical light spread (px)", .75, min(45, height / 4))
        noise = real(doc.get("noise", .0014), "Read noise (linear)", 0, .08)
        continuum = real(doc.get("continuum", .10), "Continuum base", 0, .95)
        pedestal = real(doc.get("pedestal", .011), "Background level", 0, .65)
        exposure = real(doc.get("exposure", 30.), "Exposure in milliseconds", .01, 300)
        default = [{"name": "Green line", "column": .50, "width": 2.5, "color": "green"},
                   {"name": "Blue line", "column": .26, "width": 3., "color": "blue"},
                   {"name": "Orange line", "column": .77, "width": 4., "color": "amber"}]
        lines = tuple(SyntheticLine.parse(row, i) for i, row in enumerate(
            seq(doc.get("lines", default), "Synthetic lines", 0, 24)))
        holes = tuple(ImageCrop.parse(v) for v in seq(doc.get("holes", []), "Missing regions", 0, 12))
        seed = whole(doc.get("seed", 1487), "Noise seed", 0, 100000)
        return cls(name, width, height, start, end, sigma, continuum, pedestal, noise, exposure,
                   flag(doc.get("flipped", False), "Flip dispersion"), seed, lines, holes)

    def _axis(self):
        x = (np.arange(self.width) + .5) / self.width
        y = (np.arange(self.height) + .5) / self.height
        slit = np.interp(x, [self.start[0], self.end[0]], [self.start[1], self.end[1]])
        band = np.exp(-.5 * ((y[:, None] - slit[None, :]) / self.sigma * self.height) ** 2)
        return x, y, band

    def _noise(self):
        generator = np.random.default_rng(self.seed)
        return generator.normal(0., self.noise, (self.height, self.width, 1))

    def _spectrum(self, x):
        intensity = self.continuum * (1. + .17 * np.sin(np.pi * x))
        curve = np.repeat(intensity[:, None], 3, axis=1)
        colors = {"white": [1., 1., 1.], "blue": [.18, .30, 1.], "red": [1., .22, .15],
                  "green": [.18, 1., .38], "amber": [1., .61, .18]}
        for item in self.lines:
            p = 1. - item.u if self.flipped else item.u
            shape = item.height * np.exp(-.5 * ((x - p) * self.width / item.width) ** 2)
            curve += shape[:, None] * np.asarray(colors[item.color])
        return curve

    def render(self):
        x, y, pixels = self._axis()
        spectrum = self._spectrum(x)
        limited = (x[None, :] >= self.start[0]) & (x[None, :] <= self.end[0])
        ramp = self.pedestal * (1. + .18 * y[:, None] + .11 * x[None, :])
        linear = ramp[..., None] + self._noise() + np.where(limited, pixels, 0.)[..., None] * spectrum[None] * self.exposure / 30
        rgb = np.rint(transfer(linear) * 255).astype(np.uint8)
        alpha = np.full((self.height, self.width), 255, dtype=np.uint8)
        for hole in self.holes:
            a, b, c, d = hole.pixels(self.width, self.height)
            alpha[b:d, a:c] = 0
        return encode_png(np.dstack((rgb, alpha)))

    def scale(self):
        width = int(round(self.sigma * 1.5))
        raw = Strip.default().public()
        raw.update(endpoints=[list(self.start), list(self.end)], half_height=width, side_spacing=width + 2)
        return Strip.parse(raw)

    def description(self):
        return {"native_size": [self.width, self.height], "slit": [list(self.start), list(self.end)],
                "native_row_sigma": self.sigma, "requested_lines": [item.public() for item in self.lines],
                "transparent_regions": [item.public() for item in self.holes], "random_seed": self.seed,
                "exposure_time_ms": self.exposure, "synthetic": True}

    def capture(self):
        return Capture.parse({"title": self.title, "role": "calibration", "exposure_ms": self.exposure,
                              "synthetic": True, "notes": f"Generated detector image / deterministic seed {self.seed}."})
