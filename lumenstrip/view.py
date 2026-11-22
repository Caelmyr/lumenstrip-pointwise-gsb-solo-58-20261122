"""Plots and source overlays rendered from the same arrays as table readings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import Camera
from .contracts import rounded
from .strip import CLIPPED, NO_PIXELS, OK, Trace, contiguous

CANVAS = (22, 32, 43)
PANEL = (33, 47, 61)
GRID = (58, 74, 89)
TEXT = (231, 241, 241)
QUIET = (157, 177, 190)
CYAN = (66, 218, 212)
AMBER = (247, 190, 99)
PINK = (245, 132, 171)
BLUE = (127, 164, 255)
GREEN = (139, 221, 142)
RED = (251, 111, 97)
RAMP = (CYAN, AMBER, PINK, BLUE, GREEN, RED)


@lru_cache(maxsize=12)
def font(size: int):
    return ImageFont.load_default(size=size)


def palette(i):
    return RAMP[i % len(RAMP)]


def _ticks(start, stop, count=5):
    return np.linspace(start, stop, count + 1)


def _bounds(lines, hard_limits):
    if hard_limits is not None:
        return hard_limits
    finite = [np.asarray(s.values)[np.isfinite(s.values)] for s in lines]
    observed = np.concatenate(finite) if finite else np.empty(0)
    if len(observed) == 0:
        return 0., 1.
    low = float(np.min(observed))
    high = float(np.max(observed))
    span = high - low
    padding = max(.005, .09 * span)
    return low - padding, high + padding


@dataclass(frozen=True)
class Series:
    name: str
    coordinates: np.ndarray
    values: np.ndarray
    color: tuple[int, int, int] = CYAN


@dataclass(frozen=True)
class Region:
    low: float
    high: float
    name: str
    color: tuple[int, int, int] = AMBER


@dataclass(frozen=True)
class Plot:
    image: Image.Image
    x: tuple[float, float]
    y: tuple[float, float]
    area: tuple[int, int, int, int]

    def public(self):
        width, height = self.image.size
        left, top, right, bottom = self.area
        return {"x": list(self.x), "y": list(self.y),
                "plot_uv": [left / width, top / height, right / width, bottom / height]}


def plot(title: str, x_label: str, y_label: str, series: list[Series],
         *, x_bounds=None, y_bounds=None, regions=(), horizontal=None, markers=()) -> Plot:
    width, height = 880, 392
    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 72, 68, width - 26, height - 92
    draw.rounded_rectangle((left - 7, top - 7, right + 7, bottom + 7), radius=7, fill=PANEL)
    draw.text((25, 19), title, font=font(21), fill=TEXT)
    draw.text((left, 44), y_label, font=font(12), fill=QUIET)
    if x_bounds is None:
        support = np.concatenate([s.coordinates[np.isfinite(s.coordinates)] for s in series]) if series else np.empty(0)
        if len(support) >= 2 and np.min(support) != np.max(support):
            x0, x1 = float(np.min(support)), float(np.max(support))
        else:
            x0, x1 = 0., 1.
    else:
        x0, x1 = x_bounds
    y0, y1 = _bounds(series, y_bounds)
    if y0 == y1:
        y0 -= 1
        y1 += 1

    def xpx(x):
        return left + (x - x0) * (right - left) / (x1 - x0)

    def ypx(y):
        return bottom - (y - y0) * (bottom - top) / (y1 - y0)

    for zone in regions:
        start = max(left, xpx(zone.low))
        stop = min(right, xpx(zone.high))
        if stop > start:
            fill = tuple(round(.24 * a + .76 * b) for a, b in zip(zone.color, PANEL, strict=True))
            draw.rectangle((start, top, stop, bottom), fill=fill)
            if stop - start > 58:
                draw.text((start + 4, top + 4), zone.name[:18], fill=zone.color, font=font(10))

    for value in _ticks(x0, x1):
        x = xpx(value)
        draw.line((x, top, x, bottom), fill=GRID)
        draw.text((x, bottom + 9), f"{value:.5g}", anchor="mt", font=font(11), fill=QUIET)
    for value in _ticks(y0, y1):
        y = ypx(value)
        draw.line((left, y, right, y), fill=GRID)
        draw.text((left - 9, y), f"{value:.4g}", anchor="rm", font=font(11), fill=QUIET)
    if horizontal is not None and y0 <= horizontal <= y1:
        y = ypx(horizontal)
        draw.line((left, y, right, y), fill=QUIET, width=1)
    drawing = Image.new("RGBA", (right - left + 1, bottom - top + 1))
    lines = ImageDraw.Draw(drawing)
    for entry in series:
        support = np.isfinite(entry.coordinates) & np.isfinite(entry.values)
        for run in contiguous(support):
            positions = [(xpx(x) - left, ypx(y) - top) for x, y in
                         zip(entry.coordinates[run], entry.values[run], strict=True)]
            if len(positions) >= 2:
                lines.line(positions, fill=entry.color, width=2)
            elif positions:
                x, y = positions[0]
                lines.ellipse((x - 2, y - 2, x + 2, y + 2), fill=entry.color)
    image.paste(drawing, (left, top), drawing)
    for x, y, label in markers:
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        sx, sy = xpx(x), ypx(y)
        draw.line((sx, max(top, sy - 12), sx, min(bottom, sy + 9)), fill=AMBER, width=2)
        draw.ellipse((sx - 2, sy - 2, sx + 2, sy + 2), fill=AMBER)
        if label and sx <= right - 50:
            draw.text((sx + 4, max(top, sy - 14)), str(label)[:9], fill=AMBER, font=font(10))
    draw.text(((left + right) / 2, bottom + 30), x_label, anchor="mt", fill=QUIET, font=font(12))
    posx = left
    for entry in series:
        label = entry.name[:34]
        span = draw.textlength(label, font=font(11))
        if posx + span + 27 > right:
            break
        draw.line((posx, height - 22, posx + 12, height - 22), fill=entry.color, width=2)
        draw.text((posx + 17, height - 28), label, fill=TEXT, font=font(11))
        posx += span + 33
    return Plot(image, (x0, x1), (y0, y1), (left, top, right, bottom))


def _backdrop(camera: Camera, maximum: int = 1040):
    preview = camera.image(maximum)
    backdrop = Image.new("RGB", preview.size, CANVAS)
    backdrop.paste(preview, (0, 0), preview)
    return backdrop


def _point_geom(camera: Camera, image: Image.Image, region, y):
    w, h = camera.size
    return [(int((x + .5) / w * image.width), int(np.clip((v + .5) / h, 0, 1) * image.height))
            for x, v in zip(region, y, strict=True)]


def strip_overlay(camera: Camera, trace: Trace, native_ix: int | None = None) -> Image.Image:
    image = _backdrop(camera)
    draw = ImageDraw.Draw(image)
    for key, y in trace.plan.traces().items():
        color = AMBER if key.startswith("side") else CYAN
        draw.line(_point_geom(camera, image, trace.plan.x, y), fill=color,
                  width=1 if key == "central" or key.startswith("side") else 2)
    for status, color in ((CLIPPED, RED), (NO_PIXELS, BLUE)):
        for run in contiguous(trace.statuses == status):
            positions = _point_geom(camera, image, trace.plan.x[run], trace.plan.centers[run])
            if len(positions) > 1:
                draw.line(positions, fill=color, width=6)
            elif positions:
                x, y = positions[0]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    if native_ix is not None:
        u, v = trace.plan.uv[native_ix]
        x, y = u * image.width, v * image.height
        draw.line((x - 9, y, x + 9, y), fill=TEXT, width=2)
        draw.line((x, y - 9, x, y + 9), fill=TEXT, width=2)
    return image


def plot_anchor_view(camera: Camera, trace: Trace, anchors) -> Image.Image:
    image = strip_overlay(camera, trace)
    draw = ImageDraw.Draw(image)
    for index, anchor in enumerate(anchors):
        x = anchor.column * image.width
        px = min(max(x, 1), image.width - 2)
        color = palette(index)
        draw.line((px, 0, px, image.height), fill=color, width=1)
        y = min(8 + (index % 3) * 19, image.height - 13)
        text = f"{anchor.wavelength:g} nm"
        text_w = draw.textlength(text, font=font(11))
        start = min(max(2, px + 3), max(2, image.width - text_w - 5))
        draw.rectangle((start - 1, y - 1, start + text_w + 2, y + 13), fill=PANEL)
        draw.text((start, y), text, fill=color, font=font(11))
    return image


def strip_heat(camera: Camera, trace: Trace) -> Image.Image:
    plan = trace.plan
    min_row = max(0, int(min(plan.adjacent.min(), plan.middle.min())))
    max_row = min(plan.height, int(max(plan.adjacent.max(), plan.middle.max())) + 1)
    start, stop = plan.x[0], plan.x[-1] + 1
    image = Image.fromarray(camera.rgba()[min_row:max_row, start:stop]).convert("RGB")
    draw = ImageDraw.Draw(image)
    a, b = plan.traces()["upper"] - min_row, plan.traces()["lower"] - min_row
    x = plan.x - start
    draw.line(list(zip(x.tolist(), a.tolist(), strict=True)), fill=CYAN, width=1)
    draw.line(list(zip(x.tolist(), b.tolist(), strict=True)), fill=CYAN, width=1)
    factor = min(4, max(1, 1120 // image.width))
    return image.resize((image.width * factor, image.height * factor), Image.Resampling.NEAREST)


def annotate_loupe(camera: Camera, trace: Trace, i: int, radius: int):
    uv = tuple(trace.plan.uv[i])
    image, point = camera.loupe(uv, radius)
    draw = ImageDraw.Draw(image)
    p = trace.row(i, exact=True)
    ox, oy = point["loupe_offset"]
    scale = point["loupe_factor"]
    x = (p["column_px"] - ox) * scale + scale // 2
    for a, color in (("used_strip_rows", CYAN), ("used_sideband_rows", AMBER)):
        for row in p[a]:
            y = (row - oy) * scale + scale // 2
            if 0 <= y < image.height:
                d = max(2, scale // 4)
                draw.rectangle((x - d, y - d, x + d, y + d), outline=color, width=2)
    return image, {**point, "column_result": p}


def tile_view(camera: Camera, tiles: list[dict]):
    image = _backdrop(camera)
    draw = ImageDraw.Draw(image, mode="RGBA")
    width, height = camera.size
    for t in tiles:
        x, y, x1, y1 = t["bounds_px"]
        a, b, c, d = x * image.width / width, y * image.height / height, x1 * image.width / width, y1 * image.height / height
        supported = t["supported"] / max(1, (x1 - x) * (y1 - y))
        low = t["dark_fraction_of_supported"] or 0
        high = t["clip_fraction_of_supported"] or 0
        color = BLUE if supported < .9 else RED if high > .05 else AMBER if low > .1 else GREEN
        draw.rectangle((a, b, c, d), outline=color + (230,), fill=color + (38,), width=1)
        if c - a > 27 and d - b > 18:
            draw.text((a + 3, b + 3), t["tile"], fill=TEXT + (255,), font=font(11))
    return image


def cropped_camera(camera: Camera, native, *, marks=()):
    rgb, alpha, offset = camera.crop(native)
    image = Image.fromarray(np.rint(np.dstack((rgb, alpha)) * 255).astype(np.uint8))
    if marks:
        draw = ImageDraw.Draw(image)
        for col, row, name in marks:
            x, y = col - offset[0], row - offset[1]
            draw.rectangle((x - 3, y - 3, x + 3, y + 3), outline=CYAN, width=1)
            draw.text((x + 4, y), name, font=font(10), fill=TEXT)
    ratio = max(1, min(5, 1000 // max(image.size)))
    if ratio > 1:
        image = image.resize((image.width * ratio, image.height * ratio), Image.Resampling.NEAREST)
    return image
