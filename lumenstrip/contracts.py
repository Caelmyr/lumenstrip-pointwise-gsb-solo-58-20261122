"""Checked values shared by the library, workbench and image processing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
from math import isfinite
from pathlib import PurePath
import re
from typing import Any, Literal
from uuid import uuid4


class Problem(Exception):
    """A condition which the user can fix or which the server can explain."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def uid() -> str:
    return uuid4().hex


def identity(value: Any, label: str = "item") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise Problem(f"Invalid {label} identifier.", 404)
    return value


def token(value: Any, label: str = "file") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise Problem(f"Invalid {label} name.")
    return value


def json_bytes(value: Any, pretty: bool = False) -> bytes:
    args = {"indent": 2} if pretty else {"separators": (",", ":")}
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, **args)
            + ("\n" if pretty else "")).encode("utf-8")


def signature(value: Any) -> str:
    return sha256(json_bytes(value)).hexdigest()


def file_label(value: str) -> str:
    name = PurePath(str(value).replace("\\", "/")).name
    name = "".join(c if c.isalnum() or c in "-_. " else "_" for c in name)
    return name.strip(" .")[:92] or "original-image"


def plain_text(value: Any, label: str, limit: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise Problem(f"{label} must be text.")
    value = " ".join(value.split())
    if len(value) > limit or (required and not value):
        raise Problem(f"{label} must be {'1' if required else '0'}–{limit} characters.")
    return value


def mapping(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise Problem(f"{label} must be an object.")
    return value


def allowed(value: dict, keys: set[str], label: str) -> None:
    extra = value.keys() - keys
    if extra:
        raise Problem(f"Unknown {label} field: {sorted(extra)[0]}.")


def seq(value: Any, label: str, low: int, high: int) -> list:
    if not isinstance(value, list) or not low <= len(value) <= high:
        raise Problem(f"{label} requires {low}–{high} entries.")
    return value


def real(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise Problem(f"{label} must be a number.")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise Problem(f"{label} must be a finite number.") from None
    if not isfinite(number) or not low <= number <= high:
        raise Problem(f"{label} must be between {low:g} and {high:g}.")
    return number


def whole(value: Any, label: str, low: int, high: int) -> int:
    number = real(value, label, low, high)
    if number != int(number):
        raise Problem(f"{label} must be a whole number.")
    return int(number)


def flag(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise Problem(f"{label} must be true or false.")
    return value


def choice(value: Any, values: tuple[str, ...], label: str) -> str:
    if value not in values:
        raise Problem(f"{label} must be one of {', '.join(values)}.")
    return value


def rounded(value, digits: int = 6):
    if value is None:
        return None
    output = float(value)
    return round(output, digits) if isfinite(output) else None


def point(value: Any, label: str) -> tuple[float, float]:
    u, v = seq(value, label, 2, 2)
    return real(u, f"{label} x", 0, 1), real(v, f"{label} y", 0, 1)


def points(value: Any, label: str, low: int, high: int) -> tuple[tuple[float, float], ...]:
    return tuple(point(p, f"{label} {i + 1}") for i, p in enumerate(seq(value, label, low, high)))


CHANNELS = ("linear-luma", "red", "green", "blue", "mean")
ROLES = ("sample", "open-reference", "dark-reference", "calibration", "repeat")


@dataclass(frozen=True)
class ImageCrop:
    bounds: tuple[float, float, float, float]

    @classmethod
    def parse(cls, value) -> ImageCrop:
        first, second = points(value, "Crop corners", 2, 2)
        x0, x1 = sorted((first[0], second[0]))
        y0, y1 = sorted((first[1], second[1]))
        if x1 - x0 < .004 or y1 - y0 < .004:
            raise Problem("The crop must have positive width and height.")
        return cls((x0, y0, x1, y1))

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        u, v, w, h = self.bounds
        return (max(0, min(width - 1, int(u * width))),
                max(0, min(height - 1, int(v * height))),
                max(1, min(width, int(w * width + .999))),
                max(1, min(height, int(h * height + .999))))

    def contains(self, x: float, y: float) -> bool:
        u, v, w, h = self.bounds
        return u <= x <= w and v <= y <= h

    def public(self) -> list[list[float]]:
        u, v, w, h = self.bounds
        return [[u, v], [w, h]]


@dataclass(frozen=True)
class Capture:
    title: str
    role: str
    exposure_ms: float
    notes: str
    synthetic: bool

    @classmethod
    def parse(cls, value) -> Capture:
        data = mapping(value, "Capture")
        allowed(data, {"title", "role", "exposure_ms", "notes", "synthetic"}, "capture")
        return cls(
            plain_text(data.get("title", ""), "Observation name", 90),
            choice(data.get("role", "sample"), ROLES, "Capture role"),
            real(data.get("exposure_ms", 20), "Exposure in milliseconds", .01, 360000),
            plain_text(data.get("notes", ""), "Notes", 320, required=False),
            flag(data.get("synthetic", False), "Synthetic flag"),
        )

    def public(self) -> dict:
        return {"title": self.title, "role": self.role, "exposure_ms": self.exposure_ms,
                "notes": self.notes, "synthetic": self.synthetic}


@dataclass(frozen=True)
class Strip:
    first: tuple[float, float]
    last: tuple[float, float]
    half_height: int
    side_spacing: int
    side_height: int
    channel: str
    level: Literal["mean", "median"]
    subtract_sides: bool
    high_guard: float
    low_guard: float
    step: int

    @classmethod
    def default(cls) -> Strip:
        return cls((.08, .53), (.92, .53), 5, 7, 4, "linear-luma", "median", True, .995, .002, 1)

    @classmethod
    def parse(cls, value) -> Strip:
        data = mapping(value, "Strip")
        allowed(data, {"endpoints", "half_height", "side_spacing", "side_height", "channel",
                       "level", "subtract_sides", "high_guard", "low_guard", "step"}, "strip")
        defaults = cls.default()
        first, last = points(data.get("endpoints", [list(defaults.first), list(defaults.last)]),
                             "Strip center anchors", 2, 2)
        if first[0] > last[0]:
            first, last = last, first
        if last[0] - first[0] < .025:
            raise Problem("Separate the strip anchors horizontally.")
        low = real(data.get("low_guard", defaults.low_guard), "Shadow guard", 0, .9)
        high = real(data.get("high_guard", defaults.high_guard), "Clip guard", .1, 1)
        if high - low < .05:
            raise Problem("Separate the shadow and clip guards by at least 0.05.")
        return cls(first, last,
                   whole(data.get("half_height", defaults.half_height), "Strip half height (px)", 0, 32),
                   whole(data.get("side_spacing", defaults.side_spacing), "Sideband gap (px)", 1, 80),
                   whole(data.get("side_height", defaults.side_height), "Sideband height (px)", 1, 32),
                   choice(data.get("channel", defaults.channel), CHANNELS, "Channel"),
                   choice(data.get("level", defaults.level), ("mean", "median"), "Estimator"),
                   flag(data.get("subtract_sides", defaults.subtract_sides), "Subtract sidebands"), high, low,
                   whole(data.get("step", defaults.step), "Horizontal step (px)", 1, 16))

    def public(self) -> dict:
        return {"endpoints": [list(self.first), list(self.last)], "half_height": self.half_height,
                "side_spacing": self.side_spacing, "side_height": self.side_height,
                "channel": self.channel, "level": self.level, "subtract_sides": self.subtract_sides,
                "high_guard": self.high_guard, "low_guard": self.low_guard, "step": self.step}


@dataclass(frozen=True)
class Window:
    lower: float
    upper: float

    @classmethod
    def parse(cls, value, label: str = "Spectral window") -> Window:
        left, right = seq(value, label, 2, 2)
        a = real(left, f"{label} start", 0, 24000)
        b = real(right, f"{label} end", 0, 24000)
        if b <= a:
            raise Problem(f"{label} must be increasing.")
        return cls(a, b)

    def public(self) -> list[float]:
        return [self.lower, self.upper]

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class NamedWindow:
    name: str
    interval: Window

    @classmethod
    def read(cls, value, i: int) -> NamedWindow:
        data = mapping(value, f"Window {i + 1}")
        allowed(data, {"name", "interval"}, "window")
        return cls(plain_text(data.get("name", f"Band {i + 1}"), "Band name", 55),
                   Window.parse(data.get("interval"), "Band interval"))

    def public(self) -> dict:
        return {"name": self.name, "interval": self.interval.public()}
