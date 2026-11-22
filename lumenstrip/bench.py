"""Acquisition context and immutable inputs of a workbench calculation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from threading import BoundedSemaphore

from .calibration import Fit
from .camera import Camera
from .contracts import Problem, Strip, allowed, flag, identity, mapping, plain_text, seq, signature
from .strip import Trace

ALGORITHM_VERSION = "lumenstrip-v1"


class Capacity:
    def __init__(self, slots=2):
        self._slots = BoundedSemaphore(slots)

    def __enter__(self):
        if not self._slots.acquire(timeout=1.0):
            raise Problem("Two readings are already running. Try again shortly.", 429)

    def __exit__(self, exc_type, exc, traceback):
        self._slots.release()
        return False


@dataclass(frozen=True)
class Order:
    observations: tuple[str, ...]
    settings: dict
    strip_versions: dict[str, dict]
    fits: dict[str, str | None]
    retain: bool
    expected: str
    name: str

    @classmethod
    def parse(cls, value):
        data = mapping(value, "Reading request")
        allowed(data, {"sources", "settings", "strips", "fits", "retain", "name", "expected"}, "reading")
        chosen = tuple(identity(v, "observation") for v in seq(data.get("sources"), "Observations", 1, 4))
        if len(set(chosen)) != len(chosen):
            raise Problem("Do not use the same source twice in a reading.")
        overrides = mapping(data.get("fits", {}), "Calibration versions")
        strips = mapping(data.get("strips", {}), "Strip versions")
        if (overrides.keys() | strips.keys()) - set(chosen):
            raise Problem("Choose settings only for the included sources.")
        saved = flag(data.get("retain", False), "Retain reading")
        expected = data.get("expected", "")
        if saved and (not isinstance(expected, str) or len(expected) != 64):
            raise Problem("Compute the reading before retaining it.", 409)
        return cls(chosen, mapping(data.get("settings", {}), "Tool settings"),
                   {key: Strip.parse(raw).public() for key, raw in strips.items()},
                   {key: identity(raw, "calibration") if raw is not None else None
                    for key, raw in overrides.items()}, saved, expected,
                   plain_text(data.get("name", ""), "Reading name", 88, required=saved))

    def public(self):
        return {"sources": list(self.observations), "settings": self.settings,
                "strips": self.strip_versions, "fits": self.fits}


@dataclass(frozen=True)
class Source:
    record: dict
    fit_record: dict | None
    camera: Camera

    @classmethod
    def from_stored(cls, library, record, fit_record, document_key):
        camera = Camera.read(library.object(document_key), record["image"]["name"])
        if record["image"]["sha256"] != camera.sha256:
            raise Problem("Source image fingerprint disagrees with the reading.", 409)
        return cls(record, fit_record, camera)

    @property
    def id(self):
        return self.record["id"]

    @property
    def title(self):
        return self.record["meta"]["title"]

    @property
    def exposure(self):
        return self.record["meta"]["exposure_ms"]

    @property
    def role(self):
        return self.record["meta"]["role"]

    @cached_property
    def strip(self):
        return Strip.parse(self.record["strip"])

    @cached_property
    def fit(self):
        return Fit.parse(self.fit_record["calibration"]) if self.fit_record else None

    @cached_property
    def trace(self):
        return Trace.read(self.camera, self.strip, self.fit)

    @property
    def basis(self):
        return f"{self.strip.channel} / {self.trace.units}"

    def public(self):
        return {**self.record, "reading_fit": self.fit_record}

    def warnings(self):
        warnings = []
        if self.fit is None:
            warnings.append(f"{self.title} is read in native columns.")
        if self.record["meta"]["synthetic"]:
            warnings.append(f"{self.title} is labelled synthetic.")
        if self.camera.public()["supported_fraction"] < .98:
            warnings.append(f"{self.title} contains transparent or partially covered pixels.")
        if self.strip.step > 1:
            warnings.append(f"{self.title} is sampled every {self.strip.step} native columns.")
        if self.strip.subtract_sides and any(
            not 0 < q[1] < 1 for q in [self.strip.first, self.strip.last]
        ):
            warnings.append(f"{self.title} has an anchor on the image boundary.")
        return warnings


@dataclass(frozen=True)
class Bench:
    station: dict
    sources: tuple[Source, ...]
    order: Order

    @classmethod
    def load(cls, library, station, order):
        record, sources = library.acquire(station, order.observations, order.fits, order.strip_versions)
        if sum(s[0]["image"]["width"] * s[0]["image"]["height"] for s in sources) > 15_000_000:
            raise Problem("Together, the selected photos exceed 15 million decoded pixels.", 413)
        resolved = tuple(Source.from_stored(library, *row) for row in sources)
        return cls(record, resolved, order)

    @cached_property
    def merged(self):
        from .merge import fuse
        if len(self.sources) < 2:
            raise Problem("Merging needs at least two camera observations.")
        return fuse(self.sources, {"match_gain": self.order.settings.get("match_gain", True)})

    def trace(self):
        """The single source trace, or a fused trace for a multi-exposure reading."""
        return self.sources[0].trace if len(self.sources) == 1 else self.merged

    def one(self):
        if len(self.sources) != 1:
            raise Problem("Use a single camera observation for this tool.")
        return self.sources[0]

    def require(self, low, high):
        if not low <= len(self.sources) <= high:
            raise Problem(f"Choose {low}–{high} observations for this tool.")
        return self.sources

    def same_basis(self):
        if len({s.basis for s in self.sources}) != 1:
            raise Problem("Use the same signal channel and axis type for all compared images.")
        return self.sources

    def frozen_sources(self):
        return [source.public() for source in self.sources]

    def fingerprint(self, tool):
        return signature({"model": ALGORITHM_VERSION, "station": self.station["id"],
                          "tool": tool, "order": self.order.public(),
                          "sources": self.frozen_sources()})

    def verified_key(self, tool):
        fingerprint = self.fingerprint(tool)
        if self.order.retain and fingerprint != self.order.expected:
            raise Problem("The reading inputs changed. Recompute the preview before retaining.", 409)
        return fingerprint

    def warnings(self):
        return [warning for source in self.sources for warning in source.warnings()]
