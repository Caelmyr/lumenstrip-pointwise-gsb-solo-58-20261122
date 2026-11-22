"""Resolve chart and image selections to native original pixels."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

from .bench import Source
from .camera import Camera, encode_png
from .contracts import Problem, allowed, choice, identity, mapping, point, real, rounded, whole
from .merge import merged_witness
from .strip import Trace
from .view import AMBER, CANVAS, CYAN, TEXT, annotate_loupe, font


@dataclass(frozen=True)
class Request:
    space: str
    value: float | tuple[float, float]
    radius: int
    source_index: int
    expected_revision: int | None = None
    merged: bool = False
    sources: tuple[str, ...] = ()
    match_gain: bool = True

    @classmethod
    def parse(cls, value):
        data = mapping(value, "Witness request")
        allowed(data, {"on", "value", "radius", "source_index", "expected_revision", "merged", "sources",
                       "match_gain"}, "witness")
        space = choice(data.get("on", "camera"), ("camera", "axis"), "Witness coordinates")
        value = point(data.get("value"), "Position") if space == "camera" else (
            real(data.get("value"), "Axis position", 0, 26000))
        radius = whole(data.get("radius", 12), "Native witness radius", 3, 35)
        chosen = whole(data.get("source_index", 0), "Observation position", 0, 3)
        revision = data.get("expected_revision")
        if revision is not None:
            revision = whole(revision, "Expected source version", 1, 1000000)
        merged = data.get("merged", False)
        if type(merged) is not bool:
            raise Problem("Merged witness flag must be true or false.")
        gain = data.get("match_gain", True)
        if type(gain) is not bool:
            raise Problem("Brightness-matching flag must be true or false.")
        raw = data.get("sources", [])
        if not isinstance(raw, list) or len(raw) > 4:
            raise Problem("Merged witness sources must be a list of up to four observations.")
        return cls(space, value, radius, chosen, revision, merged,
                   tuple(identity(v, "merged observation") for v in raw), gain)


def saved_source(library, snapshot: dict) -> Source:
    key = identity(snapshot.get("id"), "saved observation")
    original = snapshot.get("image", {})
    expected = original.get("sha256")
    raw = library.object(expected)
    camera = Camera.read(raw, original.get("name", key))
    if camera.sha256 != expected:
        raise Problem("Saved reading no longer agrees with its camera fingerprint.", 409)
    return Source(snapshot, snapshot["reading_fit"], camera)


def _near_trace(trace: Trace, value: tuple[float, float]) -> int | None:
    same_column = trace.plan.nearest_uv_column(value[0])
    if same_column is None:
        return None
    y = value[1] * trace.plan.height
    if y < trace.plan.adjacent[same_column, 0] - 1 or y > trace.plan.adjacent[same_column, -1] + 1:
        return None
    return same_column


def _supports_picture(photo: Camera, state: dict, radius: int):
    image, reading = photo.loupe(tuple(state["uv"]), radius)
    draw = ImageDraw.Draw(image)
    ox, oy = reading["loupe_offset"]
    x, y = reading["pixel"]
    size = reading["loupe_factor"]
    lx, ly = (x - ox) * size, (y - oy) * size
    draw.rectangle((lx, ly, lx + size - 1, ly + size - 1),
                   outline=CYAN if state["supported"] else AMBER, width=2)
    return image


def _attach_header(original, title, reading):
    width, height = original.size
    decoded = Image.new("RGB", (max(306, width), height + 48), CANVAS)
    decoded.paste(original, (0, 48), original if original.mode == "RGBA" else None)
    draw = ImageDraw.Draw(decoded)
    draw.text((9, 7), title[:40], fill=TEXT, font=font(13))
    if "pixel" in reading:
        pixel = reading["pixel"]
        supported = reading.get("supported", True)
        label = f"Original {pixel[0]}, {pixel[1]}  /  {'supported' if supported else 'no camera support'}"
        fill = CYAN if supported else AMBER
    else:
        count = len(reading.get("contributors") or [])
        label = f"{reading.get('sampled_axis', reading.get('requested_axis'))}  /  {count} contributing original(s)"
        fill = CYAN if reading.get("state") == "merged" else AMBER
    draw.text((9, 25), label, fill=fill, font=font(11))
    return decoded


def replay(source: Source, request: Request):
    return replay_sources((source,), request)


def replay_sources(sources, request: Request):
    """Resolve a witness against one source or against a set of merged exposures."""
    multi = len(sources) > 1
    if request.expected_revision is not None and any(
            source.record["revision"] != request.expected_revision for source in sources):
        raise Problem("The source changed after this chart was drawn. Run the reading again.", 409)
    if request.space == "axis":
        axis = float(request.value)
        if multi:
            result, image = merged_witness(sources, {"match_gain": request.match_gain},
                                           axis, request.radius)
            if image is not None:
                return result, encode_png(_attach_header(image, "Merged exposures", result))
            return result, None
        source = sources[0]
        trace = source.trace
        index = trace.interval(axis)
        if index is None:
            return {"state": "outside saved trace", "requested_axis": axis, "source": source.public()}, None
        image, reading = annotate_loupe(source.camera, trace, index, request.radius)
        decision = {**reading, "requested_axis": axis, "sampled_axis": rounded(trace.axis[index]),
                    "state": trace.row(index)["state"], "strip": trace.config.public()}
    else:
        uv = request.value
        source = sources[request.source_index % len(sources)]
        reading = source.camera.pixel(uv)
        index = _near_trace(source.trace, uv)
        decision = {**reading, "state": "camera pixel" if reading["supported"] else "no camera support",
                    "column_result": source.trace.row(index, exact=True) if index is not None else None}
        image = _supports_picture(source.camera, reading, request.radius)
    decorated = _attach_header(image, "Merged exposures" if multi else source.title, decision)
    return {**decision, "source": source.public()}, encode_png(decorated)


def bookmark_replay(library, bookmark: dict, radius: int):
    record = bookmark.get("source_snapshot")
    if record is None:
        return {"state": "bookmark predates saved strip geometry",
                "source": library.read_source(bookmark["source"]),
                "bookmark": {"name": bookmark["title"], "created": bookmark["created"],
                             "message": bookmark["message"], "camera_uv": bookmark["uv"]}}, None
    captured = saved_source(library, record)
    request = Request("camera", tuple(bookmark["uv"]), radius, 0)
    result, witness = replay(captured, request)
    current = library.read_source(bookmark["source"])
    return {**result, "bookmark": {"name": bookmark["title"], "created": bookmark["created"],
                                   "message": bookmark["message"], "camera_uv": bookmark["uv"]},
            "retained_source_revision": captured.record["revision"],
            "current_source_revision": current["revision"],
            "live_geometry_differs": current["strip"] != captured.record["strip"],
            "live_fit_differs": current["active_fit"] != bookmark["fit_id"]}, witness
