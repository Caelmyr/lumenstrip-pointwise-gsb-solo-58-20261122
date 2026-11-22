"""Resolve chart and image selections to native original pixels."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

from .bench import Source
from .camera import Camera, encode_png
from .contracts import Problem, allowed, choice, identity, mapping, point, real, rounded, whole
from .strip import Trace
from .view import AMBER, CANVAS, CYAN, TEXT, annotate_loupe, font


@dataclass(frozen=True)
class Request:
    space: str
    value: float | tuple[float, float]
    radius: int
    source_index: int
    expected_revision: int | None = None
    source_index_2: int | None = None
    applied_shift: float | None = None

    @classmethod
    def parse(cls, value):
        data = mapping(value, "Witness request")
        allowed(data, {"on", "value", "radius", "source_index", "expected_revision",
                       "source_index_2", "applied_shift"}, "witness")
        space = choice(data.get("on", "camera"), ("camera", "axis", "paired-axis"), "Witness coordinates")
        value = point(data.get("value"), "Position") if space == "camera" else (
            real(data.get("value"), "Axis position", 0, 26000))
        radius = whole(data.get("radius", 12), "Native witness radius", 3, 35)
        chosen = whole(data.get("source_index", 0), "Observation position", 0, 3)
        revision = data.get("expected_revision")
        if revision is not None:
            revision = whole(revision, "Expected source version", 1, 1000000)
        second = data.get("source_index_2")
        shift = data.get("applied_shift")
        if space == "paired-axis":
            second = whole(second if second is not None else 1, "Second observation position", 0, 3)
            shift = real(shift if shift is not None else 0.0, "Frozen registration shift", -26000, 26000)
        elif second is not None:
            second = whole(second, "Second observation position", 0, 3)
        elif shift is not None:
            shift = real(shift, "Registration shift", -26000, 26000)
        return cls(space, value, radius, chosen, revision, second, shift)


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
    pixel = reading["pixel"]
    label = f"Original {pixel[0]}, {pixel[1]}  /  {'supported' if reading['supported'] else 'no camera support'}"
    draw.text((9, 25), label, fill=CYAN if reading["supported"] else AMBER, font=font(11))
    return decoded


def _axis_witness(source: Source, axis: float, radius: int):
    trace = source.trace
    index = trace.interval(axis)
    if index is None:
        return {"state": "outside saved trace", "requested_axis": axis,
                "source": source.public()}, None
    image, reading = annotate_loupe(source.camera, trace, index, radius)
    decision = {**reading, "requested_axis": axis, "sampled_axis": rounded(trace.axis[index]),
                "state": trace.row(index)["state"], "strip": trace.config.public()}
    return decision, image


def paired_replay(sources, request: Request):
    """Resolve one corrected axis position to the two native positions actually compared.

    The second frame is looked up at ``axis + frozen shift``, i.e. the exact
    displacement applied when the chart was drawn; nothing is re-estimated.
    """
    first = sources[request.source_index]
    second = sources[request.source_index_2]
    axis = float(request.value)
    shift = float(request.applied_shift or 0.0)
    left, left_image = _axis_witness(first, axis, request.radius)
    right_axis = axis + shift
    right, right_image = _axis_witness(second, right_axis, request.radius)
    pair = {
        "state": "paired original positions",
        "requested_axis": axis,
        "applied_shift": rounded(shift),
        "first": {"title": first.title, "requested_axis": axis,
                  "sampled_axis": left.get("sampled_axis"), "state": left.get("state"),
                  "source_uv": left.get("uv"), "pixel": left.get("pixel")},
        "second": {"title": second.title, "requested_axis": rounded(right_axis),
                   "sampled_axis": right.get("sampled_axis"), "state": right.get("state"),
                   "source_uv": right.get("uv"), "pixel": right.get("pixel")},
    }
    if left_image is None or right_image is None:
        return pair, None
    height = max(left_image.height, right_image.height)
    gap = 10
    canvas = Image.new("RGB", (left_image.width + right_image.width + gap, height + 34), CANVAS)
    draw = ImageDraw.Draw(canvas)
    canvas.paste(left_image, (0, 34))
    canvas.paste(right_image, (left_image.width + gap, 34))
    draw.text((6, 6), f"{first.title[:30]}  axis {axis:g}", fill=CYAN, font=font(13))
    draw.text((left_image.width + gap + 6, 6),
              f"{second.title[:28]}  axis {right_axis:g} (shift {shift:g})", fill=AMBER, font=font(13))
    return pair, encode_png(canvas)


def replay(source: Source, request: Request):
    if request.expected_revision is not None and source.record["revision"] != request.expected_revision:
        raise Problem("The source changed after this chart was drawn. Run the reading again.", 409)
    if request.space == "axis":
        decision, image = _axis_witness(source, float(request.value), request.radius)
        if image is None:
            return {**decision, "source": source.public()}, None
        decorated = _attach_header(image, source.title, decision)
        return {**decision, "source": source.public()}, encode_png(decorated)
    uv = request.value
    reading = source.camera.pixel(uv)
    index = _near_trace(source.trace, uv)
    decision = {**reading, "state": "camera pixel" if reading["supported"] else "no camera support",
                "column_result": source.trace.row(index, exact=True) if index is not None else None}
    image = _supports_picture(source.camera, reading, request.radius)
    decorated = _attach_header(image, source.title, decision)
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
