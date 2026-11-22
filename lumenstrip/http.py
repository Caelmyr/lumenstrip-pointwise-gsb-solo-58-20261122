"""Local HTTP entry points for normal optical readings and retained evidence."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
from threading import Lock

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from .archive import MAX_IMPORT, package, packet_audit, standalone_sheet
from .bench import Bench, Capacity, Order, Source
from .calibration import Fit
from .camera import MAX_BYTES, Camera, encode_png
from .contracts import Capture, Problem, Strip, allowed, flag, identity, mapping, plain_text, point, signature, whole
from .fixture import Pattern
from .fitfile import MAX_CSV, export_fit, import_fit
from .library import Library
from .proofs import camera_health, folio_page, record_health, studio_folio
from .strip import Trace
from .tools import TOOLS
from .view import plot_anchor_view, strip_overlay
from .witness import Request, bookmark_replay, paired_replay, replay, saved_source


def json_body(label):
    return mapping(request.get_json(), label)


def body_fields(label, names):
    data = json_body(label)
    allowed(data, set(names), label)
    return data


def bounded_body(field, limit):
    src = request.files.get(field)
    if src is None:
        raise Problem(f"Choose a file for {field}.")
    raw = src.stream.read(limit + 1)
    if len(raw) > limit:
        raise Problem(f"{field} exceeds the upload byte limit.", 413)
    return raw, src.filename or "input"


def nested_form(key):
    try:
        return json.loads(request.form.get(key, "{}"))
    except (ValueError, TypeError) as exc:
        raise Problem(f"Invalid {key} metadata.") from exc


def attachment(data, mime, filename):
    return send_file(BytesIO(data), mimetype=mime, download_name=filename, as_attachment=True,
                     max_age=0, conditional=False)


def inline(data, mime):
    return Response(data, mimetype=mime, headers={"Cache-Control": "no-store"})


def base_image(data):
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def live_source(lib: Library, key, overrides=None):
    saved = lib.read_source(key)
    stations, rows = lib.acquire(saved["station"], [key], overrides or {}, {})
    del stations
    return Source.from_stored(lib, *rows[0])


def _normalize_station(data):
    return (plain_text(data.get("name", ""), "Studio name", 85),
            plain_text(data.get("notes", ""), "Notes", 360, required=False))


def stations(app, lab, demo_lock):
    @app.get("/api/stations")
    def list_stations():
        return {"stations": lab.stations()}

    @app.post("/api/stations")
    def new_station():
        doc = body_fields("Studio", ("name", "notes", "synthetic"))
        name, notes = _normalize_station(doc)
        synthetic = flag(doc.get("synthetic", False), "Demonstration flag")
        return {"station": lab.create(name, notes, synthetic)}, 201

    @app.get("/api/stations/<key>")
    def station(key):
        return {"station": lab.station(key), "sources": lab.captures(key), "records": lab.records(key)}

    @app.patch("/api/stations/<key>")
    def change_station(key):
        doc = body_fields("Studio edit", ("name", "notes", "revision"))
        name, notes = _normalize_station(doc)
        return {"station": lab.edit_station(key, name, notes, whole(doc.get("revision"), "Revision", 1, 1000000))}

    @app.get("/api/stations/<key>/activity")
    def events(key):
        return {"activity": lab.events(key)}

    @app.post("/api/demo")
    def demo_station():
        from .demo import seed
        with demo_lock:
            data = seed(lab)
        return data, 201


def captures(app, lab, capacity):
    @app.post("/api/stations/<key>/sources")
    def add_source(key):
        raw, filename = bounded_body("image", MAX_BYTES)
        meta = Capture.parse(nested_form("capture"))
        with capacity:
            return {"source": lab.ingest(key, raw, filename, meta)}, 201

    @app.post("/api/stations/<key>/synthetic")
    def synthetic_capture(key):
        data = body_fields("Synthetic observation", ("pattern", "retain"))
        pattern = Pattern.parse(data.get("pattern"))
        persist = flag(data.get("retain", False), "Import generated image")
        with capacity:
            lab.station(key)
            image = pattern.render()
            source = None
            if persist:
                built = lab.ingest(key, image, "synthetic-detector.png", pattern.capture())
                source = lab.set_strip(built["id"], built["revision"], pattern.scale())
            return {"synthetic": pattern.description(), "preview": base_image(image),
                    "source": source}, 201 if persist else 200

    @app.get("/api/sources/<key>")
    def camera_detail(key):
        return {"source": lab.read_source(key), "calibrations": lab.fits(key), "bookmarks": lab.bookmarks(key)}

    @app.get("/api/sources/<key>/health")
    def source_health(key):
        with capacity:
            health = camera_health(lab, key)
        return {"health": health}

    @app.get("/api/sources/<key>/original")
    def camera_original(key):
        picture, info = lab.original(key)
        return attachment(picture, info["mime"], info["name"])

    @app.get("/api/sources/<key>/preview")
    def camera_thumbnail(key):
        return inline(lab.preview_image(key), "image/png")

    @app.post("/api/sources/<key>/capture")
    def change_capture(key):
        doc = body_fields("Capture metadata", ("revision", "capture"))
        rev = whole(doc.get("revision"), "Revision", 1, 1000000)
        payload = Capture.parse(doc.get("capture"))
        return {"source": lab.set_meta(key, rev, payload)}

    @app.post("/api/sources/<key>/strip")
    def update_strip(key):
        doc = body_fields("Strip preview", ("revision", "strip", "retain"))
        rev = whole(doc.get("revision"), "Revision", 1, 1000000)
        settings = Strip.parse(doc.get("strip"))
        persist = flag(doc.get("retain", False), "Retain strip")
        with capacity:
            source = live_source(lab, key)
            if source.record["revision"] != rev:
                raise Problem("Reload the camera record before changing its strip.", 409)
            trace = Trace.read(source.camera, settings, source.fit)
            image = strip_overlay(source.camera, trace)
            saved = lab.set_strip(key, rev, settings) if persist else None
            return {"source": saved, "configuration": settings.public(), "summary": trace.summary(),
                    "overlay": base_image(encode_png(image))}

    @app.post("/api/sources/<key>/calibration")
    def fit_axis(key):
        doc = body_fields("Calibration preview", ("revision", "calibration", "retain", "expected"))
        rev = whole(doc.get("revision"), "Revision", 1, 1000000)
        fit = Fit.parse(doc.get("calibration"))
        persist = flag(doc.get("retain", False), "Retain calibration")
        with capacity:
            source = live_source(lab, key)
            if source.record["revision"] != rev:
                raise Problem("Reload the camera record before changing the wavelength ruler.", 409)
            fingerprint = signature({"image": source.camera.sha256, "revision": rev, "fit": fit.public()})
            trace = Trace.read(source.camera, source.strip, fit)
            geometry = plot_anchor_view(source.camera, trace, fit.anchors)
            retained = lab.add_fit(key, rev, fit, doc.get("expected")) if persist else None
            return {"preview_key": fingerprint, "fit": fit.description(source.camera.size[0]),
                    "settings": fit.public(), "result": retained, "overlay": base_image(encode_png(geometry))}

    @app.get("/api/sources/<key>/calibration.csv")
    def download_axis(key):
        source = live_source(lab, key)
        if source.fit is None:
            raise Problem("Select a saved wavelength ruler before downloading it.", 409)
        raw = export_fit(source.fit, source.camera.size[0], request.args.get("portable", "1") == "1")
        return attachment(raw, "text/csv; charset=utf-8", "wavelength-ruler.csv")

    @app.post("/api/sources/<key>/calibration/import")
    def import_axis(key):
        raw, name = bounded_body("anchors", MAX_CSV)
        meta = nested_form("metadata")
        with capacity:
            source = live_source(lab, key)
            width = source.camera.size[0]
            fit = import_fit(raw, width, meta)
            trace = Trace.read(source.camera, source.strip, fit)
            overlay = plot_anchor_view(source.camera, trace, fit.anchors)
            preview = signature({"image": source.camera.sha256, "revision": source.record["revision"],
                                 "fit": fit.public()})
            return {"preview_key": preview, "fit": fit.description(width), "settings": fit.public(),
                    "filename": name, "overlay": base_image(encode_png(overlay))}
    
    @app.post("/api/sources/<key>/calibrations/select")
    def select_fit(key):
        doc = body_fields("Calibration selection", ("revision", "fit_id"))
        rev = whole(doc.get("revision"), "Revision", 1, 1000000)
        chosen = identity(doc["fit_id"], "calibration") if doc.get("fit_id") else None
        return {"source": lab.use_fit(key, rev, chosen)}

    @app.post("/api/sources/<key>/bookmarks")
    def mark_source(key):
        doc = body_fields("Source note", ("uv", "name", "message", "revision"))
        value = {"uv": list(point(doc.get("uv"), "Camera location")),
                 "title": plain_text(doc.get("name", ""), "Bookmark", 75),
                 "message": plain_text(doc.get("message", ""), "Notes", 270, required=False)}
        revision = whole(doc.get("revision"), "Revision", 1, 1000000)
        return {"bookmark": lab.add_bookmark(key, value, revision)}, 201

    @app.get("/api/sources/<key>/bookmarks/<mark>")
    def inspect_bookmark(key, mark):
        with capacity:
            radius = 12
            bookmark = lab.bookmark(key, mark)
            result, image = bookmark_replay(lab, bookmark, radius)
        return {"reading": result, "witness": base_image(image) if image is not None else None}

    @app.post("/api/sources/<key>/probe")
    def probe_camera(key):
        selected = Request.parse(json_body("Camera witness"))
        with capacity:
            if selected.space == "paired-axis":
                raise Problem("Paired registration witnesses run from a two-source reading.")
            result, image = replay(live_source(lab, key), selected)
        return {"reading": result, "witness": base_image(image) if image is not None else None}


def _render_tool(lab, station, tool, order):
    bench = Bench.load(lab, station, order)
    bench.require(tool.minimum, tool.maximum)
    fingerprint = bench.verified_key(tool.name)
    output = tool.run(bench)
    output.warnings[:0] = bench.warnings()
    data = {**output.outline(), "order": order.public(), "sources": bench.frozen_sources(),
            "fingerprint": fingerprint}
    if order.retain:
        return lab.retain(station, tool.name, order.name, data, output.assets, fingerprint)
    items = [{**asset.describe(), "url": "data:" + asset.mime.split(";")[0] + ";base64,"
              + base64.b64encode(asset.data).decode("ascii")} for asset in output.assets]
    return {**data, "assets": items, "station": station}


def reading(app, lab, capacity):
    @app.get("/api/tools")
    def tools():
        return {"tools": [tool.public() for tool in TOOLS.values()]}

    @app.post("/api/stations/<key>/tools/<name>")
    def run_tool(key, name):
        tool = TOOLS.get(name)
        if tool is None:
            raise Problem("Unknown inspection tool.", 404)
        order = Order.parse(json_body("Reading"))
        with capacity:
            result = _render_tool(lab, key, tool, order)
        return result, 201 if order.retain else 200

    @app.post("/api/stations/<key>/tools/<name>/probe")
    def probe_reading(key, name):
        from .bench import Bench
        del name
        body = json_body("Reading witness")
        selected = Request.parse(body.get("witness", body))
        order = Order.parse(mapping(body.get("order", body), "Reading order"))
        with capacity:
            bench = Bench.load(lab, key, order)
            if selected.space == "paired-axis":
                result, image = paired_replay(bench.sources, selected)
            else:
                result, image = replay(bench.sources[selected.source_index], selected)
        return {"reading": result, "witness": base_image(image) if image is not None else None}

    @app.get("/api/records/<key>")
    def saved_record(key):
        return lab.record(key)

    @app.get("/api/records/<key>/assets/<filename>")
    def saved_asset(key, filename):
        data, mime = lab.asset(key, filename)
        return inline(data, mime)

    @app.get("/api/records/<key>/sheet")
    def record_sheet(key):
        with capacity:
            html = standalone_sheet(lab, key)
        return inline(html, "text/html; charset=utf-8")

    @app.get("/api/records/<key>/download")
    def record_download(key):
        originals = request.args.get("originals", "1") == "1"
        with capacity:
            name, data = package(lab, key, originals)
        return attachment(data, "application/zip", name)

    @app.post("/api/records/<key>/review")
    def review_record(key):
        data = body_fields("Review note", ("message",))
        text = plain_text(data.get("message", ""), "Review note", 380)
        return {"note": lab.add_review(key, text)}, 201

    @app.post("/api/records/<key>/probe")
    def saved_probe(key):
        selected = Request.parse(json_body("Saved source witness"))
        record = lab.record(key)
        with capacity:
            if selected.space == "paired-axis":
                sources = tuple(saved_source(lab, snapshot) for snapshot in record["sources"])
                result, image = paired_replay(sources, selected)
                return {"reading": result, "witness": base_image(image) if image is not None else None}
            try:
                source = record["sources"][selected.source_index]
            except IndexError:
                raise Problem("The saved reading has no observation in that position.", 404) from None
            result, image = replay(saved_source(lab, source), selected)
        return {"reading": result, "witness": base_image(image) if image is not None else None}

    @app.get("/api/records/<key>/health")
    def retained_health(key):
        with capacity:
            value = record_health(lab, key, request.args.get("originals") == "1")
        return {"health": value}

    @app.get("/api/stations/<key>/folio")
    def folio(key):
        records = request.args.getlist("reading")
        with capacity:
            doc = studio_folio(lab, key, records)
        return inline(folio_page(doc), "text/html; charset=utf-8")

    @app.post("/api/packet-audit")
    def audit():
        raw, filename = bounded_body("packet", MAX_IMPORT)
        del filename
        with capacity:
            result = packet_audit(raw)
        return {"audit": result}


def create_app(data_dir=None):
    static = Path(__file__).resolve().parent.parent / "static"
    app = Flask(__name__, static_folder=static, static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_IMPORT + 120_000
    app.config["JSON_SORT_KEYS"] = False
    lib = Library(data_dir or os.environ.get("LUMENSTRIP_DATA", "runtime-data"))
    capacity = Capacity()
    demo_lock = Lock()
    stations(app, lib, demo_lock)
    captures(app, lib, capacity)
    reading(app, lib, capacity)

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    @app.errorhandler(Problem)
    def known(error):
        return jsonify({"error": str(error)}), error.status

    @app.errorhandler(BadRequest)
    @app.errorhandler(RequestEntityTooLarge)
    def bad(error):
        return jsonify({"error": "Request is too large." if error.code == 413 else "Invalid request body."}), error.code

    @app.get("/api/health")
    def health():
        return {"app": "LumenStrip", "status": "ready", "inspection_tools": len(TOOLS)}

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    return app
