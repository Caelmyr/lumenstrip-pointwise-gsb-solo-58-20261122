"""Studio-level provenance inspection and side-by-side saved measurements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape

from .archive import _cell, _style, _table
from .contracts import Problem, identity, rounded, seq


@dataclass(frozen=True)
class RecordSource:
    id: str
    name: str
    sha256: str
    fit_id: str | None
    fit_name: str | None
    revision: int
    configuration: dict

    @classmethod
    def read(cls, source):
        fit = source.get("reading_fit")
        return cls(source["id"], source["meta"]["title"], source["image"]["sha256"],
                   fit["id"] if fit else None, fit["calibration"]["title"] if fit else None,
                   source["revision"], source["strip"])

    def report(self):
        return {"source": self.id, "name": self.name, "camera_sha256": self.sha256,
                "fit_version": self.fit_id, "fit_name": self.fit_name,
                "source_revision": self.revision, "strip": self.configuration}


def _document_probe(library, key, snapshot):
    status = {"source_id": snapshot.id, **snapshot.report()}
    try:
        row = library.read_source(snapshot.id)
        status["present"] = True
    except Problem as error:
        status.update(present=False, storage_state=error.status, error=str(error))
        return status
    try:
        original = library.object(snapshot.sha256)
        status["retained_original_intact"] = len(original) > 0
    except Problem as error:
        status.update(retained_original_intact=False, storage_state=error.status, error=str(error))
    status["live_revision"] = row["revision"]
    status["live_image_same"] = row["image"]["sha256"] == snapshot.sha256
    status["live_fit_same"] = row["active_fit"] == snapshot.fit_id
    status["live_strip_same"] = row["strip"] == snapshot.configuration
    stamp = row["created"]
    status["observation_imported"] = stamp
    status["current_name"] = row["meta"]["title"]
    status["still_belongs_to_studio"] = row["station"] == key
    return status


def _asset_probe(library, record, row):
    evidence = {"name": row["name"], "hash": row["digest"], "declared_bytes": row["size"], "mime": row["mime"]}
    try:
        raw, actual = library.asset(record, row["name"])
        evidence.update(present=True, matches_bytes=len(raw) == row["size"],
                        matches_mime=actual == row["mime"])
    except Problem as error:
        evidence.update(present=False, error=str(error), status=error.status)
    return evidence


def record_health(library, record_key, include_originals):
    record = library.record(record_key)
    snapshots = [RecordSource.read(item) for item in record["sources"]]
    if include_originals:
        sources = [_document_probe(library, record["station"], source) for source in snapshots]
    else:
        sources = [source.report() for source in snapshots]
    assets = [_asset_probe(library, record_key, item) for item in record["assets"]]
    return {"record": record["id"], "name": record["name"], "tool": record["tool"], "sources": sources,
            "assets": assets, "originals_checked": include_originals,
            "healthy_binary_assets": sum(a["present"] and a.get("matches_bytes") is True and
                                         a.get("matches_mime") is True for a in assets),
            "snapshot_count": len(snapshots),
            "active_settings_differences": sum(r.get("live_fit_same") is False or r.get("live_strip_same") is False
                                               or r.get("live_image_same") is False for r in sources)}


def _comparable(value):
    return type(value) in (float, int)


def _metric_lines(records):
    common = set.intersection(*(set(r.get("metrics", {})) for r in records))
    lines = []
    for key in sorted(common):
        values = [record["metrics"][key] for record in records]
        if not all(type(x) in {bool, str, float, int} or x is None for x in values):
            continue
        row = {"metric": key, "all_equal": all(x == values[0] for x in values)}
        for i, v in enumerate(values):
            row[f"reading_{i + 1}"] = v
        if all(_comparable(value) for value in values):
            row["numerical_range"] = rounded(max(values) - min(values))
        lines.append(row)
    return lines


def _matches(records):
    names = []
    all_sources = [[RecordSource.read(s) for s in doc["sources"]] for doc in records]
    index = {(s.id, s.name) for choices in all_sources for s in choices}
    for source, name in sorted(index, key=lambda pair: pair[1]):
        history = {}
        for i, selected in enumerate(all_sources):
            found = next((row for row in selected if row.id == source and row.name == name), None)
            history[f"reading_{i + 1}"] = ({"saved_fit": found.fit_name,
                                          "camera_hash": found.sha256,
                                          "revision": found.revision,
                                          "center": found.configuration["endpoints"]} if found else None)
        names.append({"source": name, "observation_id": source, **history})
    return names


def list_readings(library, station, keys):
    items = [identity(v, "reading") for v in seq(keys, "Retained readings", 2, 4)]
    if len(set(items)) != len(items):
        raise Problem("Choose distinct retained readings.")
    records = [library.record(key) for key in items]
    if any(doc["station"] != station for doc in records):
        raise Problem("Choose retained readings from the same studio.")
    return records


def studio_folio(library, station, keys):
    records = list_readings(library, station, keys)
    profile = library.station(station)
    identity_rows = []
    for index, data in enumerate(records):
        sources = data["sources"]
        identity_rows.append({"order": index + 1, "reading": data["name"], "measurement": data["tool"],
                              "created": data["created"], "camera_count": len(sources),
                              "measurement_rows": data["row_count"], "source_revisions": [s["revision"] for s in sources]})
    tools = {r["tool"] for r in records}
    metrics = _metric_lines(records) if len(tools) == 1 else []
    return {"studio": profile, "readings": identity_rows, "sources": _matches(records),
            "matching_metrics": metrics, "same_tool": len(tools) == 1, "records": records}


def _caption(record):
    key = escape(record["id"], quote=True)
    name = escape(record["name"])
    cat = escape(record["tool"])
    return (f'<section><h3>{name}</h3><p>{cat}</p><p><a href="/api/records/{key}/sheet">Reading sheet</a> · '
            f'<a href="/api/records/{key}/download">Evidence ZIP</a></p></section>')


def folio_page(doc):
    notes = ("Common top-level measurements share a tool." if doc["same_tool"] else
             "Different tools were selected. Their metrics remain separate.")
    heading = escape(doc["studio"]["name"])
    records = "".join(_caption(record) for record in doc["records"])
    return ("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width'>"
            "<title>Studio records</title><style>" + _style() + "</style><main>"
            f"<p class=muted>LumenStrip studio readings</p><h1>{heading}</h1><p>{notes}</p>{records}"
            f"<h2>Retained reading details</h2>{_table(doc['readings'])}"
            f"<h2>Source versions</h2>{_table(doc['sources'])}"
            f"<h2>Common measurements</h2>{_table(doc['matching_metrics'])}</main></html>")


def camera_health(library, source):
    from .camera import Camera
    here = library.read_source(source)
    metadata = here["image"]
    facts = {"source": source, "observation": here["meta"]["title"], "stored_sha256": metadata["sha256"],
             "exposure_ms": here["meta"]["exposure_ms"], "stored_dimensions": [metadata["width"], metadata["height"]]}
    try:
        raw, desc = library.original(source)
        camera = Camera.read(raw, desc["name"])
        preview = library.preview_image(source)
    except Problem as error:
        return {**facts, "error": str(error), "status": error.status, "verified": False}
    public = camera.public()
    preview_camera = Camera.read(preview, "preview.png")
    fits = library.fits(source)
    rows = []
    from .calibration import Fit
    for row in fits:
        fit = Fit.parse(row["calibration"])
        description = fit.description(camera.size[0])
        rows.append({"id": row["id"], "title": description["title"], "active": row["id"] == here["active_fit"],
                     "wavelength_scope": description["mapped_extent_nm"],
                     "native_columns": [rounded(x * camera.size[0] - .5) for x in fit.columns]})
    return {**facts, "verified": all(metadata.get(key) == public[key] for key in
                                    ("sha256", "mime", "width", "height", "bytes")),
            "original_supported_fraction": public["supported_fraction"],
            "preview_dimensions": list(preview_camera.size),
            "preview_supported_fraction": preview_camera.public()["supported_fraction"],
            "strip_configuration": here["strip"], "source_revision": here["revision"], "known_rulers": rows}
