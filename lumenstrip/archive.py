"""Display, export and check actual evidence for retained camera readings."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import PurePosixPath
from typing import Iterable
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from .contracts import Problem, file_label, json_bytes

VERSION = "lumenstrip-reading-1"
MAX_IMPORT = 30_000_000
MAX_EXPANDED = 110_000_000
MAX_MEMBERS = 84


@dataclass(frozen=True)
class File:
    path: str
    mime: str
    data: bytes

    def fingerprint(self):
        return {"path": self.path, "mime": self.mime, "size": len(self.data),
                "sha256": sha256(self.data).hexdigest()}


def _cell(value):
    if value is None:
        return "—"
    if type(value) is bool:
        return "Yes" if value else "No"
    if isinstance(value, (dict, list)):
        return escape(json.dumps(value, ensure_ascii=False))
    return escape(str(value))


def _table(rows: list[dict], maximum: int = 42) -> str:
    if not rows:
        return "<p>No recorded rows.</p>"
    fields = list(dict.fromkeys(key for row in rows[:maximum] for key in row))[:12]
    header = "".join("<th>" + escape(name.replace("_", " ").title()) + "</th>" for name in fields)
    body = ["<tr>" + "".join("<td>" + _cell(row.get(key)) + "</td>" for key in fields) + "</tr>"
            for row in rows[:maximum]]
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def _facts(metrics: dict) -> str:
    useful = list(metrics.items())[:25]
    return "<dl>" + "".join("<div><dt>" + escape(key.replace("_", " ").title())
                           + "</dt><dd>" + _cell(value) + "</dd></div>" for key, value in useful) + "</dl>"


def _style():
    return (
        "body{background:#f8f6f0;color:#172c3d;font:16px system-ui;margin:0}"
        "main{max-width:1000px;margin:36px auto;padding:0 26px 30px}h1{font-size:2em}"
        "h2{margin:26px 0 9px;padding-bottom:8px;border-bottom:1px solid #cfd6d7}"
        "small,p.muted{color:#506878}dl{display:flex;gap:10px;flex-wrap:wrap}"
        "dl div{background:#edf2f0;padding:9px;max-width:420px}dt{font-size:.8em;color:#506878}dd{margin:3px 0}"
        "table{width:100%;border-collapse:collapse;font-size:12px;overflow-wrap:anywhere}"
        "td,th{border-bottom:1px solid #cfd6d7;padding:6px 3px;vertical-align:top;text-align:left}"
        "img{width:100%;height:auto;max-height:420px;object-fit:contain;background:#12212d}"
        "figure{margin:14px 0;padding:8px;background:#e6ece9}a{color:#214f63}"
        "figcaption{padding-top:5px;font-size:.9em}code{overflow-wrap:anywhere}"
        "@media print{main{margin:0;padding:0}figure{break-inside:avoid}table{font-size:8px}}"
    )


def sheet(record: dict, encoded_assets: dict[str, str] | None = None,
          archived_sources: dict[str, str] | None = None) -> str:
    idx = {entry["name"]: entry["title"] for entry in record.get("asset_index", [])}
    figures = []
    for asset in record.get("assets", []):
        if not asset["mime"].startswith("image/"):
            continue
        name = asset["name"]
        url = encoded_assets.get(name) if encoded_assets is not None else asset["url"]
        if not url:
            continue
        title = escape(idx.get(name, name))
        figures.append(f'<figure><img alt="{title}" src="{escape(url, quote=True)}">'
                       f"<figcaption>{title}</figcaption></figure>")
    source_notes = []
    for r in record.get("sources", []):
        label = escape(r["meta"]["title"])
        image = r["image"]
        hash_text = escape(image["sha256"])
        fit = r.get("reading_fit")
        saved = archived_sources.get(r["id"]) if archived_sources is not None else r["files"]["original"]
        url = f'<a href="{escape(saved, quote=True)}">Camera original</a>' if saved else "Camera original omitted"
        synthetic = " · Synthetic input" if r["meta"].get("synthetic") else ""
        calibration = escape(fit["calibration"]["title"]) if fit else "Native columns"
        gaps = len(fit["calibration"].get("excluded") or []) if fit else 0
        if gaps:
            calibration += f" · {gaps} unmeasured interval{'s' if gaps > 1 else ''}"
        source_notes.append(f"<li><strong>{label}</strong> · {calibration}{synthetic} · {url}"
                            f"<br><small>{escape(image['name'])} / {image['width']} × {image['height']}"
                            f" / {r['meta']['exposure_ms']} ms</small><br><code>{hash_text}</code></li>")
    notices = "".join("<li>" + escape(w) + "</li>" for w in record.get("warnings", []))
    notes = "".join(f"<li><time>{escape(x['created'])}</time> {escape(x['message'])}</li>"
                    for x in record.get("review_notes", []))
    ui_title = escape(record.get("name", record.get("title", "Camera reading")))
    return "".join([
        "<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width'>",
        "<title>", ui_title, "</title><style>", _style(), "</style><main><small>LumenStrip reading</small><h1>",
        ui_title, "</h1><p>", escape(record.get("summary", "")), "</p><p class=muted>",
        escape(record.get("created", "")), "</p>", _facts(record.get("metrics", {})),
        "<h2>Camera observations</h2><ol>", "".join(source_notes), "</ol>",
        "<h2>Recorded figures</h2>", "".join(figures),
        "<h2>Measured rows</h2>", _table(record.get("rows", [])),
        "<p class=muted>", str(record.get("row_count", 0)), " total rows in this reading.</p>",
        "<h2>Review notes</h2><ul>", notes, "</ul><h2>Reading notices</h2><ul>", notices,
        "</ul></main></html>",
    ])


def standalone_sheet(library, key):
    record = library.record(key)
    inline = {}
    for asset in record["assets"]:
        if not asset["mime"].startswith("image/"):
            continue
        data, mime = library.asset(key, asset["name"])
        inline[asset["name"]] = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return sheet(record, inline)


def _assets(library, record):
    result = []
    known = {}
    for asset in record["assets"]:
        raw, mime = library.asset(record["id"], asset["name"])
        name = "measurements/" + asset["name"]
        known[asset["name"]] = name
        result.append(File(name, mime, raw))
    return result, known


def _sources(library, record):
    result = []
    known = {}
    for i, source in enumerate(record["sources"], 1):
        name = f"originals/{i:02d}-{file_label(source['image']['name'])}"
        raw = library.object(source["image"]["sha256"])
        known[source["id"]] = name
        result.append(File(name, source["image"]["mime"], raw))
    return result, known


def _create_archive(files: Iterable[File]) -> bytes:
    target = BytesIO()
    with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=5) as zipfile:
        for file in files:
            zipfile.writestr(file.path, file.data)
    return target.getvalue()


def package(library, key, originals: bool):
    record = library.record(key)
    assets, asset_urls = _assets(library, record)
    source_files, source_urls = _sources(library, record) if originals else ([], {})
    content = {key: value for key, value in record.items() if key not in {"assets"}}
    content["asset_paths"] = asset_urls
    content["original_paths"] = source_urls
    browser = sheet(record, asset_urls, source_urls)
    members = [File("reading.json", "application/json", json_bytes(content, pretty=True)),
               File("reading.html", "text/html; charset=utf-8", browser.encode("utf-8")),
               *assets, *source_files]
    manifest = File("manifest.json", "application/json", json_bytes({
        "schema": VERSION, "reading": key, "includes_camera_originals": originals,
        "files": [file.fingerprint() for file in members],
    }, pretty=True))
    label = file_label(record["name"]) + ("-evidence.zip" if originals else "-measurements.zip")
    return label, _create_archive([manifest, *members])


def _safe_path(name):
    path = PurePosixPath(name)
    return isinstance(name, str) and not path.is_absolute() and all(p not in {".", "..", ""} for p in name.split("/")) and "\\" not in name and len(name) <= 160


def _headers(archive):
    total = 0
    distinct = set()
    for item in archive.infolist():
        name = item.filename
        if item.is_dir() or name in distinct or not _safe_path(name):
            raise Problem("This evidence packet contains an unexpected member name.")
        if item.flag_bits & 1:
            raise Problem("Encrypted members cannot be checked.")
        distinct.add(name)
        total += item.file_size
        if total > MAX_EXPANDED or len(distinct) > MAX_MEMBERS:
            raise Problem("Expanded evidence exceeds the reading size limit.", 413)
    if not {"manifest.json", "reading.json", "reading.html"} <= distinct:
        raise Problem("The archive does not contain a reading and its manifest.")
    return distinct


def _decode(archive, name):
    item = archive.getinfo(name)
    if item.file_size > 10_000_000:
        raise Problem("The reading metadata exceeds the supported size.", 413)
    try:
        return json.loads(archive.read(name))
    except (UnicodeError, ValueError) as err:
        raise Problem("Reading metadata is unreadable.") from err


def _member_digest(archive, name):
    witness = sha256()
    count = 0
    with archive.open(name, "r") as fh:
        while block := fh.read(128 * 1024):
            count += len(block)
            witness.update(block)
    return count, witness.hexdigest()


def packet_audit(raw: bytes):
    if not raw or len(raw) > MAX_IMPORT:
        raise Problem("Choose a reading packet smaller than 30 MB.", 413)
    try:
        with ZipFile(BytesIO(raw)) as packet:
            files = _headers(packet)
            manifest = _decode(packet, "manifest.json")
            record = _decode(packet, "reading.json")
            if not isinstance(manifest, dict) or manifest.get("schema") != VERSION or not isinstance(record, dict):
                raise Problem("This packet version is not a LumenStrip reading.")
            index = manifest.get("files")
            if not isinstance(index, list) or len(index) > MAX_MEMBERS:
                raise Problem("The packet manifest did not list reading members.")
            states = []
            checked = set()
            for member in index:
                if not isinstance(member, dict) or not isinstance(member.get("path"), str):
                    raise Problem("A member in the packet manifest lacks a filename.")
                name = member["path"]
                if name not in files or name == "manifest.json" or name in checked:
                    raise Problem("The manifest and actual ZIP members disagree.")
                checked.add(name)
                size, digest = _member_digest(packet, name)
                states.append({"path": name, "bytes": size, "sha256": digest,
                               "matches_index": digest == member.get("sha256") and size == member.get("size")})
            if files != checked | {"manifest.json"}:
                raise Problem("There are archive members outside the manifest.")
            consistency = _source_consistency(record, manifest, states)
            return {"reading": record.get("name", ""), "camera_references": record.get("original_paths", {}),
                    "member_count": len(states), "members": states, "sources": consistency,
                    "valid": all(s["matches_index"] for s in states) and all(r["agree"] for r in consistency)
                    and manifest.get("reading") == record.get("id")}
    except (BadZipFile, NotImplementedError, RuntimeError, OSError) as err:
        raise Problem("The archived reading is not a valid, readable ZIP packet.") from err


def _source_consistency(record, manifest, states):
    index = {member["path"]: member for member in states}
    sources = record.get("sources")
    paths = record.get("original_paths", {})
    wanted = manifest.get("includes_camera_originals") is True
    if not isinstance(sources, list) or not isinstance(paths, dict) or len(sources) > 4:
        raise Problem("The packet source index is malformed.")
    rows = []
    for s in sources:
        if not isinstance(s, dict) or not isinstance(s.get("image"), dict) or not isinstance(s.get("id"), str):
            raise Problem("The packet source index is malformed.")
        path = paths.get(s["id"])
        row = index.get(path) if isinstance(path, str) else None
        actual = row["sha256"] if row else None
        stored = s["image"].get("sha256")
        match = (actual == stored and actual is not None) if wanted else row is None
        rows.append({"source": s["id"], "included": row is not None, "expected_hash": stored,
                     "actual_hash": actual, "agree": match})
    return rows
