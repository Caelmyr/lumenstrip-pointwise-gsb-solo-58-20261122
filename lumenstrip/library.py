"""Store immutable camera bytes, versioned geometry, reports and audit events."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock

from .calibration import Fit
from .camera import Camera, encode_png
from .contracts import Capture, Problem, Strip, identity, json_bytes, signature, stamp, token, uid

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, notes TEXT NOT NULL,
 synthetic INTEGER NOT NULL, created TEXT NOT NULL, revision INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
 id TEXT PRIMARY KEY, station TEXT NOT NULL REFERENCES stations(id),
 meta TEXT NOT NULL, image TEXT NOT NULL, blob TEXT NOT NULL,
 created TEXT NOT NULL, revision INTEGER NOT NULL, strip TEXT NOT NULL, active_fit TEXT
);
CREATE INDEX IF NOT EXISTS source_station ON sources(station,created,id);
CREATE TABLE IF NOT EXISTS fits (
 id TEXT PRIMARY KEY, source TEXT NOT NULL REFERENCES sources(id),
 calibration TEXT NOT NULL, created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS fit_source ON fits(source,created,id);
CREATE TABLE IF NOT EXISTS records (
 id TEXT PRIMARY KEY, station TEXT NOT NULL REFERENCES stations(id),
 name TEXT NOT NULL, tool TEXT NOT NULL, document TEXT NOT NULL, created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS record_station ON records(station,created,id);
CREATE TABLE IF NOT EXISTS assets (
 record TEXT NOT NULL REFERENCES records(id), name TEXT NOT NULL, mime TEXT NOT NULL,
 digest TEXT NOT NULL, size INTEGER NOT NULL, PRIMARY KEY (record,name)
);
CREATE TABLE IF NOT EXISTS notes (
 id TEXT PRIMARY KEY, record TEXT NOT NULL REFERENCES records(id),
 message TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bookmarks (
 id TEXT PRIMARY KEY, source TEXT NOT NULL REFERENCES sources(id), document TEXT NOT NULL,
 created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS bookmark_source ON bookmarks(source,created,id);
CREATE TABLE IF NOT EXISTS eventlog (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, station TEXT NOT NULL REFERENCES stations(id),
 type TEXT NOT NULL, item TEXT NOT NULL, detail TEXT NOT NULL, created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS event_station ON eventlog(station,seq);
"""

_HEX = frozenset("abcdef0123456789")


def serialized(value) -> str:
    return json_bytes(value).decode("utf-8")


class Library:
    def __init__(self, directory):
        self.root = Path(directory).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "lumen.sqlite3"
        self.objects = self.root / "objects"
        self.objects.mkdir(exist_ok=True)
        self.preview_dir = self.root / "previews"
        self.preview_dir.mkdir(exist_ok=True)
        self._write_lock = RLock()
        with self.connection(write=True) as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connection(self, write: bool = False):
        with self._write_lock if write else _no_lock():
            db = sqlite3.connect(self.database, timeout=15)
            db.row_factory = sqlite3.Row
            try:
                db.execute("PRAGMA foreign_keys = ON")
                if write:
                    db.execute("BEGIN IMMEDIATE")
                yield db
                if write:
                    db.commit()
            except BaseException:
                if write:
                    db.rollback()
                raise
            finally:
                db.close()

    def _object(self, digest: str) -> Path:
        if len(digest) != 64 or set(digest) - _HEX:
            raise Problem("An object hash is invalid.", 500)
        return self.objects / digest[:2] / digest

    def _put(self, data: bytes) -> str:
        key = sha256(data).hexdigest()
        path = self._object(key)
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            return key
        staged = path.with_name(f".{key}.{uid()}")
        try:
            staged.write_bytes(data)
            os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)
        return key

    def object(self, digest: str, *, check: bool = True) -> bytes:
        try:
            raw = self._object(digest).read_bytes()
        except FileNotFoundError:
            raise Problem("The saved image or asset is missing.", 410) from None
        if check and sha256(raw).hexdigest() != digest:
            raise Problem("The saved image or asset has changed.", 409)
        return raw

    @staticmethod
    def _fetch(db, query, params, missing):
        row = db.execute(query, params).fetchone()
        if row is None:
            raise Problem(missing, 404)
        return row

    @classmethod
    def _station(cls, db, key):
        return cls._fetch(db, "SELECT * FROM stations WHERE id=?", (identity(key, "studio"),),
                          "Studio not found.")

    @classmethod
    def _source(cls, db, key):
        return cls._fetch(db, "SELECT * FROM sources WHERE id=?", (identity(key, "observation"),),
                          "Observation not found.")

    @classmethod
    def _record(cls, db, key):
        return cls._fetch(db, "SELECT * FROM records WHERE id=?", (identity(key, "record"),),
                          "Retained result not found.")

    @staticmethod
    def _event(db, station, event, item, detail):
        db.execute("INSERT INTO eventlog(station,type,item,detail,created) VALUES(?,?,?,?,?)",
                   (station, event, item, serialized(detail), stamp()))

    def events(self, key, limit=120) -> list[dict]:
        with self.connection() as db:
            station = self._station(db, key)
            rows = db.execute("SELECT * FROM eventlog WHERE station=? ORDER BY seq DESC LIMIT ?",
                              (station["id"], limit)).fetchall()
        return [{**dict(row), "detail": json.loads(row["detail"])} for row in rows]

    def create(self, name, note, synthetic=False) -> dict:
        key = uid()
        with self.connection(write=True) as db:
            db.execute("INSERT INTO stations VALUES(?,?,?,?,?,1)", (key, name, note, int(synthetic), stamp()))
            self._event(db, key, "station opened", key, {"name": name})
            return self._public_station(db, self._station(db, key))

    def edit_station(self, key, name, note, rev):
        with self.connection(write=True) as db:
            current = self._station(db, key)
            if rev != current["revision"]:
                raise Problem("This studio was updated in another tab. Reload it before editing.", 409)
            db.execute("UPDATE stations SET name=?,notes=?,revision=revision+1 WHERE id=?", (name, note, key))
            self._event(db, key, "station edited", key, {"name": name})
            return self._public_station(db, self._station(db, key))

    @staticmethod
    def _public_station(db, row) -> dict:
        key = row["id"]
        images = db.execute("SELECT COUNT(*) FROM sources WHERE station=?", (key,)).fetchone()[0]
        reports = db.execute("SELECT COUNT(*) FROM records WHERE station=?", (key,)).fetchone()[0]
        return {**dict(row), "synthetic": bool(row["synthetic"]), "images": images, "reports": reports}

    def stations(self):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM stations ORDER BY created DESC,id").fetchall()
            return [self._public_station(db, row) for row in rows]

    def station(self, key):
        with self.connection() as db:
            return self._public_station(db, self._station(db, key))

    @staticmethod
    def _public_source(row):
        record = dict(row)
        del record["blob"]
        for name in ("meta", "image", "strip"):
            record[name] = json.loads(record[name])
        key = row["id"]
        record["files"] = {"original": f"/api/sources/{key}/original", "preview": f"/api/sources/{key}/preview"}
        return record

    def captures(self, key):
        with self.connection() as db:
            station = self._station(db, key)
            rows = db.execute("SELECT * FROM sources WHERE station=? ORDER BY created,id", (station["id"],))
            return [self._public_source(row) for row in rows]

    def read_source(self, key):
        with self.connection() as db:
            return self._public_source(self._source(db, key))

    def ingest(self, key, raw: bytes, name: str, meta: Capture):
        camera = Camera.read(raw, name)
        digest = self._put(raw)
        key = identity(key, "studio")
        sid = uid()
        preview = self.preview_dir / f"{digest}.png"
        if not preview.is_file():
            staged = preview.with_name(f".{preview.name}.{uid()}")
            try:
                staged.write_bytes(encode_png(camera.image()))
                os.replace(staged, preview)
            finally:
                staged.unlink(missing_ok=True)
        with self.connection(write=True) as db:
            self._station(db, key)
            if db.execute("SELECT COUNT(*) FROM sources WHERE station=?", (key,)).fetchone()[0] >= 80:
                raise Problem("This studio already holds 80 images. Start another studio.", 413)
            db.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,1,?,NULL)",
                       (sid, key, serialized(meta.public()), serialized(camera.public()), digest,
                        stamp(), serialized(Strip.default().public())))
            self._event(db, key, "image imported", sid, {"sha256": digest, "role": meta.role,
                                                       "title": meta.title})
            return self._public_source(self._source(db, sid))

    @staticmethod
    def _matching_revision(row, rev):
        if row["revision"] != rev:
            raise Problem("The observation was changed in another tab. Reload it and check the current version.", 409)

    def set_meta(self, key, rev, meta):
        with self.connection(write=True) as db:
            row = self._source(db, key)
            self._matching_revision(row, rev)
            db.execute("UPDATE sources SET meta=?,revision=revision+1 WHERE id=?",
                       (serialized(meta.public()), key))
            self._event(db, row["station"], "capture description changed", key, {"role": meta.role})
            return self._public_source(self._source(db, key))

    def set_strip(self, key, rev, strip: Strip):
        with self.connection(write=True) as db:
            row = self._source(db, key)
            self._matching_revision(row, rev)
            db.execute("UPDATE sources SET strip=?,revision=revision+1 WHERE id=?",
                       (serialized(strip.public()), key))
            self._event(db, row["station"], "strip retained", key,
                        {"endpoints": [strip.first, strip.last], "width": 2 * strip.half_height + 1})
            return self._public_source(self._source(db, key))

    @staticmethod
    def _public_fit(row):
        return {**dict(row), "calibration": json.loads(row["calibration"])}

    @classmethod
    def _fit(cls, db, key, source):
        if key is None:
            return None
        return cls._public_fit(cls._fetch(db, "SELECT * FROM fits WHERE id=? AND source=?",
                                          (identity(key, "calibration"), source),
                                          "Calibration was not found for this source."))

    def fits(self, key):
        with self.connection() as db:
            row = self._source(db, key)
            found = db.execute("SELECT * FROM fits WHERE source=? ORDER BY created DESC,id", (row["id"],))
            return [self._public_fit(one) for one in found]

    def add_fit(self, key, rev, fit: Fit, preview_key):
        with self.connection(write=True) as db:
            row = self._source(db, key)
            self._matching_revision(row, rev)
            expected = signature({"image": row["blob"], "revision": rev, "fit": fit.public()})
            if expected != preview_key:
                raise Problem("Preview this exact calibration before saving it.", 409)
            if db.execute("SELECT COUNT(*) FROM fits WHERE source=?", (key,)).fetchone()[0] >= 48:
                raise Problem("This image has reached its saved calibration limit.", 413)
            fit_id = uid()
            db.execute("INSERT INTO fits VALUES(?,?,?,?)", (fit_id, key, serialized(fit.public()), stamp()))
            db.execute("UPDATE sources SET active_fit=?,revision=revision+1 WHERE id=?", (fit_id, key))
            self._event(db, row["station"], "calibration saved", key,
                        {"calibration": fit_id, "method": fit.method, "title": fit.title})
            return self._fit(db, fit_id, key)

    def use_fit(self, key, rev, selected):
        with self.connection(write=True) as db:
            row = self._source(db, key)
            self._matching_revision(row, rev)
            if selected is not None:
                self._fit(db, selected, key)
            db.execute("UPDATE sources SET active_fit=?,revision=revision+1 WHERE id=?", (selected, key))
            self._event(db, row["station"], "active calibration changed", key, {"calibration": selected})
            return self._public_source(self._source(db, key))

    def acquire(self, key, source_ids, fit_overrides, strip_overrides):
        with self.connection() as db:
            station = self._station(db, key)
            items = []
            for source in source_ids:
                row = self._source(db, source)
                if row["station"] != station["id"]:
                    raise Problem("Choose images in the same studio.", 422)
                fit = self._fit(db, fit_overrides.get(source, row["active_fit"]), row["id"])
                view = self._public_source(row)
                view["strip"] = strip_overrides.get(source, view["strip"])
                items.append((view, fit, row["blob"]))
            return self._public_station(db, station), items

    def preview_image(self, key):
        with self.connection() as db:
            row = self._source(db, key)
            target = self.preview_dir / f'{row["blob"]}.png'
        try:
            return target.read_bytes()
        except FileNotFoundError:
            raise Problem("Preview is missing.", 410) from None

    def original(self, key):
        with self.connection() as db:
            row = self._source(db, key)
            digest = row["blob"]
            image = json.loads(row["image"])
        return self.object(digest), image

    def retain(self, station, tool, name, document, files, expected: str):
        key = uid()
        prepared = [(key, asset.name, asset.mime, self._put(asset.data), len(asset.data)) for asset in files]
        with self.connection(write=True) as db:
            self._station(db, station)
            for source in document["sources"]:
                row = self._source(db, source["id"])
                if row["revision"] != source["revision"] or source["image"]["sha256"] != row["blob"]:
                    raise Problem("The source changed while computing the result. Reopen the preview.", 409)
            if db.execute("SELECT COUNT(*) FROM records WHERE station=?", (station,)).fetchone()[0] >= 200:
                raise Problem("This studio already contains 200 retained readings.", 413)
            full = {**document, "result_fingerprint": expected}
            db.execute("INSERT INTO records VALUES(?,?,?,?,?,?)",
                       (key, station, name, tool, serialized(full), stamp()))
            db.executemany("INSERT INTO assets VALUES(?,?,?,?,?)", prepared)
            self._event(db, station, "reading retained", key, {"tool": tool, "name": name})
            return self._public_record(db, self._record(db, key))

    @staticmethod
    def _asset_rows(db, record_id):
        rows = db.execute("SELECT * FROM assets WHERE record=? ORDER BY rowid", (record_id,))
        return [{**dict(row), "url": f"/api/records/{record_id}/assets/{row['name']}"} for row in rows]

    def _public_record(self, db, row):
        doc = json.loads(row["document"])
        notes = db.execute("SELECT * FROM notes WHERE record=? ORDER BY created,id", (row["id"],))
        return {**doc, "id": row["id"], "name": row["name"], "created": row["created"],
                "station": row["station"], "assets": self._asset_rows(db, row["id"]),
                "review_notes": [dict(note) for note in notes]}

    def records(self, key):
        with self.connection() as db:
            station = self._station(db, key)
            rows = db.execute("SELECT id,name,tool,created FROM records WHERE station=? ORDER BY created DESC,id",
                              (station["id"],))
            return [dict(row) for row in rows]

    def record(self, key):
        with self.connection() as db:
            return self._public_record(db, self._record(db, key))

    def asset(self, key, name):
        with self.connection() as db:
            record = self._record(db, key)
            row = self._fetch(db, "SELECT digest,size,mime FROM assets WHERE record=? AND name=?",
                              (record["id"], token(name)), "Reading asset not found.")
        raw = self.object(row["digest"])
        if len(raw) != row["size"]:
            raise Problem("Saved asset byte count disagrees with its index.", 409)
        return raw, row["mime"]

    def add_review(self, key, message):
        with self.connection(write=True) as db:
            record = self._record(db, key)
            n = db.execute("SELECT COUNT(*) FROM notes WHERE record=?", (key,)).fetchone()[0]
            if n >= 40:
                raise Problem("This retained reading has reached its review note limit.", 413)
            note = {"id": uid(), "record": key, "message": message, "created": stamp()}
            db.execute("INSERT INTO notes VALUES(:id,:record,:message,:created)", note)
            self._event(db, record["station"], "reading noted", key, {"note": note["id"]})
            return note

    def add_bookmark(self, source, data, revision):
        with self.connection(write=True) as db:
            row = self._source(db, source)
            self._matching_revision(row, revision)
            n = db.execute("SELECT COUNT(*) FROM bookmarks WHERE source=?", (source,)).fetchone()[0]
            if n >= 64:
                raise Problem("This source has too many witness bookmarks.", 413)
            view = self._public_source(row)
            fit = self._fit(db, row["active_fit"], source)
            snapshot = {**view, "reading_fit": fit}
            payload = {**data, "image_sha256": row["blob"], "revision": row["revision"],
                       "fit_id": row["active_fit"], "source_snapshot": snapshot}
            key = uid()
            created = stamp()
            db.execute("INSERT INTO bookmarks VALUES(?,?,?,?)", (key, source, serialized(payload), created))
            self._event(db, row["station"], "source witness marked", source, {"bookmark": key})
            return {"id": key, "source": source, "created": created, **payload}

    def bookmarks(self, source):
        with self.connection() as db:
            row = self._source(db, source)
            found = db.execute("SELECT * FROM bookmarks WHERE source=? ORDER BY created DESC,id", (row["id"],))
            rows = []
            for mark in found:
                value = json.loads(mark["document"])
                value.pop("source_snapshot", None)
                public = {**dict(mark), **value}
                del public["document"]
                rows.append(public)
            return rows

    def bookmark(self, source, key):
        with self.connection() as db:
            observation = self._source(db, source)
            found = self._fetch(db, "SELECT * FROM bookmarks WHERE id=? AND source=?",
                                (identity(key, "bookmark"), observation["id"]), "Source bookmark was not found.")
            return {**json.loads(found["document"]), "id": found["id"],
                    "source": found["source"], "created": found["created"]}


@contextmanager
def _no_lock():
    yield
