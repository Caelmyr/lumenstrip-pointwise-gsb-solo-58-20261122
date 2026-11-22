"""Result documents and named binary assets."""

from collections import Counter
import csv
from dataclasses import dataclass, field
from hashlib import sha256
from io import StringIO
import json

from .camera import encode_png
from .contracts import Problem, token
from .view import Plot


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def table_bytes(rows, header=None):
    target = StringIO()
    columns = header or list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.writer(target)
    if columns:
        writer.writerow(columns)
        for row in rows:
            writer.writerow(_cell(row.get(name)) for name in columns)
    return target.getvalue().encode("utf-8-sig")


@dataclass(frozen=True)
class Asset:
    name: str
    title: str
    mime: str
    data: bytes

    def describe(self):
        return {"name": self.name, "mime": self.mime, "title": self.title,
                "bytes": len(self.data), "sha256": sha256(self.data).hexdigest()}


@dataclass
class Result:
    identifier: str
    title: str
    summary: str
    metrics: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assets: list[Asset] = field(default_factory=list)
    probes: dict = field(default_factory=dict)

    def append(self, name, title, mime, data):
        token(name)
        if name in {asset.name for asset in self.assets}:
            raise Problem("A reading unexpectedly repeated an asset name.", 500)
        self.assets.append(Asset(name, title, mime, data))
        return name

    def image(self, name, title, image, probe=None):
        name = self.append(name + ".png", title, "image/png", encode_png(image))
        if probe:
            self.probes[name] = probe
        return name

    def chart(self, name, title, graph: Plot):
        return self.image(name, title, graph.image, {"kind": "axis", **graph.public()})

    def csv(self, name, title, rows, header=None):
        return self.append(name + ".csv", title, "text/csv; charset=utf-8", table_bytes(rows, header))

    def outline(self):
        missing_states = Counter(row.get("state") for row in self.rows if row.get("state"))
        return {"tool": self.identifier, "title": self.title, "summary": self.summary,
                "metrics": self.metrics, "warnings": self.warnings,
                "rows": self.rows[:160], "row_count": len(self.rows),
                "states": dict(missing_states), "probes": self.probes,
                "asset_index": [asset.describe() for asset in self.assets]}
