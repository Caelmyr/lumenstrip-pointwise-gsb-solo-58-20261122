"""Exchange wavelength rulers with native units, explicit column bounds and provenance."""

from __future__ import annotations

import csv
from io import StringIO

import numpy as np

from .calibration import Fit
from .contracts import Problem, allowed, choice, mapping, plain_text, real, rounded, whole
from .result import table_bytes

MAX_CSV = 160_000
NUMBER_COLUMNS = ("nm", "column", "native_px", "weight")
DATA_COLUMNS = ("name", "nm", "column", "weight", "role")
ROLE_FIT, ROLE_DISABLED = "anchor", "disabled"
PORTABLE = ("name", "nm", "column", "weight", "native_px", "role")


def _decode(raw):
    if not raw or len(raw) > MAX_CSV:
        raise Problem("Choose a wavelength anchor CSV smaller than 160 KB.", 413)
    try:
        data = raw.decode("utf-8-sig")
    except UnicodeDecodeError as err:
        raise Problem("The wavelength table must use UTF-8.") from err
    if "\x00" in data:
        raise Problem("The wavelength table contains binary data.")
    return data


def _columns(header):
    columns = tuple(str(name).strip().lower() for name in header)
    if len(set(columns)) != len(columns):
        raise Problem("Wavelength table columns must be unique.")
    if "nm" not in columns or not ({"column", "native_px"} & set(columns)):
        raise Problem("Use an nm column and either column or native_px.")
    unknown = set(columns) - set(PORTABLE)
    if unknown:
        raise Problem("Unsupported wavelength table column: " + sorted(unknown)[0] + ".")
    return columns


def _number(text, title, low, high):
    try:
        value = float(text)
    except (ValueError, TypeError) as err:
        raise Problem(f"{title} must contain a number.") from err
    return real(value, title, low, high)


def _column(row, width, row_number):
    xval = row.get("column", "").strip()
    px = row.get("native_px", "").strip()
    if not xval and not px:
        raise Problem(f"Row {row_number} has no camera column.")
    x = _number(xval, f"Row {row_number} relative column", 0, 1) if xval else None
    native = _number(px, f"Row {row_number} pixel column", -.5, width - .5) if px else None
    if x is not None and native is not None:
        if abs((native + .5) / width - x) > max(1e-7, .026 / width):
            raise Problem(f"Row {row_number} records different relative and native positions.")
    return x if x is not None else (native + .5) / width


def _row(values, columns, index, width):
    if len(values) != len(columns):
        raise Problem(f"Row {index} has the wrong number of fields.")
    row = dict(zip(columns, values, strict=True))
    label = (row.get("name") or f"Line {index - 1}").strip()
    raw_weight = (row.get("weight") or "1").strip()
    nm = _number(row["nm"].strip(), f"Row {index} wavelength", 320, 1200)
    role = (row.get("role") or ROLE_FIT).strip().lower()
    if role not in (ROLE_FIT, ROLE_DISABLED):
        raise Problem(f"Row {index} role must be {ROLE_FIT} or {ROLE_DISABLED}.")
    return {"name": plain_text(label, "Line name", 58), "column": _column(row, width, index),
            "nm": nm, "weight": _number(raw_weight, f"Row {index} weight", .05, 50),
            "role": role}


def parse_anchors(raw: bytes, width):
    """Read anchors plus any explicitly disabled boundary pairs from a ruler table."""
    text = _decode(raw)
    reader = csv.reader(StringIO(text), strict=True)
    try:
        header = next(reader, [])
        cols = _columns(header)
        anchors, disabled = [], []
        for index, entry in enumerate(reader, 2):
            if not entry or not any(item.strip() for item in entry):
                continue
            row = _row(entry, cols, index, width)
            (anchors if row["role"] == ROLE_FIT else disabled).append(row)
            if len(anchors) + len(disabled) > 28:
                raise Problem("Use no more than 28 wavelength anchors.")
    except csv.Error as err:
        raise Problem("Read the wavelength anchors from a well-formed comma-separated table.") from err
    if len(anchors) < 2:
        raise Problem("At least two wavelength anchors are required.")
    return anchors, disabled


def _excluded_pairs(disabled):
    """Consecutive ``disabled`` rows bracket an interval that must stay unmeasured."""
    excluded = []
    if not disabled:
        return excluded
    columns = sorted(line["column"] for line in disabled)
    if len(columns) % 2:
        raise Problem("A disabled ruler table needs paired boundary rows.")
    for a, b in zip(columns[::2], columns[1::2]):
        if b - a < 1e-6:
            raise Problem("A disabled ruler interval must be wider than one pixel.")
        excluded.append([a, b])
    return excluded


def import_fit(raw, width, metadata):
    data = mapping(metadata, "Calibration import")
    allowed(data, {"title", "fit", "limit"}, "calibration import")
    title = plain_text(data.get("title", "Imported wavelength table"), "Calibration title", 86)
    mode = choice(data.get("fit", "linear"), ("linear", "interpolate"), "Calibration fit")
    limit = choice(data.get("limit", "anchors"), ("anchors", "frame"), "Axis support")
    anchors, disabled = parse_anchors(raw, width)
    kept = [{key: value for key, value in row.items() if key != "role"} for row in anchors]
    fit = Fit.parse({"title": title, "method": mode, "limit": limit, "anchors": kept,
                     "excluded": _excluded_pairs(disabled)})
    return fit


def export_fit(fit: Fit, width: int, portable: bool):
    fields = PORTABLE if portable else DATA_COLUMNS
    anchors = []
    for item in fit.anchors:
        row = {**item.public(), "role": ROLE_FIT}
        if portable:
            row["native_px"] = rounded(item.column * width - .5, 6)
        anchors.append(row)
    anchors.sort(key=lambda row: row["column"])
    for low, high in fit.excluded:
        for column in (low, high):
            anchors.append({"name": "Disabled boundary", "nm": rounded(fit._raw(np.asarray([column]))[0]),
                            "column": rounded(column), "weight": 1., "role": ROLE_DISABLED,
                            **({"native_px": rounded(column * width - .5, 6)} if portable else {})})
    anchors.sort(key=lambda row: row["column"])
    return table_bytes(anchors, fields)
