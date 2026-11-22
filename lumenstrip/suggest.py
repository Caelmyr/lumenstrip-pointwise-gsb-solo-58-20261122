"""Suggest a conservative wavelength ruler from repeated clicks on known lines.

Each line's repeated clicks are combined by their median, so one shaky click cannot drag a
wavelength. An independent check line is never fitted. A fitting point is left out only when
the clicks cannot otherwise form one ordered ruler, and a stretch is disabled when the check
line contradicts it or when a wide anchor gap has no independent confirmation. Disabled
intervals are written into the ruler itself, so every later reading skips them instead of
interpolating through them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import Anchor, Fit
from .contracts import Problem, allowed, choice, mapping, plain_text, real, rounded, seq
from .view import BLUE, CYAN, RED, Region, Series, plot

METHODS = ("linear", "interpolate")
ROLES = ("fit", "check")
PRIORITY = {"linear": 0, "interpolate": 1}


@dataclass(frozen=True)
class Line:
    name: str
    wavelength: float
    reads: tuple[float, ...]
    role: str
    weight: float

    @classmethod
    def parse(cls, value, index: int) -> Line:
        data = mapping(value, "Measured line")
        allowed(data, {"name", "nm", "reads", "role", "weight"}, "measured line")
        reads = tuple(real(v, f"Line {index + 1} click", 0, 1)
                      for v in seq(data.get("reads"), f"Line {index + 1} repeated clicks", 1, 8))
        return cls(plain_text(data.get("name", f"Line {index + 1}"), "Line name", 58),
                   real(data.get("nm"), "Wavelength (nm)", 320, 1200), reads,
                   choice(data.get("role", "fit"), ROLES, "Line role"),
                   real(data.get("weight", 1.), "Line weight", .05, 50))

    @property
    def column(self) -> float:
        return float(np.median(self.reads))

    @property
    def spread(self) -> float:
        return float(max(self.reads) - min(self.reads))

    def public(self) -> dict:
        return {"name": self.name, "nm": self.wavelength, "reads": list(self.reads),
                "role": self.role, "weight": self.weight}


@dataclass(frozen=True)
class Proposal:
    title: str
    methods: tuple[str, ...]
    limit: str
    lines: tuple[Line, ...]
    max_gap: float
    tolerance: float
    repeat_limit: float

    @classmethod
    def parse(cls, value) -> Proposal:
        data = mapping(value, "Ruler suggestion")
        allowed(data, {"title", "methods", "limit", "lines", "max_gap", "tolerance",
                       "repeat_limit_px"}, "ruler suggestion")
        methods = tuple(choice(v, METHODS, "Suggested fit") for v in
                        seq(data.get("methods", list(METHODS)), "Suggested fits", 1, 2))
        if len(set(methods)) != len(methods):
            raise Problem("Choose each suggested fit only once.")
        lines = tuple(Line.parse(v, i) for i, v in enumerate(
            seq(data.get("lines"), "Measured lines", 3, 34)))
        if not any(line.role == "check" for line in lines):
            raise Problem("Mark one measured line as the independent check.")
        return cls(plain_text(data.get("title", "Suggested wavelength ruler"), "Calibration name", 86),
                   methods,
                   choice(data.get("limit", "anchors"), ("anchors", "frame"), "Suggested support"),
                   lines,
                   real(data.get("max_gap", .16), "Largest trusted anchor gap", .04, .9),
                   real(data.get("tolerance", .75), "Check-line tolerance (nm)", .02, 20),
                   real(data.get("repeat_limit_px", 3.), "Repeated-click allowance (px)", .1, 50))

    def public(self) -> dict:
        return {"title": self.title, "methods": list(self.methods), "limit": self.limit,
                "max_gap": self.max_gap, "tolerance": self.tolerance,
                "repeat_limit_px": self.repeat_limit,
                "lines": [line.public() for line in self.lines]}


def _fit(lines, proposal: Proposal, method: str, excluded=()) -> Fit:
    return Fit.parse({"title": proposal.title, "method": method, "limit": proposal.limit,
                      "anchors": [{"column": line.column, "nm": line.wavelength,
                                   "name": line.name, "weight": line.weight} for line in lines],
                      "excluded": [list(pair) for pair in excluded]})


def _try_fit(lines, proposal, method, excluded=()):
    try:
        return _fit(lines, proposal, method, excluded)
    except Problem:
        return None


def _raw(fit: Fit, column: float) -> float:
    return float(fit._raw(np.asarray([column]))[0])


def _merge(spans):
    merged = []
    for low, high in sorted(((min(a, b), max(a, b)) for a, b in spans)):
        if merged and low <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _inset(merged, width: int):
    """A boundary anchor stays measured; only the stretch between two anchors is disabled."""
    edge = min(.5 / max(width, 1), .0005)
    rows = []
    for low, high in merged:
        a = min(low + edge, (low + high) / 2)
        b = max(high - edge, (low + high) / 2)
        if b - a > 1e-6:
            rows.append((a, b))
    return rows


def _between(points, column):
    left = [line for line in points if line.column < column]
    right = [line for line in points if line.column > column]
    return (left[-1], right[0]) if left and right else (None, None)


def _span_around(line, points):
    left, right = _between(points, line.column)
    if left is not None and right is not None:
        return left.column, right.column
    return (min(item.column for item in points), line.column) if right else (
        line.column, max(item.column for item in points))


def _residual_rms(lines) -> float:
    """How well one straight ruler explains these points; lower means more self-consistent."""
    u = np.asarray([line.column for line in lines])
    w = np.asarray([line.wavelength for line in lines])
    weights = np.asarray([line.weight for line in lines])
    mean_u = np.average(u, weights=weights)
    mean_w = np.average(w, weights=weights)
    slope = np.dot(weights * (u - mean_u), w - mean_w) / np.dot(weights, (u - mean_u) ** 2)
    return float(np.sqrt(np.average((w - (mean_w + slope * (u - mean_u))) ** 2, weights=weights)))


def _order_clean(usable, proposal: Proposal, method: str):
    """Drop only the points the other clicks contradict, so the rest describe one ruler."""
    remaining = sorted(usable, key=lambda line: line.column)
    dropped = []
    while len(remaining) > 2 and _try_fit(remaining, proposal, method) is None:
        options = []
        for line in remaining:
            trial = [item for item in remaining if item is not line]
            if _try_fit(trial, proposal, method) is None:
                continue
            try:
                options.append((_residual_rms(trial), line.column, line))
            except (ValueError, ZeroDivisionError):
                continue
        if not options:
            break
        _, _, pick = min(options, key=lambda item: item[:2])
        remaining.remove(pick)
        dropped.append(pick)
    return remaining, dropped


def _evaluate(proposal: Proposal, usable, checks, method: str, width: int):
    """Build one candidate ruler. Only an independent check line decides what stays measured."""
    points, conflict = _order_clean(usable, proposal, method)
    if len(points) < 2:
        raise Problem("Fewer than two fitting lines can describe a ruler.")
    if len({round(line.column, 6) for line in points}) != len(points):
        raise Problem("Two fitting lines were clicked at the same detector column.")
    fit = _try_fit(points, proposal, method)
    if fit is None:
        raise Problem("The suggested fitting lines cannot describe a single ruler.")
    ordered = sorted(points, key=lambda line: line.column)
    low, high = ordered[0].column, ordered[-1].column
    inside_span = [line for line in checks if low < line.column < high]

    intervals = []  # (low, high, kind, reason)
    for line in conflict:
        intervals.append((*_span_around(line, ordered), "conflict",
                          f"'{line.name}' is not used because its wavelength conflicts with "
                          "the order of the other clicks"))
    for a, b in zip(ordered, ordered[1:]):
        if b.column - a.column > proposal.max_gap:
            intervals.append((a.column, b.column, "wide gap",
                              f"anchors {a.name} and {b.name} are {b.column - a.column:.3f} of "
                              "the image apart"))
    for line in inside_span:
        residual = _raw(fit, line.column) - line.wavelength
        if abs(residual) <= proposal.tolerance:
            continue
        if any(a <= line.column <= b for a, b, *_ in intervals):
            continue
        left, right = _between(ordered, line.column)
        if left is not None and right is not None:
            intervals.append((left.column, right.column, "check",
                              f"the check line '{line.name}' misses by {residual:+.3f} nm "
                              f"between {left.name} and {right.name}"))

    grouped = {}
    for a, b, kind, reason in intervals:
        grouped.setdefault((min(a, b), max(a, b)), []).append((kind, reason))
    disabled, kept = [], []
    for (a, b), items in sorted(grouped.items()):
        seen = [line for line in inside_span if a < line.column < b]
        bad = [line for line in seen
               if abs(_raw(fit, line.column) - line.wavelength) > proposal.tolerance]
        if seen and not bad:
            kept.append({"relative_columns": [rounded(a), rounded(b)],
                         "check_lines": [line.name for line in seen],
                         "kinds": sorted({kind for kind, _ in items}),
                         "reasons": [reason for _, reason in items]})
        else:
            detail = ("no independent line confirms this stretch" if not seen
                      else "the independent line inside it disagrees")
            disabled.append((a, b, [reason + "; " + detail for _, reason in items]))
    merged = _inset(_merge([(a, b) for a, b, _ in disabled]), width)
    reasons = {}
    for a, b, texts in disabled:
        low_edge, high_edge = min(a, b), max(a, b)
        reasons.setdefault((low_edge, high_edge), []).extend(texts)
    final = _try_fit(points, proposal, method, [pair for pair in merged])
    if final is None:
        raise Problem("The suggested ruler would not leave any measured interval.")

    rows = []
    for line in checks:
        raw = _raw(final, line.column)
        residual = raw - line.wavelength
        value, inside = final.mapped(np.asarray([line.column]))
        if not low <= line.column <= high:
            status = "outside the suggested measured span"
        elif abs(residual) > proposal.tolerance and any(
                a - 1e-9 <= line.column <= b + 1e-9 for a, b in merged):
            status = "disagrees; its interval is disabled"
        elif any(a - 1e-9 <= line.column <= b + 1e-9 for a, b in merged):
            status = "agrees, but its position is disabled for another reason"
        elif abs(residual) > proposal.tolerance:
            status = "disagrees with the suggested ruler"
        else:
            status = "verified"
        rows.append({"name": line.name, "nm": line.wavelength, "column": rounded(line.column),
                     "mapped_nm": rounded(value[0]), "raw_nm": rounded(raw),
                     "residual_nm": rounded(residual), "status": status})

    warnings = []
    for line in conflict:
        warnings.append(f"Point '{line.name}' was not used: its wavelength conflicts with the order of "
                        "the other clicks. Re-click it if it was mis-marked.")
    if not inside_span:
        warnings.append("No independent check line falls between the fitting points, so the span "
                        "between them stays unverified.")
    for row in rows:
        if row["status"] == "verified":
            continue
        if row["status"] == "outside the suggested measured span":
            warnings.append(f"Check line '{row['name']}' lies outside the suggested measured span.")
        elif row["status"] == "disagrees; its interval is disabled":
            warnings.append(f"Check line '{row['name']}' misses by {row['residual_nm']:+.3f} nm; "
                            "the stretch around it is disabled rather than smoothed over.")
        elif row["status"] == "disagrees with the suggested ruler":
            warnings.append(f"Check line '{row['name']}' misses by {row['residual_nm']:+.3f} nm.")
        else:
            warnings.append(f"Check line '{row['name']}' agrees, but its position lies in a disabled "
                            "stretch, so it does not verify that stretch.")
    for item in kept:
        warnings.append("Span {:.3f}–{:.3f} is kept only because {} agrees inside it; treat "
                        "ordinary readings there with care."
                        .format(item["relative_columns"][0], item["relative_columns"][1],
                                ", ".join(item["check_lines"])))
    for (a, b), texts in sorted(reasons.items()):
        lead = "Point left out" if any("is not used" in text for text in texts) else "Not measured"
        warnings.append(f"{lead} between {a:.3f} and {b:.3f} of the image: "
                        + "; ".join(dict.fromkeys(texts)) + ".")

    verified = [row for row in rows if row["status"] == "verified"]
    disagreeing = [row for row in rows
                   if row["status"].startswith("disagrees")]
    return {"method": method, "fit": final, "points": ordered, "checks": rows,
            "excluded": _public_spans(merged, reasons, final, width), "disabled": merged,
            "kept": kept, "warnings": warnings,
            "influence": influence(final, width, checks),
            "worst_check_nm": max((abs(row["residual_nm"]) for row in rows), default=None),
            "verified": len(verified), "disagreeing": len(disagreeing),
            "dropped": [line.name for line in conflict]}


def _public_spans(merged, reasons, fit: Fit, width: int):
    rows = []
    for a, b in merged:
        values = fit._raw(np.asarray([a, b]))
        texts = []
        for (low, high), found in reasons.items():
            if low <= a + 1e-9 and high >= b - 1e-9:
                texts.extend(found)
        rows.append({"relative_columns": [rounded(a), rounded(b)],
                     "native_columns": [rounded(a * width - .5, 3), rounded(b * width - .5, 3)],
                     "wavelength_nm": [rounded(min(values)), rounded(max(values))],
                     "width_px": rounded((b - a) * width, 3),
                     "reasons": list(dict.fromkeys(texts))})
    return rows


def influence(fit: Fit, width: int, checks=()) -> list[dict]:
    """Rank used anchors by how far a reading moves if that anchor is dropped.

    This answers "which of my clicked lines can change the reading most" without fitting the
    reading itself, so it is also available for any already saved ruler.
    """
    ordered = sorted(fit.anchors, key=lambda anchor: anchor.column)
    if len(ordered) < 3:
        return []
    grid = np.linspace(ordered[0].column, ordered[-1].column, 321)
    _, supported = fit.mapped(grid)
    grid = grid[supported]
    base = fit._raw(grid)
    if fit.method == "linear":
        u = np.asarray([a.column for a in ordered])
        w = np.asarray([a.weight for a in ordered])
        design = np.column_stack((np.ones(len(u)), u))
        leverage = [float(v) for v in w * np.einsum(
            "ij,jk,ik->i", design, np.linalg.pinv(design.T @ np.diag(w) @ design), design)]
    else:
        leverage = [None] * len(ordered)
    rows = []
    for index, anchor in enumerate(ordered):
        others = [a for j, a in enumerate(ordered) if j != index]
        trial = (Fit.parse({"title": fit.title, "method": fit.method, "limit": fit.limit,
                            "anchors": [a.public() for a in others]}) if len(others) >= 2 else None)
        moved = (np.abs(trial._raw(grid) - base) if trial is not None and len(grid)
                 else np.full(max(len(grid), 1), np.nan))
        reached = ([abs(_raw(trial, check.column) - _raw(fit, check.column)) for check in checks]
                   if trial is not None else [])
        gaps = [abs(anchor.column - other.column) for other in ordered if other is not anchor]
        rows.append({"name": anchor.name, "nm": anchor.wavelength,
                     "relative_column": rounded(anchor.column),
                     "largest_reading_change_nm": rounded(float(np.nanmax(moved)))
                     if np.any(np.isfinite(moved)) else None,
                     "check_reading_change_nm": rounded(max(reached)) if reached else None,
                     "leverage": rounded(leverage[index]) if leverage[index] is not None else None,
                     "nearest_other_line": rounded(min(gaps)) if gaps else None})
    rows.sort(key=lambda row: (-(row["largest_reading_change_nm"] or 0.),
                               -(row["check_reading_change_nm"] or 0.)))
    for rank, row in enumerate(rows, 1):
        row["influence_rank"] = rank
    return rows


def _rank(candidate):
    return (candidate["disagreeing"], len(candidate["excluded"]),
            PRIORITY[candidate["method"]], -candidate["verified"])


def chart(candidate):
    """Plot a candidate's measured intervals, disabled spans and every clicked line."""
    fit = candidate["fit"]
    columns = np.linspace(0, 1, 401)
    values, mask = fit.mapped(columns)
    return plot("Suggested wavelength ruler", "Relative camera column", "Wavelength (nm)",
                [Series("Measured interval", np.where(mask, columns, np.nan),
                        np.where(mask, values, np.nan), CYAN),
                 Series("Not measured", np.where(~mask, columns, np.nan),
                        np.where(~mask, fit._raw(columns), np.nan), BLUE)],
                x_bounds=(0., 1.),
                regions=[Region(row["relative_columns"][0], row["relative_columns"][1],
                                "disabled", RED) for row in candidate["excluded"]],
                markers=[(line.column, line.wavelength, line.name) for line in candidate["points"]] +
                        [(row["column"], row["nm"], row["name"]) for row in candidate["checks"]])


def recommend(proposal: Proposal, width: int):
    lines = list(proposal.lines)
    fits = [line for line in lines if line.role == "fit"]
    checks = [line for line in lines if line.role == "check"]
    usable = [line for line in fits if line.spread * width <= proposal.repeat_limit]
    shaky = [line for line in fits if all(line is not item for item in usable)]
    if len(usable) < 2:
        names = ", ".join(line.name for line in shaky) or "the fitting lines"
        raise Problem(f"At least two fitting lines must agree with their repeated clicks; review {names}.")
    candidates, failures = [], []
    for method in proposal.methods:
        try:
            candidates.append(_evaluate(proposal, usable, checks, method, width))
        except Problem as error:
            failures.append(f"{method}: {error}")
    if not candidates:
        raise Problem("No suggested ruler could be built. " + " ".join(failures))
    best = min(candidates, key=_rank)
    used = {line.name for line in best["points"]}
    diagnostics = []
    dropped = list(best["dropped"]) + [line.name for line in shaky]
    for line in sorted(lines, key=lambda item: item.column):
        if line.role == "check":
            state = "check only"
        elif any(line is item for item in shaky):
            state = "not used: repeated clicks disagree"
        elif line.name in used:
            state = "fitting line used"
        else:
            state = "not used: wavelength conflicts with the other clicks"
        diagnostics.append({"name": line.name, "nm": line.wavelength, "role": line.role,
                            "reads": [rounded(v) for v in line.reads],
                            "combined_column": rounded(line.column),
                            "combined_native_px": rounded(line.column * width - .5, 3),
                            "repeat_spread_px": rounded(line.spread * width, 3),
                            "offset_from_ruler_nm": rounded(
                                line.wavelength - _raw(best["fit"], line.column), 3),
                            "weight": line.weight, "state": state})
    warnings = []
    for line in shaky:
        warnings.append(f"Line '{line.name}' was not used: its repeated clicks are "
                        f"{line.spread * width:.2f} px apart, above the {proposal.repeat_limit:g} px allowance. "
                        "Re-click it or leave it out.")
    warnings += best["warnings"]
    public = [{"method": item["method"], "fit": item["fit"].description(width),
               "excluded": item["excluded"], "check_lines": item["checks"],
               "kept_wide_spans": item["kept"], "checks_verified": item["verified"],
               "checks_disagreeing": item["disagreeing"],
               "worst_check_error_nm": rounded(item["worst_check_nm"]),
               "influence": item["influence"], "warnings": item["warnings"]}
              for item in candidates]
    shown = {key: value for key, value in best.items() if key != "fit"}
    shown["fit"] = best["fit"]
    return {"proposal": proposal.public(), "recommended": best["method"], "candidates": public,
            "method": best["method"], "check_lines": best["checks"],
            "checks_verified": best["verified"], "checks_disagreeing": best["disagreeing"],
            "chosen": shown,
            "settings": best["fit"].public(), "fit": best["fit"].description(width),
            "lines": diagnostics, "excluded": best["excluded"], "influence": best["influence"],
            "kept_wide_spans": best["kept"], "warnings": warnings,
            "checks": best["checks"], "validated": best["kept"],
            "summary": {"fitting_lines": len(best["points"]), "check_lines": len(checks),
                        "not_used_lines": dropped,
                        "excluded_intervals": len(best["excluded"]),
                        "checks_verified": best["verified"],
                        "checks_disagreeing": best["disagreeing"],
                        "check_tolerance_nm": proposal.tolerance},
            "fit_object": best["fit"]}
