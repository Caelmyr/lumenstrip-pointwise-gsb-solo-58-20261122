"""Estimate each lamp's light inside one capture where both lamps were lit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import NamedWindow, Problem, Window, allowed, flag, real, rounded, seq
from .result import Result
from .spectral import area
from .strip import contiguous, resample
from .view import AMBER, BLUE, CYAN, GREEN, PINK, Region, Series, plot, support_map


# roles written into view.support_map: amber is readable but not separated
READABLE = 1
LAMP_ONE = 2
LAMP_TWO = 3


@dataclass(frozen=True)
class Mix:
    alpha: float
    beta: float
    offset: float
    se_alpha: float
    se_beta: float
    controls: int
    residual_fraction: float | None


@dataclass(frozen=True)
class Share:
    fraction: float | None
    one_error: float | None
    two_error: float | None


def _templates(mixed, first, second):
    x, mixed_values, known = mixed.axis_array()
    one = resample(x, *first.axis_array())
    two = resample(x, *second.axis_array())
    usable = known & np.isfinite(one) & np.isfinite(two)
    return x, mixed_values, known, one, two, usable


def _fit(x, mixed, first, second, usable, lower, upper):
    selected = usable & (x >= lower) & (x <= upper)
    count = int(selected.sum())
    if count < 8:
        return None, "fewer than eight columns in the analysis window are readable in all three images"
    design = np.column_stack([first[selected], second[selected], np.ones(count)])
    if np.linalg.matrix_rank(design) < 3:
        return None, "the two single-lamp spectra are too alike in the analysis window to be told apart"
    coefficients, *_ = np.linalg.lstsq(design, mixed[selected], rcond=None)
    alpha, beta, offset = (float(v) for v in coefficients)
    residual = mixed[selected] - design @ coefficients
    scale = float(np.mean(np.abs(mixed[selected])))
    fraction = float(np.sqrt(np.mean(residual ** 2)) / scale) if scale > 1e-12 else None
    covariance = np.full((3, 3), np.nan)
    if count > 3:
        variance = float(residual @ residual) / (count - 3)
        try:
            covariance = variance * np.linalg.inv(design.T @ design)
        except np.linalg.LinAlgError:
            pass
    mix = Mix(alpha, beta, offset, float(np.sqrt(covariance[0, 0])),
              float(np.sqrt(covariance[1, 1])), count, fraction)
    return mix, None


def _window(x, own, other, run, name):
    relative = (own[run] - other[run]) / np.maximum(own[run], 1e-12)
    return {"name": name, "lower": float(x[run][0]), "upper": float(x[run][-1]),
            "samples": int(run.stop - run.start),
            "minimum_relative_dominance": rounded(float(np.min(relative)))}


def _auto_window(x, own, other, selected, distinctness, name):
    lit = selected & (own > 0)
    if not np.any(lit):
        return None, f"no readable stretch where {name} carries positive model light"
    lit &= own > .05 * float(np.max(own[lit]))
    dominant = lit & (other > 0) & (own >= (1 + distinctness) * other)
    best = None
    for run in contiguous(dominant):
        if run.stop - run.start < 3:
            continue
        gain = float(np.sum((own - other)[run]))
        if best is None or gain > best[0]:
            best = (gain, run)
    if best is None:
        return None, f"no readable stretch where {name} clearly stands above the other lamp"
    return _window(x, own, other, best[1], name + " separation stretch"), None


def _given_window(x, own, other, selected, zone, distinctness, coverage, name):
    inside = selected & (x >= zone.lower) & (x <= zone.upper)
    if int(inside.sum()) < 3:
        return None, f"the supplied window for {name} holds fewer than three columns readable in all three images"
    dominant = inside & (own > 0) & (other > 0) & (own >= (1 + distinctness) * other)
    share = float(dominant.sum()) / float(inside.sum())
    if share < coverage:
        return None, (f"inside the supplied window, {name} clearly stands above the other lamp on only "
                      f"{share:.0%} of readable columns (needs {coverage:.0%})")
    run = slice(0, int(inside.sum()))
    found = _window(x[inside], own[inside], other[inside], run, name + " separation stretch")
    return {**found, "lower": zone.lower, "upper": zone.upper, "dominance_fraction": rounded(share)}, None


def _shares(area_one, area_two, mix):
    """Return the one-lamp share and each share's error relative to itself."""
    one = mix.alpha * area_one
    two = mix.beta * area_two
    total = one + two
    if total <= 1e-12 or mix.alpha <= 0 or mix.beta <= 0:
        return Share(None, None, None)
    fraction = one / total
    # share = alpha*A / (alpha*A + beta*B); linearise the fit errors of alpha and beta
    error = float(np.sqrt((area_one * mix.se_alpha * two / total ** 2) ** 2
                          + (area_two * mix.se_beta * one / total ** 2) ** 2))
    return Share(float(fraction), error / fraction if fraction > 1e-12 else None,
                 error / (1 - fraction) if 1 - fraction > 1e-12 else None)


def _amount(x, values, valid, lo, hi, multiplier=1.):
    measure = area(x, values, valid, lo, hi)
    return {key: (None if value is None else rounded(value * multiplier))
            for key, value in measure.public(hi - lo).items()}


def _band_row(zone, x, mixed, one, two, mix, usable, known, multiplier, allocate, smallest):
    lo, hi = zone.interval.lower, zone.interval.upper
    own = area(x, mixed, known, lo, hi)
    total = area(x, mixed, usable, lo, hi)
    width = hi - lo
    row = {"band": zone.name, "lower": lo, "upper": hi,
           "original_integration": rounded(own.positive * multiplier),
           "original_signed_integration": rounded(own.signed * multiplier),
           "readable_combination": rounded(total.positive * multiplier),
           "readable_signed_combination": rounded(total.signed * multiplier),
           "readable_width": rounded(own.covered),
           "model_readable_width": rounded(total.covered),
           "unobserved_width": rounded(max(0., width - own.covered)),
           "readable_fraction": rounded(min(1., own.covered / width)) if width > 0 else None,
           "lamp_1_light": None, "lamp_2_light": None,
           "lamp_1_share_in_band": None, "lamp_2_share_in_band": None,
           "explained_light": None, "unexplained_light": None}
    if not allocate or total.covered <= 0:
        row["state"] = "combined only"
        return row
    one_values = mix.alpha * one
    two_values = mix.beta * two
    one_area = area(x, one_values, usable, lo, hi)
    two_area = area(x, two_values, usable, lo, hi)
    leftover = area(x, mixed - (one_values + two_values + mix.offset), usable, lo, hi)
    floor = mix.offset * one_area.covered
    positive = one_area.positive + two_area.positive
    share = one_area.positive / positive if positive > 1e-12 else None
    row.update({
        "lamp_1_light": rounded(one_area.positive * multiplier),
        "lamp_2_light": rounded(two_area.positive * multiplier),
        "lamp_1_share_in_band": rounded(share) if share is not None else None,
        "lamp_2_share_in_band": rounded(1 - share) if share is not None else None,
        "model_floor_light": rounded(floor * multiplier),
        "explained_light": rounded((one_area.positive + two_area.positive + floor) * multiplier),
        "unexplained_light": rounded(leftover.positive * multiplier),
        "unexplained_of_negative_sign": rounded(leftover.negative * multiplier),
        "unexplained_fraction": rounded(leftover.positive / total.positive)
        if total.positive > 1e-12 else None,
    })
    row["state"] = ("lamps allocated" if share is not None and min(share, 1 - share) >= smallest
                    else "combined only")
    return row


def two_lamp(bench):
    sources = bench.require(3, 3)
    bench.same_basis()
    mixed, first, second = sources[0], sources[1], sources[2]
    if mixed.trace.units != first.trace.units or mixed.trace.units != second.trace.units:
        raise Problem("All three images must share one axis type (calibrated or native columns).")
    cfg = bench.order.settings
    allowed(cfg, {"window", "separation", "distinctness", "min_coverage", "min_share",
                  "tolerance", "bands", "per_ms"}, "two-lamp mixture")
    distinctness = real(cfg.get("distinctness", .35), "Clear dominance margin", .01, 2.)
    coverage = real(cfg.get("min_coverage", .6), "Minimum dominance coverage", .05, 1.)
    smallest = real(cfg.get("min_share", .02), "Smallest believable lamp share", .0005, .45)
    tolerance = real(cfg.get("tolerance", .25), "Largest share relative error", .01, 5.)
    per_ms = flag(cfg.get("per_ms", False), "Divide by mixed exposure")
    multiplier = 1. / mixed.exposure if per_ms else 1.
    supplied = [item for item in seq(cfg.get("separation") or [], "Separation windows", 0, 2)
                if isinstance(item, dict) and isinstance(item.get("interval"), list)
                and len(item["interval"]) == 2 and item["interval"][1] > item["interval"][0]]
    if supplied and len(supplied) != 2:
        raise Problem("Give exactly two separation windows, one for each single-lamp image.")

    x, mixed_values, mixed_known, one_values, two_values, usable = _templates(
        mixed.trace, first.trace, second.trace)
    measured = mixed.trace.axis[np.isfinite(mixed.trace.axis) & mixed.trace.known]
    if not len(measured):
        raise Problem("The mixed capture has no readable columns in its saved strip.")
    bounds = Window.parse(cfg["window"], "Analysis window") if cfg.get("window") else None
    lower, upper = (bounds.lower, bounds.upper) if bounds else (float(measured.min()), float(measured.max()))

    mix, problem = _fit(x, mixed_values, one_values, two_values, usable, lower, upper)
    reasons = [problem] if problem else []
    separation = [None, None]
    model_usable = usable & (x >= lower) & (x <= upper)
    zones = [NamedWindow.read(v, i) for i, v in enumerate(supplied)] if supplied else []
    if mix is not None:
        for index, source in enumerate((first, second)):
            own = (mix.alpha if index == 0 else mix.beta) * (one_values if index == 0 else two_values)
            other = (mix.beta if index == 0 else mix.alpha) * (two_values if index == 0 else one_values)
            if zones:
                zone = zones[index].interval
                if zone.lower < lower or zone.upper > upper:
                    reasons.append(f"the separation window for {source.title} leaves the analysis window")
                    continue
                found, issue = _given_window(x, own, other, model_usable, zone, distinctness, coverage,
                                             source.title)
            else:
                found, issue = _auto_window(x, own, other, model_usable, distinctness, source.title)
            if issue:
                reasons.append(issue)
            separation[index] = found

    allocate = mix is not None and all(separation)
    if mix is not None:
        if mix.alpha <= 0 or mix.beta <= 0:
            allocate = False
            reasons.append("one fitted lamp amplitude is not positive, so both lamps are not present")
    totals = _amount(x, mixed_values, model_usable, lower, upper, multiplier)
    readable = _amount(x, mixed_values, mixed_known, lower, upper, multiplier)
    share = Share(None, None, None)
    if allocate:
        one_area = area(x, one_values, model_usable, lower, upper).positive
        two_area = area(x, two_values, model_usable, lower, upper).positive
        share = _shares(one_area, two_area, mix)
        if share.fraction is None:
            allocate = False
            reasons.append("one lamp contributes no positive light inside the analysis window")
        elif min(share.fraction, 1 - share.fraction) < smallest:
            allocate = False
            reasons.append(f"one lamp holds less than the {smallest:.1%} smallest believable share")
        elif share.one_error is None or share.two_error is None or max(share.one_error, share.two_error) > tolerance:
            allocate = False
            reasons.append("the readable separation stretches do not pin the two shares down precisely enough")
    if not allocate:
        share = Share(None, None, None)

    bands = seq(cfg.get("bands") or [{"name": "Analysis window", "interval": [lower, upper]}],
                "Allocation windows", 1, 12)
    rows = [_band_row(NamedWindow.read(raw, i), x, mixed_values, one_values, two_values,
                      mix, model_usable, mixed_known, multiplier, allocate, smallest)
            for i, raw in enumerate(bands)]

    result = Result(
        "two-lamp-mixture", "Two lamps in one capture",
        "Two single-lamp captures model the mixed capture. Shares appear only where the templates prove "
        "the lamps separate; otherwise only the readable combined light is reported.")
    result.metrics = {
        "estimate": "each lamp's share" if allocate else "combined readable light only",
        "estimate_reason": None if allocate else "; ".join(dict.fromkeys(reasons)) or "the two lamps are not separable",
        "lamp_1": first.title, "lamp_2": second.title,
        "lamp_1_share": rounded(share.fraction) if allocate else None,
        "lamp_2_share": rounded(1 - share.fraction) if allocate else None,
        "lamp_1_relative_standard_error": rounded(share.one_error) if allocate else None,
        "lamp_2_relative_standard_error": rounded(share.two_error) if allocate else None,
        "combined_readable_light": readable["positive_area"],
        "combined_readable_signed_light": readable["signed_area"],
        "combined_light_on_model_support": totals["positive_area"],
        "readable_width": readable["observed_width"],
        "unobserved_width": readable["unobserved_width"],
        "readable_fraction": readable["supported_fraction"],
        "analysis_window": [rounded(lower), rounded(upper)],
        "axis_units": mixed.trace.units,
        "separated_stretches": sum(item is not None for item in separation),
        "mixed_exposure_ms": mixed.exposure,
        "exposure_normalized": per_ms,
        "fitted_lamp_1_amplitude": rounded(mix.alpha) if allocate else None,
        "fitted_lamp_2_amplitude": rounded(mix.beta) if allocate else None,
        "fitted_common_offset": rounded(mix.offset) if allocate else None,
        "model_columns": mix.controls if allocate else None,
        "model_residual_fraction": mix.residual_fraction if allocate else None,
    }
    for number, item in enumerate((item for item in separation if item), 1):
        result.metrics[f"separation_stretch_{number}"] = [rounded(item["lower"]), rounded(item["upper"])]
    anchors = []
    for number, (source, item) in enumerate(zip((first, second), separation), 1):
        if not item:
            continue
        found = np.isfinite(mixed.trace.axis) & (mixed.trace.axis >= item["lower"]) & (
            mixed.trace.axis <= item["upper"])
        columns = [int(column) for column in mixed.trace.plan.x[found]]
        anchors.append({"band": f"Lamp {number} separation support", "lower": rounded(item["lower"]),
                        "upper": rounded(item["upper"]), "source": source.title,
                        "native_columns": columns, "native_column_count": len(columns),
                        "state": "estimate anchor" if allocate else "not used for the estimate"})
        if columns:
            result.metrics[f"lamp_{number}_separation_columns"] = [columns[0], columns[-1]]
    result.rows = rows + anchors

    regions = [Region(item["lower"], item["upper"], item["name"], GREEN if number == 0 else BLUE)
               for number, item in enumerate(separation) if item]
    plotted = np.where(usable, mixed_values * multiplier, np.nan)
    shapes = [Series("Mixed capture", x, plotted)]
    if mix is not None:
        for number, (amplitude, curve) in enumerate(zip((mix.alpha, mix.beta),
                                                        (one_values, two_values), strict=True)):
            if separation[number]:
                shapes.append(Series(f"Lamp {number + 1} model", x,
                                     np.where(model_usable, amplitude * curve * multiplier, np.nan),
                                     PINK if number == 0 else BLUE))
    result.chart("two-lamp-mixture", "Mixed capture and the fitted lamp contributions", plot(
        "Two-lamp mixture", mixed.trace.units, "Signal / ms" if per_ms else "Linear signal",
        shapes, regions=regions))
    if allocate:
        leftover = mixed_values - (mix.alpha * one_values + mix.beta * two_values + mix.offset)
        result.chart("two-lamp-unexplained", "Light the two-lamp model does not explain", plot(
            "Unexplained light", mixed.trace.units, "Signal / ms" if per_ms else "Linear signal",
            [Series("Mixed minus model", x, np.where(model_usable, leftover * multiplier, np.nan), AMBER)],
            horizontal=0., regions=regions))

    trace = mixed.trace
    states = np.zeros(len(trace.axis), dtype=np.uint8)
    inside = (trace.axis >= lower) & (trace.axis <= upper) & trace.known
    states[inside] = READABLE
    for item, value in zip(separation, (LAMP_ONE, LAMP_TWO), strict=True):
        if item:
            states[(trace.axis >= item["lower"]) & (trace.axis <= item["upper"]) & trace.known] = value
    result.image("two-lamp-support", "Original capture positions supporting this estimate",
                 support_map(mixed.camera, trace, states), {"kind": "source", "source_index": 0})

    support_rows = []
    for i in np.flatnonzero(inside):
        role = ("lamp 1 separation" if states[i] == LAMP_ONE else
                "lamp 2 separation" if states[i] == LAMP_TWO else "readable, not separated")
        support_rows.append({"axis": rounded(trace.axis[i]), "native_column": int(trace.plan.x[i]),
                             "source_uv": trace.row(i)["source_uv"], "support_role": role,
                             "mixed_signal": rounded(trace.signal[i] * multiplier),
                             "state": trace.row(i)["state"]})
    curves = [{"axis": rounded(x[i]),
               "mixed_signal": rounded(mixed_values[i] * multiplier),
               "single_lamp_1": rounded(one_values[i] * multiplier),
               "single_lamp_2": rounded(two_values[i] * multiplier),
               "readable_in_all_three": bool(usable[i])} for i in range(len(x))]
    if allocate:
        for i, row in enumerate(curves):
            row.update({"lamp_1_model": rounded(mix.alpha * one_values[i] * multiplier),
                        "lamp_2_model": rounded(mix.beta * two_values[i] * multiplier),
                        "model_floor": rounded(mix.offset * multiplier),
                        "unexplained": rounded((mixed_values[i] - (mix.alpha * one_values[i]
                                                                   + mix.beta * two_values[i]
                                                                   + mix.offset)) * multiplier)
                        if usable[i] else None})
    result.csv("two-lamp-band-areas", "Original integration and each lamp's light", rows)
    result.csv("two-lamp-support-columns", "Original camera columns behind the estimate", support_rows)
    result.csv("two-lamp-curves", "Mixed and single-lamp light on the original axis", curves)
    if not allocate:
        result.warnings.append("Only the combined readable light is given: "
                               + result.metrics["estimate_reason"] + ".")
    exposures = {round(source.exposure, 6) for source in sources}
    if len(exposures) > 1:
        result.warnings.append("The three captures used different exposure times; fitted amplitudes absorb that difference.")
    if mixed.role != "sample":
        result.warnings.append("The first image is read as the mixed capture even though it is not labelled a sample.")
    return result
