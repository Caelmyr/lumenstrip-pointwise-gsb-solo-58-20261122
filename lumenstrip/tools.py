"""Catalog for normal workbench tools and their settings."""

from dataclasses import dataclass
from typing import Callable

from .coverage import common_coverage, discontinuities
from .profile import distributions, energy_profile
from .spot import energy_spot, spot_balance
from .filtering import filter_contrast, filter_requirements, DEFAULT as FILTER_BANDS
from .lanes import DEFAULT, parallel, projection
from .colour import channel_balance, channel_guard, color_bands
from .noise import noise_windows, sidebands
from .continuum import envelopes, flatness, slope_edges
from .instrument import dark_field, exposure_response, trace_placement
from .comparison import band_comparison, drift, overlay, repeatability, transmission
from .readings import (aperture, calibration_check, diagnostics, height_scan, spectrum, support_audit,
                       tiles, trace_path)
from .spectral import bands, crest_report, visible_balance


@dataclass(frozen=True)
class Field:
    key: str
    kind: str
    title: str
    default: object
    units: str = ""
    step: float | None = None
    choices: tuple[str, ...] = ()

    def public(self):
        return self.__dict__


def number(key, title, default, step=1., units=""):
    return Field(key, "number", title, default, units, step)


def boolean(key, title, default):
    return Field(key, "boolean", title, default)


def select(key, title, default, *values):
    return Field(key, "choice", title, default, choices=tuple(values))


def structured(key, title, default, shape="json"):
    return Field(key, shape, title, default)


@dataclass(frozen=True)
class Tool:
    name: str
    label: str
    section: str
    minimum: int
    maximum: int
    description: str
    settings: tuple[Field, ...]
    run: Callable

    def public(self):
        return {"id": self.name, "label": self.label, "section": self.section,
                "sources": [self.minimum, self.maximum], "description": self.description,
                "settings": [setting.public() for setting in self.settings]}


GUARDS = (number("low", "Shadow code", .02, .005), number("high", "Clip code", .985, .002))
RATE = boolean("per_ms", "Divide by exposure time", False)
BANDS = structured("bands", "Named band intervals", [])
TOOLS = {tool.name: tool for tool in (
    Tool("spectrum", "Strip spectrum", "Strip", 1, 1,
         "Read native camera samples along the retained strip geometry.",
         (boolean("show_native", "Show original strip level", True),
          boolean("show_background", "Show observed sidebands", False)), spectrum),
    Tool("trace-path", "Rows taken on original", "Strip", 1, 1,
         "Mark the original capture with the exact native rows each column used and the spans left unread.",
         (number("columns", "Columns to list", 65),), trace_path),
    Tool("strip-height", "Cross-strip response", "Strip", 1, 1,
         "Measure signal and native support above and below the chosen center.",
         (number("radius", "Vertical radius", 13, 1, "px"),
          number("quantile", "Quantile tail", .2, .025)), height_scan),
    Tool("calibration-check", "Wavelength ruler", "Strip", 1, 1,
         "Check the active ruler's support and residuals at marked wavelengths.",
         (number("columns", "Inspection samples", 65),), calibration_check),
    Tool("support-runs", "Unmeasured runs", "Strip", 1, 1,
         "Locate places lost to the camera, calibration, guard or sideband coverage.",
         (number("minimum", "Minimum contiguous run", 3, 1, "columns"),), support_audit),
    Tool("camera-signal", "Camera codes", "Camera", 1, 1,
         "View clipping and transparency separately, with native per-channel histograms.",
         (*GUARDS, number("bins", "Histogram bins", 28)), diagnostics),
    Tool("signal-grid", "Spatial support", "Camera", 1, 1,
         "Find spatially localized saturation, texture or transparent parts.",
         (number("columns", "Tiles across", 8), number("rows", "Tiles down", 4), *GUARDS), tiles),
    Tool("camera-patch", "Native patch", "Camera", 1, 1,
         "Read a selected native region and its statistics column by column.",
         (structured("window", "Image window", [[.28, .37], [.70, .62]], "crop"),
          select("channel", "Channel", "linear-luma", "linear-luma", "red", "green", "blue"), *GUARDS), aperture),
    Tool("emission-lines", "Emission peaks", "Spectrum", 1, 1,
         "Locate local peaks, observed line widths and supporting source columns.",
         (number("shoulder", "Shoulder window", 16, 1, "columns"),
          number("contrast", "Minimum line contrast", .035, .005),
          number("spacing", "Minimum line spacing", 4, .25),
          number("maximum", "Maximum reported lines", 30)), crest_report),
    Tool("band-area", "Integrated bands", "Spectrum", 1, 1,
         "Integrate a defined band without bridging unmeasured wavelengths.",
         (BANDS, RATE), bands),
    Tool("visible-balance", "Relative visible bands", "Spectrum", 1, 1,
         "Compare broad camera-response cues on the calibrated visible region.",
         (select("target", "Plot emphasis", "all", "all", "violet", "green", "red"),), visible_balance),
    Tool("repeat-overlay", "Aligned repeats", "Compare", 2, 4,
         "Overlay photographs from the studio on their common ruler; keep them independent.",
         (number("count", "Grid samples", 360),
          select("normalize", "Normalize each peak", "none", "none", "max"), RATE), overlay),
    Tool("transmission", "Filter transmission", "Compare", 2, 3,
         "Select a sample first, the open lamp second and optionally a dark image third.",
         (number("count", "Grid samples", 420), number("floor", "Open signal floor", .00008, .00001),
          select("mode", "Display", "transmission", "transmission", "density")), transmission),
    Tool("repeat-spread", "Repeat spread", "Compare", 2, 4,
         "Find wavelengths where repeated cuts disagree on shared measured evidence.",
         (number("count", "Grid samples", 360), number("limit", "Absolute spread limit", .04, .002), BANDS, RATE),
         repeatability),
    Tool("axis-drift", "Axis drift check", "Compare", 2, 2,
         "Check nearby offsets by correlation while retaining both original rulers.",
         (number("count", "Grid samples", 320), number("range", "Search either side", 12),
          number("min_common", "Minimum shared support", 40)), drift),
    Tool("source-coverage", "Shared wavelength support", "Compare", 2, 4,
         "Measure the individual and shared extents of observed wavelengths.",
         (BANDS, number("samples", "Grid samples", 380),
          boolean("show_individual", "Show source-specific support", True)), common_coverage),
    Tool("coverage-gaps", "Gaps across observations", "Compare", 2, 4,
         "List contiguous unsupported runs in the selected observation set.",
         (number("samples", "Grid samples", 420), number("minimum", "Minimum gap samples", 5),
          select("consider", "Count coverage", "any", "any", "shared")), discontinuities),
    Tool("signal-distribution", "Spectral level distribution", "Spectrum", 1, 1,
         "Inspect the distribution of measured camera light inside selected bands.",
         (BANDS, number("bins", "Histogram bins", 35), number("tail", "Quantile tail", .1, .01)), distributions),
    Tool("band-energy", "Measured energy width", "Spectrum", 1, 1,
         "Find wavelength intervals containing most measured band energy.",
         (BANDS, number("tail", "Energy tail fraction", .1, .01), RATE), energy_profile),
    Tool("spot-energy", "Detector spot energy", "Camera", 1, 1,
         "Measure the centroid and the energy spread of a selected bright detector spot.",
         (structured("window", "Image window", [[.42, .4], [.58, .66]], "crop"),
          number("border", "Background border width", 2), number("clip", "Clip code", .99, .001),
          number("bins", "Radius sample bins", 26)), energy_spot),
    Tool("spot-projection", "Detector spot projections", "Camera", 1, 1,
         "Inspect horizontal and vertical spread using measured native light.",
         (structured("window", "Image window", [[.44, .39], [.62, .72]], "crop"),
          number("border", "Background border width", 2), number("clip", "Clip code", .99, .001),
          boolean("normalize", "Normalize intensities", False)), spot_balance),
    Tool("filter-specs", "Filter window qualification", "Compare", 2, 3,
         "Check a single measured filter against pass or stop bands with separate missing support.",
         (structured("bands", "Required pass and stop windows", FILTER_BANDS),
          number("floor", "Minimum open signal", .00008, .00001), number("count", "Grid samples", 480)), filter_requirements),
    Tool("filter-attenuation", "Measured filter attenuation", "Compare", 2, 3,
         "Compare total light attenuation with mean wavelength transmission over the same measured support.",
         (BANDS, number("floor", "Minimum open signal", .00008, .00001), number("count", "Grid samples", 400),
          boolean("normalize", "Normalize plot amplitudes", False)), filter_contrast),
    Tool("parallel-strips", "Parallel detector strips", "Strip", 1, 1,
         "Use two to six strips from the same detector image and compare their raw traces.",
         (structured("lanes", "Individual strip positions", DEFAULT), number("resample", "Grid samples", 360),
          select("normalize", "Display scale", "none", "none", "peak")), parallel),
    Tool("camera-projection", "Camera patch projection", "Camera", 1, 1,
         "Project light from a selected native patch onto detector columns or rows.",
         (structured("window", "Image window", [[.20, .31], [.80, .72]], "crop"),
          select("direction", "Projection axis", "horizontal", "horizontal", "vertical"),
          number("samples", "Projection intervals", 48), number("threshold", "Bright threshold", .07, .005)), projection),
    Tool("rgb-traces", "Detector channels", "Spectrum", 1, 1,
         "Read the three camera channels with independent support and common fractions.",
         (boolean("show_shares", "Show fractions", False),
          number("signal_floor", "Minimum total channel signal", .002, .001)), channel_balance),
    Tool("channel-bands", "RGB band energy", "Spectrum", 1, 1,
         "Integrate the recorded detector channels over each chosen band.",
         (BANDS, boolean("shared", "Require all three measured channels", True)), color_bands),
    Tool("channel-guard", "Channel clipping", "Camera", 1, 1,
         "Locate clipping runs and signal loss separately in the detector channels.",
         (number("intervals", "Number of intervals", 12),), channel_guard),
    Tool("strip-noise", "Spatial strip variability", "Strip", 1, 1,
         "Check within-column variability alongside the signal and native supported count.",
         (BANDS, number("minimum", "Minimum native pixels", 3)), noise_windows),
    Tool("sideband-check", "Background strip agreement", "Strip", 1, 1,
         "Compare the separately read backgrounds above and below the saved camera strip.",
         (number("limit", "Difference guard", .026, .002), number("minimum", "Minimum sideband pixels", 2),
          boolean("normalize", "Divide by exposure time", False)), sidebands),
    Tool("band-shape", "Broad band shape", "Spectrum", 1, 1,
         "Summarize the band crest, measured extent and positive-light area percentiles.",
         (BANDS, number("tail", "Area tail", .1, .01)), envelopes),
    Tool("edge-scan", "Spectral transitions", "Spectrum", 1, 1,
         "Locate rising or falling spans in native measured signal.",
         (structured("window", "Axis interval", None),
          number("gradient", "Minimum slope", .015, .001),
          select("direction", "Direction", "both", "both", "rising", "falling")), slope_edges),
    Tool("continuum-flatness", "Continuum uniformity", "Spectrum", 1, 1,
         "Check local signal variation and measured width across a continuum.",
         (structured("window", "Axis interval", None),
          number("pieces", "Number of bands", 8), number("limit", "Allowed span / median", .22, .01)), flatness),
    Tool("dark-field", "Dark camera frame", "Camera", 1, 1,
         "Inspect median dark current and unusually active source pixels.",
         (structured("window", "Image window", [[.18, .2], [.84, .78]], "crop"),
          number("across", "Tiles across", 5), number("down", "Tiles down", 3),
          number("outlier", "Median deviations for outliers", 7, .25)), dark_field),
    Tool("trace-placement", "Placement by wavelength", "Strip", 1, 1,
         "Read cross-strip signal through each selected spectral band.",
         (BANDS, number("samples", "Maximum columns per band", 100),
          number("offset", "Scan either side of the center", 16),
          number("width", "Smoothing half-height", 1)), trace_placement),
    Tool("exposure-response", "Exposure response", "Compare", 2, 4,
         "Estimate the camera's linear response over common bands and multiple exposures.",
         (BANDS, number("count", "Grid samples", 320),
          number("support", "Minimum common band coverage", .4, .05)), exposure_response),
    Tool("shared-windows", "Shared band readings", "Compare", 2, 4,
         "See how comparing only common support changes an area measurement.",
         (BANDS, number("count", "Grid samples", 440), RATE), band_comparison),
)}
