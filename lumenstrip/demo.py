"""Create clearly marked, repeatable synthetic camera observations."""

from __future__ import annotations

import numpy as np

from .calibration import Fit
from .camera import encode_png, transfer
from .contracts import Capture, Strip, signature

WIDTH = 780
HEIGHT = 256
LEFT_NM = 390.
RIGHT_NM = 715.


def crease(x, center, width):
    return np.exp(-.5 * ((x - center) / width) ** 2)


def lamp(x):
    rise = .12 + .21 * crease(x, 553, 100)
    centers = (436., 486., 546., 578., 612., 658.)
    strength = (.25, .18, .44, .17, .36, .28)
    widths = (2.1, 3.5, 2.4, 3., 3.9, 3.)
    lines = np.zeros_like(x)
    for center, peak, width in zip(centers, strength, widths, strict=True):
        lines += peak * crease(x, center, width)
    return rise + lines


def filter_curve(x):
    return .065 + .72 * crease(x, 538., 84) - .15 * crease(x, 574, 7)


def rainbow_weight(wavelength):
    return np.stack((.66 + .34 * crease(wavelength, 639, 78),
                     .60 + .40 * crease(wavelength, 543, 65),
                     .54 + .46 * crease(wavelength, 448, 72)), axis=-1)


def _texture(y, x):
    return np.cos(x / 53) * np.sin(y / 28) * .0009 + .0015 * np.sin(x / 19 + y / 90)


def draw_capture(index, kind, exposure, tilt=0., displacement=0.):
    rng = np.random.default_rng(712 + index)
    y, x = np.indices((HEIGHT, WIDTH))
    u = (x + .5) / WIDTH
    wavelength = LEFT_NM + (RIGHT_NM - LEFT_NM) * u + displacement
    integrand = lamp(wavelength)
    centers = 130. + tilt * (u - .5) + 1.5 * np.cos(u * 3)
    spread = np.exp(-.5 * ((y - centers) / 5.2) ** 2)
    if kind == "gel":
        integrand *= filter_curve(wavelength)
    if kind == "dark":
        integrand[:] = 0.
    if kind == "line lamp":
        integrand = integrand - .7 * (.12 + .21 * crease(wavelength, 553, 100))
    exposure_scale = exposure / 30.
    values = .009 + _texture(y, x) + spread * integrand * exposure_scale
    if kind == "repeat":
        values += .018 * spread * crease(wavelength, 610, 9)
        values -= .025 * spread * crease(wavelength, 451, 14)
    if kind == "dark":
        spots = ((123, 450), (110, 250), (96, 510), (157, 650))
        for a, b in spots:
            values[a, b] += .27
    noise = rng.normal(0., .0015 + .0025 * np.sqrt(np.maximum(values, 0)))
    colour = np.clip(values + noise, 0., None)[..., None] * rainbow_weight(wavelength)
    codes = transfer(colour)
    alpha = np.full((HEIGHT, WIDTH), 255, np.float32)
    if kind == "repeat":
        alpha[111:152, 529:551] = 0
    alpha[:16, :17] = 0
    alpha[236:, 752:] = 0
    return encode_png(np.dstack((np.rint(codes * 255), alpha)).astype(np.uint8))


def _axis(title, width):
    physical = [(436, "Violet line"), (486, "Blue line"), (546, "Green line"), (658, "Far red line")]
    anchors = []
    for wavelength, line in physical:
        position = (wavelength - LEFT_NM) / (RIGHT_NM - LEFT_NM)
        anchors.append({"column": position, "nm": wavelength, "name": line})
    return Fit.parse({"title": title, "method": "linear", "limit": "frame", "anchors": anchors})


def seed(lab):
    examples = [
        ("Amber gel", "sample", "gel", 30., 5., 0.),
        ("Open lamp", "open-reference", "lamp", 30., -7., 0.),
        ("Dark frame", "dark-reference", "dark", 30., 0., 0.),
        ("Repeat lamp", "repeat", "repeat", 36., 9., 2.1),
        ("Reference lines", "calibration", "line lamp", 25., -3., 0.),
        ("Short exposure", "repeat", "lamp", 11., -7., 0.),
    ]
    station = lab.create("Amber gel bench", "Synthetic tabletop camera spectra with traceable landmarks and one missing patch.", True)
    results = []
    for i, (title, role, mode, exposure, slope, dx) in enumerate(examples):
        meta = Capture.parse({"title": title, "role": role, "synthetic": True, "exposure_ms": exposure,
                              "notes": "Generated camera view, for checking the workbench."})
        image = draw_capture(i, mode, exposure, slope, dx)
        source = lab.ingest(station["id"], image, title.lower().replace(" ", "-") + ".png", meta)
        settings = Strip.default().public()
        settings["endpoints"] = [[.045, (130. + slope * -.455) / HEIGHT],
                                 [.955, (130. + slope * .455) / HEIGHT]]
        source = lab.set_strip(source["id"], source["revision"], Strip.parse(settings))
        fit = _axis("Reference-line ruler", WIDTH)
        key = signature({"image": source["image"]["sha256"], "revision": source["revision"], "fit": fit.public()})
        lab.add_fit(source["id"], source["revision"], fit, key)
        results.append(lab.read_source(source["id"]))
    return {"station": lab.station(station["id"]), "sources": results}
