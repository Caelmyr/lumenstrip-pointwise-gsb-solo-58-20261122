"""Decode immutable camera originals and inspect native signal support."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .contracts import Problem, file_label, rounded

MAX_PIXELS = 6_500_000
MAX_BYTES = 24_000_000
FORMATS = {"PNG": "image/png", "TIFF": "image/tiff", "JPEG": "image/jpeg", "WEBP": "image/webp"}
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def linear(values: np.ndarray) -> np.ndarray:
    return np.where(values <= .04045, values / 12.92,
                    ((values + .055) / 1.055) ** 2.4).astype(np.float32)


def transfer(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0, 1)
    return np.where(clipped <= .0031308, 12.92 * clipped,
                    1.055 * clipped ** (1 / 2.4) - .055)


def channel_values(encoded: np.ndarray, channel: str) -> np.ndarray:
    rgb = linear(encoded)
    if channel == "linear-luma":
        return rgb @ LUMA
    if channel == "mean":
        return np.mean(rgb, axis=-1)
    index = ("red", "green", "blue").index(channel)
    return rgb[..., index]


def encode_png(image: Image.Image | np.ndarray) -> bytes:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    output = BytesIO()
    image.save(output, "PNG", compress_level=5)
    return output.getvalue()


def _preview_rgba(rgb, alpha):
    return np.rint(np.dstack((rgb, alpha)) * 255).clip(0, 255).astype(np.uint8)


def resize_limit(image: Image.Image, maximum: int = 1200) -> Image.Image:
    if max(image.size) <= maximum:
        return image
    size = tuple(max(1, round(v * maximum / max(image.size))) for v in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def _to_srgb(source: Image.Image, image: Image.Image) -> tuple[Image.Image, str]:
    icc = source.info.get("icc_profile")
    if not icc:
        return image, "Untyped RGB interpreted as sRGB."
    try:
        source_profile = ImageCms.ImageCmsProfile(BytesIO(icc))
        lab_profile = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(image.convert("RGB"), source_profile, lab_profile,
                                              outputMode="RGB")
        converted.putalpha(image.getchannel("A"))
    except (TypeError, ValueError, OSError, ImageCms.PyCMSError):
        return image, "Unreadable embedded profile. RGB interpreted as sRGB."
    return converted, "Embedded ICC converted to sRGB for display and linear RGB analysis."


def _codes(source: Image.Image) -> tuple[np.ndarray, np.ndarray, int, str]:
    if source.mode in {"I;16", "I;16L", "I;16B", "I;16N"}:
        mono = np.asarray(source, dtype=np.float32) / 65535
        rgb = np.repeat(mono[..., None], 3, axis=-1)
        return rgb, np.ones(mono.shape, np.float32), 16, "Unsigned grayscale codes."
    if source.mode in {"F", "I"}:
        raise Problem("Use an unsigned grayscale 16-bit TIFF/PNG or an ordinary RGB image.")
    image, description = _to_srgb(source, source.convert("RGBA"))
    data = np.asarray(image, dtype=np.float32) / 255
    return data[..., :3], data[..., 3], 8, description


@dataclass(frozen=True)
class Camera:
    name: str
    mime: str
    sha256: str
    byte_count: int
    rgb: np.ndarray
    alpha: np.ndarray
    depth: int
    color: str

    @classmethod
    def read(cls, raw: bytes, name: str) -> Camera:
        if not raw or len(raw) > MAX_BYTES:
            raise Problem("Upload one image smaller than 24 MB.", 413)
        try:
            with Image.open(BytesIO(raw)) as image:
                if image.format not in FORMATS:
                    raise Problem("Use PNG, JPEG, TIFF or WebP camera images.")
                mime = FORMATS[image.format]
                if getattr(image, "n_frames", 1) != 1:
                    raise Problem("Choose a single-frame image.")
                width, height = image.size
                if min(width, height) < 12 or width * height > MAX_PIXELS:
                    raise Problem("Images must be at least 12×12 and no more than 6.5 million pixels.", 413)
                corrected = ImageOps.exif_transpose(image)
                rgb, alpha, depth, color = _codes(corrected)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
            raise Problem("The uploaded camera image could not be decoded.") from err
        return cls(file_label(name), mime, sha256(raw).hexdigest(), len(raw), rgb, alpha, depth, color)

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.alpha.shape
        return width, height

    @property
    def support(self) -> np.ndarray:
        return self.alpha >= .995

    def public(self) -> dict:
        width, height = self.size
        return {"name": self.name, "mime": self.mime, "width": width, "height": height,
                "sha256": self.sha256, "bytes": self.byte_count, "native_depth": self.depth,
                "color_path": self.color,
                "supported_fraction": rounded(np.count_nonzero(self.support) / width / height)}

    def image(self, maximum: int = 1200) -> Image.Image:
        return resize_limit(Image.fromarray(_preview_rgba(self.rgb, self.alpha)), maximum)

    def rgba(self) -> np.ndarray:
        return _preview_rgba(self.rgb, self.alpha)

    def pixel(self, uv: tuple[float, float]) -> dict:
        width, height = self.size
        x = min(width - 1, int(uv[0] * width))
        y = min(height - 1, int(uv[1] * height))
        supported = bool(self.support[y, x])
        encoded = self.rgb[y, x]
        return {"uv": [(x + .5) / width, (y + .5) / height], "pixel": [x, y],
                "alpha": rounded(self.alpha[y, x]), "supported": supported,
                "srgb": [rounded(v) for v in encoded] if supported else None,
                "linear_luma": rounded(linear(encoded) @ LUMA) if supported else None}

    def crop(self, bounds, expand: int = 0) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        width, height = self.size
        x0, y0, x1, y1 = bounds
        x0, y0 = max(0, x0 - expand), max(0, y0 - expand)
        x1, y1 = min(width, x1 + expand), min(height, y1 + expand)
        return self.rgb[y0:y1, x0:x1], self.alpha[y0:y1, x0:x1], (x0, y0)

    def loupe(self, uv, radius: int) -> tuple[Image.Image, dict]:
        sample = self.pixel(uv)
        x, y = sample["pixel"]
        rgb, alpha, offset = self.crop((x, y, x + 1, y + 1), expand=radius)
        image = Image.fromarray(_preview_rgba(rgb, alpha))
        factor = max(4, 260 // max(image.size))
        scaled = image.resize((image.width * factor, image.height * factor), Image.Resampling.NEAREST)
        sample["loupe_offset"] = list(offset)
        sample["loupe_factor"] = factor
        return scaled, sample


def rgb_stats(rgb: np.ndarray) -> dict:
    if len(rgb) == 0:
        return {"count": 0, "minimum": None, "middle": None, "maximum": None, "p10": None, "p90": None}
    q = np.quantile(rgb, [0, .1, .5, .9, 1], axis=0)
    return {"count": len(rgb), "minimum": [rounded(v) for v in q[0]], "p10": [rounded(v) for v in q[1]],
            "middle": [rounded(v) for v in q[2]], "p90": [rounded(v) for v in q[3]],
            "maximum": [rounded(v) for v in q[4]]}


def saturation(rgb: np.ndarray, alpha: np.ndarray, low: float, high: float) -> dict:
    support = alpha >= .995
    supported = int(support.sum())
    clip = support & np.any(rgb >= high, axis=-1)
    dark = support & np.all(rgb <= low, axis=-1)
    return {"supported": supported, "missing": int(support.size - supported),
            "clip": int(clip.sum()), "dark": int(dark.sum()),
            "clip_fraction_of_supported": rounded(clip.sum() / supported) if supported else None,
            "dark_fraction_of_supported": rounded(dark.sum() / supported) if supported else None}


def signal_overlay(camera: Camera, low: float, high: float, maximum: int = 900) -> Image.Image:
    preview = camera.image(maximum)
    channels = np.asarray(preview, dtype=np.float32) / 255
    missing = channels[..., 3] < .995
    clip = np.any(channels[..., :3] >= high, axis=-1)
    dark = np.all(channels[..., :3] <= low, axis=-1)
    merged = np.rint(channels[..., :3] * 212).astype(np.uint8)
    merged[dark] = [46, 116, 201]
    merged[clip] = [240, 73, 72]
    y, x = np.indices(missing.shape)
    merged[missing] = np.where(((x[missing] // 7 + y[missing] // 7) % 2)[:, None],
                               [47, 58, 63], [25, 40, 49])
    return Image.fromarray(merged)


def input_histogram(rgb: np.ndarray, alpha: np.ndarray, bins: int) -> tuple[np.ndarray, list[np.ndarray]]:
    support = alpha >= .995
    values = rgb[support]
    edges = np.linspace(0, 1, bins + 1)
    counts = [np.histogram(values[:, i], bins=edges)[0] for i in range(3)]
    return edges, counts


def native_grid(camera: Camera, columns: int, rows: int, low: float, high: float) -> list[dict]:
    width, height = camera.size
    x_edges = np.linspace(0, width, columns + 1).astype(int)
    y_edges = np.linspace(0, height, rows + 1).astype(int)
    output = []
    for y in range(rows):
        for x in range(columns):
            left, right = x_edges[x:x + 2]
            top, bottom = y_edges[y:y + 2]
            rgb, alpha, _ = camera.crop((left, top, right, bottom))
            stat = saturation(rgb, alpha, low, high)
            output.append({"tile": f"{y + 1}.{x + 1}", "bounds_px": [int(left), int(top), int(right), int(bottom)],
                           "center_uv": [(left + right) / width / 2, (top + bottom) / height / 2], **stat,
                           "middle_rgb": rgb_stats(rgb[alpha >= .995])["middle"]})
    return output
