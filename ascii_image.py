#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ascii_image.py
==============

统一风格的图片转 ASCII 工具模块（基于 Pillow）

- 支持色彩模式：none | ansi(16色) | truecolor(24位) | html
- 支持字符集预设与自定义，伽马与对比度调整，反相
- 支持字符纵横比补偿 `char_aspect`，缩放策略：width/height/scale
- HTML 输出：使用 <pre> + <span style="color: rgb(...)">

依赖：Pillow
"""

from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

try:
    from PIL import Image, ImageOps, ImageEnhance
except Exception as exc:  # pragma: no cover - 运行环境依赖
    print("This module requires Pillow. Install it with: pip install pillow", file=sys.stderr)
    raise


# ------------------------ Character Sets ------------------------
CHARSETS = {
    "sparse": " .:-=+*#%@",
    "standard": " .'`\"^,,:;Il!i><~+_-?][}{1)(|\\/*tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "dense": "@%#*+=-:. ",
    "blocks": " ▁▂▃▄▅▆▇█",
    "binary": " .",
}


# ------------------------ Utilities (Color + Luma) ------------------------
# 8x8 Bayer threshold matrix (0..63)
BAYER_8x8: List[List[int]] = [
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21],
]

def _clamp(value: float, lo: float = 0.0, hi: float = 255.0) -> float:
    """Clamp a value into [lo, hi]."""
    return max(lo, min(hi, value))


def _srgb_to_linear(c: float) -> float:
    """Convert sRGB [0-255] to linear [0-1]."""
    c = c / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> int:
    """Convert linear [0-1] to sRGB [0-255] (rounded and clamped)."""
    if c <= 0.0031308:
        v = 12.92 * c
    else:
        v = 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return int(_clamp(round(v * 255)))


def _luminance(rgb: Tuple[int, int, int]) -> float:
    """Relative luminance using sRGB coefficients (0-1)."""
    r, g, b = rgb
    R = _srgb_to_linear(r)
    G = _srgb_to_linear(g)
    B = _srgb_to_linear(b)
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def _rgb_to_ansi_16(r: int, g: int, b: int) -> int:
    """Approximate an RGB color to ANSI 16-color palette index [0-15]."""
    L = _luminance((r, g, b))
    bright = L > 0.5
    idx = 0
    if r > 128:
        idx |= 1
    if g > 128:
        idx |= 2
    if b > 128:
        idx |= 4
    if idx == 0 and L > 0.25:
        idx = 7  # white-ish for mid luma blacks
    return idx + (8 if bright else 0)


def _ansi_16_seq(fg_idx: int) -> str:
    """Build ANSI 16-color foreground sequence for index [0-15]."""
    if fg_idx < 8:
        return f"\x1b[{30 + fg_idx}m"
    return f"\x1b[{90 + (fg_idx - 8)}m"


def _ansi_truecolor_seq(r: int, g: int, b: int) -> str:
    """Build ANSI truecolor (24-bit) foreground sequence."""
    return f"\x1b[38;2;{r};{g};{b}m"


ANSI_RESET = "\x1b[0m"


def _resolve_charset(name_or_literal: str) -> str:
    """Resolve charset name to literal string; fallback to given literal."""
    if name_or_literal in CHARSETS:
        return CHARSETS[name_or_literal]
    if not any(ch.strip() for ch in name_or_literal):
        raise ValueError("Custom charset must contain at least one non-space character.")
    return name_or_literal


# ------------------------ Config ------------------------
@dataclass
class AsciiImageConfig:
    """Configuration for image-to-ASCII conversion."""

    width: Optional[int] = None
    height: Optional[int] = None
    scale: Optional[float] = None
    charset: str = "standard"
    color: str = "none"  # none | ansi | truecolor | html
    bg: str = "black"     # affects only HTML background theme
    invert: bool = False
    gamma: float = 1.0
    contrast: float = 1.0
    char_aspect: float = 0.5  # character cell height/width ratio compensation
    dither: bool = False

    def to_dict(self) -> dict:
        """Return config as dict."""
        return asdict(self)


# ------------------------ Core Converter ------------------------
class AsciiImageConverter:
    """Convert PIL images to ASCII text with optional coloring."""

    def __init__(self, config: AsciiImageConfig):
        self.cfg = config

    # ---- loading & preprocessing ----
    def load_image(self, path: str) -> Image.Image:
        """Load image and convert to RGBA for uniform processing."""
        img = Image.open(path)
        try:
            img.seek(0)  # use first frame if animated
        except Exception:
            pass
        return img.convert("RGBA")

    def _preprocess(self, img: Image.Image) -> Image.Image:
        """Apply contrast and inversion; keep alpha channel intact."""
        if self.cfg.contrast and abs(self.cfg.contrast - 1.0) > 1e-6:
            img = ImageEnhance.Contrast(img).enhance(self.cfg.contrast)
        if self.cfg.invert:
            rgb = img.convert("RGB")
            inv = ImageOps.invert(rgb)
            img = Image.merge("RGBA", (*inv.split(), img.split()[-1]))
        return img

    # ---- sizing ----
    def _compute_output_size(self, img: Image.Image) -> Tuple[int, int]:
        """Compute output width/height in characters, considering char aspect."""
        w, h = img.size
        effective_h = max(1, int(round(h * self.cfg.char_aspect)))

        if self.cfg.scale is not None:
            tw = max(1, int(round(w * self.cfg.scale)))
            th = max(1, int(round(effective_h * self.cfg.scale)))
            return tw, th
        if self.cfg.width and self.cfg.height:
            return max(1, self.cfg.width), max(1, self.cfg.height)
        if self.cfg.width:
            tw = max(1, self.cfg.width)
            th = max(1, int(round(effective_h * (tw / float(w)))))
            return tw, th
        if self.cfg.height:
            th = max(1, self.cfg.height)
            tw = max(1, int(round(w * (th / float(effective_h)))))
            return tw, th
        # default width=100
        tw = min(100, w)
        th = max(1, int(round(effective_h * (tw / float(w)))))
        return tw, th

    def _resize_image(self, img: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """High-quality resize to target character grid size."""
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        return img.resize(size, resample).convert("RGB")

    # ---- mapping ----
    def _map_pixel_to_char(self, rgb: Tuple[int, int, int], charset: str, x: int, y: int) -> str:
        """Map one RGB pixel to ASCII character by luminance with optional Bayer dithering."""
        lum = _luminance(rgb)
        if self.cfg.gamma != 1.0:
            lum = max(1e-8, lum) ** (1.0 / self.cfg.gamma)
        # Ordered dithering on luminance before quantization to charset
        if self.cfg.dither:
            # amplitude controls perceived grain; 0.10~0.15 is a good range
            amplitude = 0.12
            threshold = (BAYER_8x8[y % 8][x % 8] / 63.0 - 0.5) * amplitude * 2.0
            lum = min(1.0, max(0.0, lum + threshold))
        idx = int(round(lum * (len(charset) - 1)))
        idx = max(0, min(len(charset) - 1, idx))
        return charset[idx]

    def _rgb_to_ansi_16_with_dither(self, r: int, g: int, b: int, x: int, y: int) -> int:
        """ANSI 16 色量化，使用 Bayer 抖动在亮度阈值处抖动 bright 位。"""
        if not self.cfg.dither:
            return _rgb_to_ansi_16(r, g, b)
        L = _luminance((r, g, b))
        # Slightly larger amplitude than char mapping to better toggle bright bit
        amplitude = 0.15
        threshold = (BAYER_8x8[y % 8][x % 8] / 63.0 - 0.5) * amplitude * 2.0
        Ld = min(1.0, max(0.0, L + threshold))
        bright = Ld > 0.5
        idx = 0
        if r > 128:
            idx |= 1
        if g > 128:
            idx |= 2
        if b > 128:
            idx |= 4
        if idx == 0 and Ld > 0.25:
            idx = 7
        return idx + (8 if bright else 0)

    # ---- conversion ----
    def convert(self, img: Image.Image) -> str:
        """Convert image to ASCII text according to configured color mode."""
        cfg = self.cfg
        img = self._preprocess(img)

        # Background for compositing: dark by default; white if HTML+white bg
        if cfg.color == "html" and cfg.bg.lower() in ("white", "#fff", "#ffffff"):
            bg = (255, 255, 255, 255)
        else:
            bg = (0, 0, 0, 255)

        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bg_img = Image.new("RGBA", img.size, bg)
        img = Image.alpha_composite(bg_img, img)

        out_w, out_h = self._compute_output_size(img)
        resized = self._resize_image(img, (out_w, out_h))

        charset = _resolve_charset(cfg.charset)
        color_mode = (cfg.color or "none").lower()
        if color_mode not in ("none", "ansi", "truecolor", "html"):
            raise ValueError("color must be one of: none, ansi, truecolor, html")

        # HTML mode
        if color_mode == "html":
            return self._to_html(resized, out_w, out_h, charset)

        # Text (terminal) modes
        return self._to_text(resized, out_w, out_h, charset, color_mode)

    # ---- render helpers ----
    def _to_html(self, img: Image.Image, out_w: int, out_h: int, charset: str) -> str:
        """Render ASCII with per-character color to HTML string."""
        bg_css = "black" if self.cfg.bg.lower() == "black" else "white"
        lines: List[str] = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html><head><meta charset='utf-8'><title>ASCII Art</title>")
        # Fixed-size monospace grid; 10px is a practical default for readability
        lines.append(
            "<style>body{margin:0;background:%s;} pre{line-height:1; font: 10px/10px monospace; padding:8px;}</style>"
            % bg_css
        )
        lines.append("</head><body><pre>")

        px = img.load()
        for y in range(out_h):
            row_parts: List[str] = []
            for x in range(out_w):
                r, g, b = px[x, y]
                ch = self._map_pixel_to_char((r, g, b), charset, x, y)
                row_parts.append(f"<span style=\"color: rgb({r},{g},{b})\">{ch}</span>")
            lines.append("".join(row_parts))
        lines.append("</pre></body></html>")
        return "\n".join(lines)

    def _to_text(self, img: Image.Image, out_w: int, out_h: int, charset: str, color_mode: str) -> str:
        """Render ASCII as plain/ANSI/truecolor text for terminal."""
        px = img.load()
        lines: List[str] = []
        for y in range(out_h):
            row_parts: List[str] = []
            last_seq = ""
            for x in range(out_w):
                r, g, b = px[x, y]
                ch = self._map_pixel_to_char((r, g, b), charset, x, y)

                if color_mode == "ansi":
                    idx = self._rgb_to_ansi_16_with_dither(r, g, b, x, y)
                    seq = _ansi_16_seq(idx)
                elif color_mode == "truecolor":
                    seq = _ansi_truecolor_seq(r, g, b)
                else:
                    seq = ""

                if seq != last_seq:
                    row_parts.append(seq)
                    last_seq = seq
                row_parts.append(ch)
            if last_seq:
                row_parts.append(ANSI_RESET)
            lines.append("".join(row_parts))
        return "\n".join(lines)


# ------------------------ Renderer ------------------------
class AsciiImageRenderer:
    """Handle output destinations for ASCII text (console/file)."""

    def __init__(self, ascii_text: str):
        self.text = ascii_text

    def to_console(self) -> None:
        print(self.text)

    def to_file(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(self.text)
        print(f"Saved: {path}")


# ------------------------ CLI ------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert image to ASCII art (unified module)")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", help="Output file path (.txt or .html). Omit to print")

    size = parser.add_argument_group("size controls")
    size.add_argument("--width", type=int, help="Target width (characters)")
    size.add_argument("--height", type=int, help="Target height (characters)")
    size.add_argument("--scale", type=float, help="Uniform scale factor (e.g., 0.25)")
    size.add_argument("--char-aspect", type=float, default=0.5, help="Character height/width ratio compensation")

    visuals = parser.add_argument_group("visuals")
    visuals.add_argument("--charset", type=str, default="standard", help=f"Charset name or custom string. Presets: {', '.join(CHARSETS.keys())}")
    visuals.add_argument("--color", choices=["none", "ansi", "truecolor", "html"], default="none", help="Color mode")
    visuals.add_argument("--bg", type=str, default="black", help="Background for HTML (black/white or CSS color)")
    visuals.add_argument("--invert", action="store_true", help="Invert image before mapping")
    visuals.add_argument("--gamma", type=float, default=1.0, help="Gamma correction (e.g., 0.8/1.2)")
    visuals.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier")
    visuals.add_argument("--dither", action="store_true", help="Reserved flag (no-op)")

    return parser


def parse_args(argv: List[str]) -> argparse.Namespace:
    return _build_arg_parser().parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    cfg = AsciiImageConfig(
        width=args.width,
        height=args.height,
        scale=args.scale,
        charset=args.charset,
        color=args.color,
        bg=args.bg,
        invert=args.invert,
        gamma=args.gamma,
        contrast=args.contrast,
        char_aspect=args.char_aspect,
        dither=args.dither,
    )

    try:
        converter = AsciiImageConverter(cfg)
        img = converter.load_image(args.input)
        ascii_text = converter.convert(img)
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 2

    renderer = AsciiImageRenderer(ascii_text)
    if args.output:
        renderer.to_file(args.output)
    else:
        renderer.to_console()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


