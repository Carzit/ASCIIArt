#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ascii_video.py
==============

Video to ASCII animation (frame-by-frame) using the image converter from ascii_image.

- Read frames with OpenCV, convert to PIL, map to ASCII via AsciiImageConverter
- Terminal playback supports ANSI/Truecolor
- Export MP4/GIF renders colored text by parsing ANSI SGR (16-color and truecolor)

Author: Carzit
"""

from __future__ import annotations
import argparse
import os
import re
import sys
import time
import shutil
from typing import Optional, Iterable, Tuple, List

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Use config and converter from ascii_image
from ascii_image import AsciiImageConfig, AsciiImageConverter


class AsciiVideoConverter:
    """
    Convert video to ASCII art frames (generator).
    """

    def __init__(self, config: AsciiImageConfig):
        self.cfg = config
        self.converter = AsciiImageConverter(config)

    def iter_ascii_frames(self, video_path: str) -> Tuple[Iterable[str], float]:
        """
        返回一个生成器（逐帧输出 ASCII 文本）和视频原始 FPS。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        def _gen():
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    yield self.converter.convert(img)
            finally:
                cap.release()

        return _gen(), float(video_fps)


class AsciiVideoRenderer:
    """
    渲染/保存 ASCII 视频帧：终端播放、保存 txt 序列、导出 MP4/GIF。
    """

    ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def __init__(self):
        pass

    @staticmethod
    def play_terminal(frames: Iterable[str], fps: float) -> None:
        delay = 1.0 / max(1e-3, fps)
        for ascii_text in frames:
            try:
                shutil.get_terminal_size()
            except Exception:
                pass
            os.system("cls" if os.name == "nt" else "clear")
            print(ascii_text)
            time.sleep(delay)

    @staticmethod
    def save_txt_frames(frames: Iterable[str], output_dir: str) -> int:
        os.makedirs(output_dir, exist_ok=True)
        count = 0
        for idx, ascii_text in enumerate(frames):
            frame_path = os.path.join(output_dir, f"frame_{idx:05d}.txt")
            with open(frame_path, "w", encoding="utf-8") as f:
                f.write(ascii_text)
            count += 1
        return count

    def _ansi_16_index_to_rgb(self, idx: int) -> Tuple[int, int, int]:
        """Map ANSI 16-color index [0-15] to RGB. Typical xterm palette."""
        base = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
            (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
            (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
            (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
        ]
        idx = max(0, min(15, idx))
        return base[idx]

    def _parse_ansi_line(self, line: str, default_fg: Tuple[int, int, int]) -> List[Tuple[str, Tuple[int, int, int]]]:
        """Parse a single line with ANSI SGR to spans of (text, rgb)."""
        spans: List[Tuple[str, Tuple[int, int, int]]] = []
        cur_color = default_fg
        buf: List[str] = []

        i = 0
        length = len(line)
        while i < length:
            ch = line[i]
            if ch == "\x1b" and i + 1 < length and line[i+1] == "[":
                # flush buffer
                if buf:
                    spans.append(("".join(buf), cur_color))
                    buf = []
                # parse CSI sequence until 'm' or end
                j = i + 2
                params: List[int] = []
                num = ""
                mode = None
                while j < length:
                    c = line[j]
                    if c.isdigit():
                        num += c
                    elif c == ";":
                        if num:
                            params.append(int(num))
                            num = ""
                    elif c == "m":
                        if num:
                            params.append(int(num))
                            num = ""
                        mode = "m"
                        j += 1
                        break
                    else:
                        # unsupported; break
                        j += 1
                        break
                    j += 1
                # apply SGR if found
                if mode == "m":
                    if not params:
                        params = [0]
                    k = 0
                    while k < len(params):
                        p = params[k]
                        if p == 0:
                            cur_color = default_fg
                        elif 30 <= p <= 37:
                            cur_color = self._ansi_16_index_to_rgb(p - 30)
                        elif 90 <= p <= 97:
                            cur_color = self._ansi_16_index_to_rgb(8 + (p - 90))
                        elif p == 38 and k + 4 < len(params) and params[k+1] == 2:
                            r, g, b = params[k+2], params[k+3], params[k+4]
                            cur_color = (int(r) & 255, int(g) & 255, int(b) & 255)
                            k += 4
                        elif p == 39:
                            cur_color = default_fg
                        # ignore other attributes
                        k += 1
                    i = j
                    continue
                # if not proper 'm', skip esc
                i = j
                continue
            else:
                buf.append(ch)
                i += 1

        if buf:
            spans.append(("".join(buf), cur_color))
        return spans

    def _ascii_to_image(self, ascii_text: str, font_path: Optional[str] = None,
                       font_size: int = 14, fg_hex: Optional[str] = None,
                       bg_hex: Optional[str] = None) -> Image.Image:
        """
        将 ASCII 文本渲染成 PIL 图像，保留 ANSI/Truecolor 前景色。
        """
        # 默认颜色（用于没有 ANSI 的部分）
        def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
            h = h.strip()
            if h.startswith("#"):
                h = h[1:]
            if len(h) == 3:
                h = "".join([c*2 for c in h])
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))  # type: ignore

        default_fg_rgb = _hex_to_rgb(fg_hex) if fg_hex else (255, 255, 255)
        bg_color = bg_hex if bg_hex else "#000000"
        
        # 尝试加载字体
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                # 尝试系统默认等宽字体
                try:
                    # Windows
                    font = ImageFont.truetype("consola.ttf", font_size)
                except OSError:
                    try:
                        # macOS
                        font = ImageFont.truetype("Monaco.ttf", font_size)
                    except OSError:
                        try:
                            # Linux
                            font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
                        except OSError:
                            # 默认字体
                            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # 分割文本行（保留 ANSI 以便逐行解析）
        raw_lines = ascii_text.split('\n')
        lines_spans: List[List[Tuple[str, Tuple[int, int, int]]]] = []
        for line in raw_lines:
            lines_spans.append(self._parse_ansi_line(line, default_fg_rgb))
        if not lines_spans:
            lines_spans = [[("", default_fg_rgb)]]
        
        # 计算图像尺寸
        max_width = 0
        line_height = 0
        # 估计等宽字符宽度
        bbox_M = font.getbbox("M")
        char_w = (bbox_M[2] - bbox_M[0]) if bbox_M else font_size
        for spans in lines_spans:
            text_len = sum(len(text) for text, _ in spans)
            width = text_len * char_w
            bbox_line = font.getbbox("Mg")
            height = (bbox_line[3] - bbox_line[1]) if bbox_line else font_size
            max_width = max(max_width, width)
            line_height = max(line_height, height)
        
        # 如果没有内容，设置最小尺寸
        if max_width == 0:
            max_width = font_size * 10
        if line_height == 0:
            line_height = font_size
            
        # 添加边距
        margin = 10
        img_width = max_width + 2 * margin
        img_height = line_height * len(lines_spans) + 2 * margin
        
        # 创建图像
        img = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # 绘制文本（按字符着色）
        y_offset = margin
        for spans in lines_spans:
            x_offset = margin
            for text, rgb in spans:
                for ch in text:
                    draw.text((x_offset, y_offset), ch, font=font, fill=tuple(rgb))
                    x_offset += char_w
            y_offset += line_height
        
        return img

    def save_mp4(self, frames: Iterable[str], fps: float, output_path: str,
                 font_path: Optional[str] = None, font_size: int = 14,
                 fg_hex: Optional[str] = None, bg_hex: Optional[str] = None) -> int:
        """
        将 ASCII 帧序列导出为 MP4 视频文件
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) is required for MP4 export")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        writer = None
        frame_count = 0
        
        try:
            for ascii_text in tqdm(frames):
                # 将 ASCII 转换为图像
                pil_img = self._ascii_to_image(ascii_text, font_path, font_size, fg_hex, bg_hex)
                
                # 转换为 OpenCV 格式
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # 初始化视频写入器（在第一帧时）
                if writer is None:
                    height, width = cv_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    if not writer.isOpened():
                        raise IOError(f"Failed to open video writer for {output_path}")
                
                # 写入帧
                writer.write(cv_img)
                frame_count += 1
                
        finally:
            if writer is not None:
                writer.release()
        
        return frame_count

    def save_gif(self, frames: Iterable[str], fps: float, output_path: str,
                 font_path: Optional[str] = None, font_size: int = 14,
                 fg_hex: Optional[str] = None, bg_hex: Optional[str] = None) -> int:
        """
        将 ASCII 帧序列导出为 GIF 动画文件
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        pil_images = []
        frame_count = 0
        
        # 将所有 ASCII 帧转换为 PIL 图像
        for ascii_text in tqdm(frames):
            pil_img = self._ascii_to_image(ascii_text, font_path, font_size, fg_hex, bg_hex)
            pil_images.append(pil_img)
            frame_count += 1
        
        if not pil_images:
            raise ValueError("No frames to export")
        
        # 计算每帧持续时间（毫秒）
        duration = int(1000 / max(1, fps))
        
        # 保存为 GIF
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0,  # 无限循环
            optimize=True
        )
        
        return frame_count


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Video to ASCII Art animation")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output-dir", help="输出帧目录（每帧一个 .txt）")
    parser.add_argument("-m", "--save-mp4", help="导出为 MP4 文件路径，例如 out.mp4")
    parser.add_argument("-g", "--save-gif", help="导出为 GIF 文件路径，例如 out.gif")
    parser.add_argument("-n", "--no-play", action="store_true", help="不在终端播放动画")
    parser.add_argument("--fps", type=float, help="播放/导出的帧率（默认跟随视频原始帧率）")

    # Reuse image options from AsciiConfig
    parser.add_argument("--width", type=int, help="输出宽度（字符数）")
    parser.add_argument("--height", type=int, help="输出高度（字符数）")
    parser.add_argument("--scale", type=float, help="缩放比例")
    parser.add_argument("--charset", type=str, help="字符集名称或自定义字符串")
    parser.add_argument("--color", choices=["none", "ansi", "truecolor", "html"], help="颜色模式")
    parser.add_argument("--invert", action="store_true", help="反相")
    parser.add_argument("--gamma", type=float, help="Gamma 矫正")
    parser.add_argument("--contrast", type=float, help="对比度")
    parser.add_argument("--char-aspect", type=float, help="字符宽高比修正")
    parser.add_argument("--dither", action="store_true", help="字符映射与 ANSI 亮度抖动")
    # Rendering options for video export
    parser.add_argument("--font", help="TTF 字体路径（用于导出 MP4/GIF 渲染）")
    parser.add_argument("--font-size", type=int, default=14, help="字体大小（用于导出 MP4/GIF 渲染）")
    parser.add_argument("--fg", help="前景色，十六进制，如 #FFFFFF")
    parser.add_argument("--bg", help="背景色，十六进制，如 #000000")
    return parser.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)

    cfg = AsciiImageConfig()
    # override with CLI args
    cli_overrides = {
        "width": args.width,
        "height": args.height,
        "scale": args.scale,
        "charset": args.charset,
        "color": args.color,
        "invert": args.invert,
        "gamma": args.gamma,
        "contrast": args.contrast,
        "char_aspect": args.char_aspect,
        "dither": args.dither,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            setattr(cfg, key, val)

    # 构造生成器
    video_converter = AsciiVideoConverter(cfg)
    frames_iter, src_fps = video_converter.iter_ascii_frames(args.input)
    use_fps = float(args.fps) if args.fps is not None else float(src_fps)
    if use_fps <= 0:
        use_fps = 24.0

    # 根据需求进行渲染/保存
    renderer = AsciiVideoRenderer()

    # 如果需要多路输出（播放 + 保存），我们必须将 frames materialize到列表
    need_materialize = (args.output_dir is not None) or (args.save_mp4 is not None) or (args.save_gif is not None)
    if need_materialize:
        frames_cache = list(frames_iter)
    else:
        frames_cache = None

    if not args.no_play:
        renderer.play_terminal(frames_cache if frames_cache is not None else frames_iter, use_fps)

    if args.output_dir:
        count = renderer.save_txt_frames(frames_cache if frames_cache is not None else frames_iter, args.output_dir)
        print(f"Saved {count} ASCII frames to {args.output_dir}")

    if args.save_mp4:
        # Lazy import to avoid hard dependency if用户只播放
        import numpy as np  # noqa: F401
        count = renderer.save_mp4(
            frames_cache if frames_cache is not None else frames_iter,
            use_fps,
            args.save_mp4,
            font_path=args.font,
            font_size=args.font_size,
            fg_hex=args.fg,
            bg_hex=args.bg,
        )
        print(f"Exported MP4 with {count} frames to {args.save_mp4}")

    if args.save_gif:
        count = renderer.save_gif(
            frames_cache if frames_cache is not None else frames_iter,
            use_fps,
            args.save_gif,
            font_path=args.font,
            font_size=args.font_size,
            fg_hex=args.fg,
            bg_hex=args.bg,
        )
        print(f"Exported GIF with {count} frames to {args.save_gif}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))