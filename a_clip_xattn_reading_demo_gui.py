#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive ModeMUX CLIP demo for selectable visual recognition and literal reading.

The same file supports two deployment styles. In a complete Hugging Face repo checkout
it automatically uses the local model and bundled demo_gui_images, so the GUI works
offline. If the script is copied elsewhere (for example from the model card), it loads
the public ModeMUX repository and fetches missing bundled demo assets through the
Hugging Face cache. --model and --asset-repo can override either source independently.

VISUAL evaluates the robust semantic <any> path ("what is depicted?"). TEXT evaluates
the deliberate <text> path with an explicit <null> abstention candidate ("what is
written?"). The editor locks an exact 448x448 RGB canvas before inference. Custom
provocations can be saved as a PNG plus matching JSON prompt file; save failures such
as a read-only script directory are reported in the GUI and can fall back to another
writable folder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import tkinter.font as tkfont

from PIL import Image, ImageDraw, ImageFont, ImageTk

from training_support.font_discovery import find_font_files

try:
    import torch
except Exception as exc:  # GUI can still open far enough to show the error.
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


APP_TITLE = "YOU PIECE OF CLIP! look at it / read it"
CANVAS_SIZE = 448
DEFAULT_REPO = "zer0int/CLIP-ViT-L-14-Cross-Attn-Read-NoRead-ModeMUX"
PROMPT = "a photo of a {candidate}"

HERE = Path(__file__).resolve().parent


def _looks_like_local_modemux_repo(path: Path) -> bool:
    """Return True when *path* looks like a complete local ModeMUX HF checkout."""
    try:
        config_path = path / "config.json"
        has_weights = (path / "model.safetensors").is_file() or (
            path / "model.safetensors.index.json"
        ).is_file()
        if not (config_path.is_file() and has_weights and (path / "modeling_xattn_clip.py").is_file()):
            return False
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return str(config.get("model_type", "")) == "xattn_clip"
    except Exception:
        return False


# Clone-the-repo path: if this script lives inside the complete HF checkout, use
# that checkout directly. Copy-paste path: otherwise use the public repo ID.
LOCAL_REPO = HERE if _looks_like_local_modemux_repo(HERE) else None
DEFAULT_MODEL = os.environ.get(
    "MODEMUX_MODEL",
    os.environ.get("PIECES_MODEL", str(LOCAL_REPO) if LOCAL_REPO else DEFAULT_REPO),
)
DEFAULT_ASSET_REPO = os.environ.get("MODEMUX_ASSET_REPO", DEFAULT_REPO)

DEMO_DIR = HERE / "demo_gui_images"
DEFAULT_IMAGE = DEMO_DIR / "apple_ipod.png"
WRITE_YOUR_OWN_IMAGE = DEMO_DIR / "apple_none.png"

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def center_crop_resize_448(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((CANVAS_SIZE, CANVAS_SIZE), RESAMPLE)


def split_candidates(value: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in str(value).split(","):
        candidate = part.strip()
        if not candidate:
            continue
        key = " ".join(candidate.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def union_candidates(text_candidates: list[str], visual_candidates: list[str]) -> tuple[list[str], dict[str, str]]:
    out: list[str] = []
    source: dict[str, str] = {}
    for label, values in (("text", text_candidates), ("visual", visual_candidates)):
        for candidate in values:
            key = " ".join(candidate.casefold().split())
            if key not in source:
                out.append(candidate)
                source[key] = label
            elif source[key] != label:
                source[key] = "both"
    return out, source



def hf_repo_id_from_reference(reference: str) -> str | None:
    """Return org/repo for an HF reference, or None for a local/path-like value."""
    reference = str(reference).strip()
    if not reference:
        return None
    try:
        if Path(reference).expanduser().exists():
            return None
    except Exception:
        pass
    if re.match(r"^[A-Za-z]:[\\/]", reference):
        return None
    if reference.startswith((".", "/", "\\")):
        return None
    if reference.count("/") == 1 and not any(ch.isspace() for ch in reference):
        return reference
    return None


@lru_cache(maxsize=64)
def resolve_pil_font(size_px: int) -> ImageFont.ImageFont:
    """Resolve a normal OS font without shipping a font file."""
    size_px = max(8, int(size_px))
    preferred = (
        "segoeui.ttf",
        "arial.ttf",
        "calibri.ttf",
        "tahoma.ttf",
        "Arial.ttf",
        "Helvetica.ttc",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
    )
    for path in find_font_files(preferred_names=preferred):
        try:
            return ImageFont.truetype(path, size_px)
        except OSError:
            continue

    # Pillow fallback. Rotation still works; size scaling may be limited.
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:
        return ImageFont.load_default()


class ResultPane(ttk.Frame):
    def __init__(self, master, title: str, subtitle: str):
        super().__init__(master)
        self.app = self.winfo_toplevel()

        ttk.Label(
            self,
            text=title,
            font=self.app.font_title,
        ).pack(anchor="w")
        self.subtitle_label = ttk.Label(
            self,
            text=subtitle,
            font=self.app.font_small,
        )
        self.subtitle_label.pack(anchor="w", pady=(0, self.app.ui_pad_small))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            body,
            width=self.app.result_canvas_width,
            height=self.app.result_canvas_height,
            highlightthickness=0,
            background="#f5f5f5",
        )
        self.scroll = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        self._last_rows: list[dict[str, Any]] = []
        self._last_view_mode = "confidence"
        self._resize_after_id = None
        self._visible_width = max(1, int(self.app.result_canvas_width))
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_canvas_configure(self, event=None):
        if event is not None:
            self._visible_width = max(1, int(event.width))
        if not self._last_rows:
            return
        if self._resize_after_id is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.after(35, self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._resize_after_id = None
        if self._last_rows:
            self.show_results(
                self._last_rows,
                view_mode=self._last_view_mode,
                remember=False,
            )

    def _current_width(self) -> int:
        width = int(getattr(self, "_visible_width", 0) or 0)
        if width <= 1:
            width = int(self.canvas.winfo_width())
        if width <= 1:
            width = int(self.app.result_canvas_width)
        return max(120, width)

    def set_subtitle(self, text: str):
        self.subtitle_label.configure(text=text)

    def clear(self, message: str = "Run the model to see results."):
        self._last_rows = []
        self.canvas.delete("all")
        width = self._current_width()
        self.canvas.create_text(
            self.app.ui_pad,
            self.app.ui_pad,
            anchor="nw",
            text=message,
            fill="#777777",
            width=max(200, width - 2 * self.app.ui_pad),
            font=self.app.font_result,
        )
        self.canvas.configure(scrollregion=(0, 0, width, self.app.result_row_h * 2))

    def show_results(
        self,
        rows: list[dict[str, Any]],
        view_mode: str,
        remember: bool = True,
    ):
        if remember:
            self._last_rows = list(rows)
            self._last_view_mode = view_mode

        self.canvas.delete("all")
        if not rows:
            self.clear("No candidates.")
            return

        width = self._current_width()
        row_h = self.app.result_row_h
        pad_x = self.app.ui_pad
        tag_x = pad_x
        candidate_x = pad_x + self.app.tag_column_width

        value_x = width - self.app.ui_pad
        numeric_gutter = max(76, self.app.value_column_width)
        bar_right_limit = value_x - numeric_gutter

        preferred_bar_w = max(64, min(135, int(width * 0.23)))
        min_bar_w = 28
        min_candidate_w = 54

        available = max(24, bar_right_limit - candidate_x)
        candidate_w = min(
            self.app.candidate_column_width,
            max(min_candidate_w, available - preferred_bar_w),
        )
        bar_x = candidate_x + candidate_w
        bar_w = min(preferred_bar_w, max(min_bar_w, bar_right_limit - bar_x))

        if bar_x + bar_w > bar_right_limit:
            overflow = (bar_x + bar_w) - bar_right_limit
            candidate_w = max(34, candidate_w - overflow)
            bar_x = candidate_x + candidate_w
            bar_w = max(12, bar_right_limit - bar_x)

        bar_h = max(16, int(row_h * 0.36))

        if view_mode == "cosine":
            ordered = sorted(
                rows,
                key=lambda row: float(row.get("cosine", float("-inf"))),
                reverse=True,
            )
            max_abs = max(
                0.05,
                max(abs(float(row.get("cosine", 0.0))) for row in ordered),
            )
        else:
            ordered = sorted(
                rows,
                key=lambda row: float(row.get("logit", float("-inf"))),
                reverse=True,
            )
            max_p = max(float(row.get("prob", 0.0)) for row in ordered) or 1.0

        for i, row in enumerate(ordered):
            y = i * row_h + self.app.ui_pad_small
            candidate = str(row["candidate"])
            avg_char_px = max(6.0, self.app.font_result[1] * 0.58)
            max_label_chars = max(
                5,
                min(20, int(candidate_w / avg_char_px) - 1),
            )
            display_candidate = (
                candidate
                if len(candidate) <= max_label_chars
                else candidate[: max(1, max_label_chars - 1)] + "…"
            )
            origin = str(row.get("origin", ""))

            if row.get("is_null"):
                tag = "NULL"
                bar_fill = "#8b8b8b"
            elif origin == "text":
                tag = "T"
                bar_fill = "#8056c2"
            elif origin == "visual":
                tag = "V"
                bar_fill = "#3478c7"
            else:
                tag = "T/V"
                bar_fill = "#357e68"

            if i == 0:
                self.canvas.create_rectangle(
                    3,
                    y - 4,
                    width - 10,
                    y + row_h - 8,
                    outline="#b8b8b8",
                    width=1,
                )

            text_y = y + bar_h / 2 + 2
            self.canvas.create_text(
                tag_x,
                text_y,
                anchor="w",
                text=tag,
                font=self.app.font_result_tag,
                fill="#666666",
            )
            self.canvas.create_text(
                candidate_x,
                text_y,
                anchor="w",
                text=display_candidate,
                font=self.app.font_result_bold if i == 0 else self.app.font_result,
                width=max(30, candidate_w - self.app.ui_pad_small),
            )

            self.canvas.create_rectangle(
                bar_x,
                y,
                bar_x + bar_w,
                y + bar_h,
                fill="#e6e6e6",
                outline="",
            )

            if view_mode == "cosine":
                cosine = float(row.get("cosine", 0.0))
                center = bar_x + bar_w / 2
                self.canvas.create_line(
                    center,
                    y - 1,
                    center,
                    y + bar_h + 1,
                    fill="#a9a9a9",
                )
                extent = (cosine / max_abs) * (bar_w / 2)
                x0, x1 = sorted((center, center + extent))
                self.canvas.create_rectangle(
                    x0,
                    y,
                    x1,
                    y + bar_h,
                    fill=bar_fill,
                    outline="",
                )
                value_text = f"{cosine:+.3f}"
            else:
                prob = float(row.get("prob", 0.0))
                frac = prob / max_p

                if self.app.result_bar_scale == "log":
                    if prob <= 0.0 or max_p <= 0.0:
                        display_frac = 0.0
                    else:
                        floor = 1e-6
                        relative = max(floor, min(1.0, prob / max_p))
                        display_frac = (
                            math.log10(relative / floor)
                            / math.log10(1.0 / floor)
                        )
                        display_frac = max(0.0, min(1.0, display_frac))
                else:
                    display_frac = frac

                self.canvas.create_rectangle(
                    bar_x,
                    y,
                    bar_x + bar_w * display_frac,
                    y + bar_h,
                    fill=bar_fill,
                    outline="",
                )
                value_text = f"{100.0 * prob:4.1f}%"

            self.canvas.create_text(
                value_x,
                text_y,
                anchor="e",
                text=value_text,
                font=self.app.font_result_value,
            )

        total_h = len(ordered) * row_h + self.app.ui_pad
        self.canvas.configure(scrollregion=(0, 0, width, total_h))


class ImageEditor:
    def __init__(self, app: "PiecesDemoApp"):
        self.app = app
        self.base_image: Image.Image | None = None
        self.actions: list[dict[str, Any]] = []

        self.tool: str | None = None
        self.selected_text_index: int | None = None
        self.selected_bbox: tuple[int, int, int, int] | None = None
        self.dragging_text = False
        self.drag_last: tuple[int, int] | None = None

        self.pen_preview: list[tuple[int, int]] | None = None
        self.circle_start: tuple[int, int] | None = None
        self.circle_preview: tuple[int, int, int] | None = None

    def set_base(self, image: Image.Image):
        self.base_image = center_crop_resize_448(image)
        self.actions.clear()
        self.selected_text_index = None
        self.selected_bbox = None
        self.pen_preview = None
        self.circle_start = None
        self.circle_preview = None
        self.app.locked_image = self.base_image.copy()
        self.render()

    def set_tool(self, tool: str | None):
        if self.selected_text_index is not None and tool != "text":
            self.commit_selected_text()
        self.tool = tool
        self.app.set_status(
            {
                "text": "Text tool: type a word, then click the image to place it.",
                "pen": "Pen tool: draw with the mouse. Each mouse-up is one undo step.",
                "circle": "Red ellipse tool.",
            }.get(tool, "Editor ready.")
        )
        if tool == "text":
            self.app.overlay_word_entry.focus_set()

    def reset_blank(self):
        self.actions.clear()
        self.selected_text_index = None
        self.selected_bbox = None
        self.pen_preview = None
        self.circle_preview = None
        self.render()
        self.app.mark_image_dirty("Edits reset.")

    def undo(self):
        if self.selected_text_index is not None:
            idx = self.selected_text_index
            self.selected_text_index = None
            self.selected_bbox = None
            if 0 <= idx < len(self.actions):
                self.actions.pop(idx)
        elif self.actions:
            self.actions.pop()
        self.render()
        self.app.mark_image_dirty("Undo.")

    def _text_bitmap(self, action: dict[str, Any]):
        text = str(action.get("text", "")).strip()
        if not text:
            return None, None
        font = resolve_pil_font(int(action.get("size", 48)))

        dummy = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), text, font=font)
        pad = 8
        w = max(1, bbox[2] - bbox[0] + 2 * pad)
        h = max(1, bbox[3] - bbox[1] + 2 * pad)

        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.text(
            (pad - bbox[0], pad - bbox[1]),
            text,
            fill=(0, 0, 0, 255),
            font=font,
        )
        angle = float(action.get("angle", 0))
        rotated = layer.rotate(angle, expand=True, resample=BICUBIC)
        return rotated, rotated.getbbox()

    def rendered_image(self, show_selection: bool = False) -> Image.Image:
        if self.base_image is None:
            return Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")

        img = self.base_image.copy().convert("RGBA")
        self.selected_bbox = None

        for idx, action in enumerate(self.actions):
            kind = action["kind"]

            if kind == "pen":
                draw = ImageDraw.Draw(img)
                pts = action["points"]
                if len(pts) == 1:
                    x, y = pts[0]
                    r = max(1, int(action.get("width", 4)) // 2)
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=(0, 0, 0, 255))
                else:
                    draw.line(
                        pts,
                        fill=(0, 0, 0, 255),
                        width=int(action.get("width", 4)),
                        joint="curve",
                    )

            elif kind == "circle":
                draw = ImageDraw.Draw(img)
                draw.ellipse(
                    tuple(action["bbox"]),
                    outline=(255, 0, 0, 255),
                    width=int(action.get("width", 4)),
                )

            elif kind == "text":
                layer, _ = self._text_bitmap(action)
                if layer is None:
                    continue
                x = int(action["x"] - layer.width / 2)
                y = int(action["y"] - layer.height / 2)
                img.alpha_composite(layer, dest=(x, y))
                if idx == self.selected_text_index:
                    self.selected_bbox = (x, y, x + layer.width, y + layer.height)

        # Live pen preview.
        if self.pen_preview:
            draw = ImageDraw.Draw(img)
            if len(self.pen_preview) > 1:
                draw.line(
                    self.pen_preview,
                    fill=(0, 0, 0, 255),
                    width=4,
                    joint="curve",
                )

        # Live red-ellipse preview.
        if self.circle_preview is not None:
            draw = ImageDraw.Draw(img)
            draw.ellipse(
                tuple(self.circle_preview),
                outline=(255, 0, 0, 255),
                width=4,
            )

        if show_selection and self.selected_bbox is not None:
            draw = ImageDraw.Draw(img)
            draw.rectangle(self.selected_bbox, outline=(40, 110, 220, 255), width=2)

        return img.convert("RGB")

    def render(self):
        image = self.rendered_image(show_selection=True)
        self.app.show_image(image)

    def commit_selected_text(self):
        self.selected_text_index = None
        self.selected_bbox = None
        self.render()
        self.app.mark_image_dirty("Text placed.")

    def update_selected_text_from_controls(self, *_):
        idx = self.selected_text_index
        if idx is None or not (0 <= idx < len(self.actions)):
            return
        action = self.actions[idx]
        if action.get("kind") != "text":
            return
        action["text"] = self.app.overlay_word_var.get()
        try:
            action["size"] = int(self.app.overlay_size_var.get())
        except Exception:
            pass
        try:
            action["angle"] = float(self.app.overlay_angle_var.get())
        except Exception:
            pass
        self.render()
        self.app.mark_image_dirty("Text edit changed.")

    def _inside_selected(self, x: int, y: int) -> bool:
        if self.selected_bbox is None:
            return False
        x0, y0, x1, y1 = self.selected_bbox
        return x0 <= x <= x1 and y0 <= y <= y1

    def on_press(self, event):
        if not self.app.image_unlocked:
            return
        mapped = self.app.canvas_to_image(event.x, event.y, clamp=False)
        if mapped is None:
            return
        x, y = mapped

        if self.tool == "text":
            if self.selected_text_index is not None:
                if self._inside_selected(x, y):
                    self.dragging_text = True
                    self.drag_last = (x, y)
                    return
                # Click outside commits the current text object.
                self.commit_selected_text()
                return

            word = self.app.overlay_word_var.get().strip()
            if not word:
                self.app.set_status("Type a word in the editor's Word field first.", error=True)
                return
            try:
                size = int(self.app.overlay_size_var.get())
            except Exception:
                size = 48
            try:
                angle = float(self.app.overlay_angle_var.get())
            except Exception:
                angle = 0.0

            self.actions.append(
                {
                    "kind": "text",
                    "text": word,
                    "x": x,
                    "y": y,
                    "size": size,
                    "angle": angle,
                }
            )
            self.selected_text_index = len(self.actions) - 1
            self.render()
            self.app.mark_image_dirty("Text object active; drag it or change size/rotation. Click outside to commit.")

        elif self.tool == "pen":
            self.pen_preview = [(x, y)]
            self.render()

        elif self.tool == "circle":
            self.circle_start = (x, y)
            # Keep a valid Pillow ellipse bbox even before the first drag event.
            self.circle_preview = (x, y, x, y)
            self.render()

    def on_motion(self, event):
        if not self.app.image_unlocked:
            return
        mapped = self.app.canvas_to_image(event.x, event.y, clamp=True)
        if mapped is None:
            return
        x, y = mapped

        if self.dragging_text and self.selected_text_index is not None and self.drag_last is not None:
            lx, ly = self.drag_last
            dx, dy = x - lx, y - ly
            action = self.actions[self.selected_text_index]
            action["x"] += dx
            action["y"] += dy
            self.drag_last = (x, y)
            self.render()
            self.app.mark_image_dirty("Text moved.")

        elif self.tool == "pen" and self.pen_preview is not None:
            self.pen_preview.append((x, y))
            self.render()

        elif self.tool == "circle" and self.circle_start is not None:
            x0, y0 = self.circle_start
            left, right = sorted((x0, x))
            top, bottom = sorted((y0, y))
            self.circle_preview = (left, top, right, bottom)
            self.render()

    def on_release(self, event):
        if not self.app.image_unlocked:
            return

        if self.dragging_text:
            self.dragging_text = False
            self.drag_last = None
            self.render()
            return

        if self.tool == "pen" and self.pen_preview is not None:
            if self.pen_preview:
                self.actions.append(
                    {"kind": "pen", "points": list(self.pen_preview), "width": 4}
                )
            self.pen_preview = None
            self.render()
            self.app.mark_image_dirty("Pen stroke added.")

        elif self.tool == "circle" and self.circle_preview is not None:
            left, top, right, bottom = self.circle_preview
            if (right - left) >= 3 and (bottom - top) >= 3:
                self.actions.append(
                    {
                        "kind": "circle",
                        "bbox": (left, top, right, bottom),
                        "width": 4,
                    }
                )
            self.circle_start = None
            self.circle_preview = None
            self.render()
            self.app.mark_image_dirty("Red ellipse added.")


class PiecesDemoApp(tk.Tk):
    def __init__(
        self,
        *,
        initial_model: str = DEFAULT_MODEL,
        asset_repo: str | None = DEFAULT_ASSET_REPO,
    ):
        super().__init__()
        self.title(APP_TITLE)

        self.screen_w = int(self.winfo_screenwidth())
        self.screen_h = int(self.winfo_screenheight())

        if self.screen_w >= 3000 or self.screen_h >= 1800:
            base_size, title_size, result_size = 16, 25, 15
            preferred_image = 520
        elif self.screen_w >= 2400 or self.screen_h >= 1350:
            base_size, title_size, result_size = 15, 24, 14
            preferred_image = 490
        elif self.screen_w >= 1750 or self.screen_h >= 1000:
            base_size, title_size, result_size = 14, 22, 14
            preferred_image = 450
        else:
            base_size, title_size, result_size = 13, 21, 13
            preferred_image = 400

        self.window_w = min(
            max(1180, int(self.screen_w * 0.70)),
            max(900, self.screen_w - 36),
        )
        self.window_h = min(
            max(720, int(self.screen_h * 0.72)),
            max(640, self.screen_h - 64),
        )

        # Reserve room for the controls and the permanently visible status bar.
        available_main_h = max(330, self.window_h - int(base_size * 15.5))
        max_image_by_width = max(330, int(self.window_w * 0.32))
        self.image_display_size = int(
            max(
                330,
                min(preferred_image, available_main_h, max_image_by_width),
            )
        )
        self.result_canvas_width = max(
            300,
            int((self.window_w - self.image_display_size - 105) / 2),
        )
        self.result_canvas_height = max(
            300,
            min(available_main_h, self.image_display_size + int(base_size * 4)),
        )

        x = max(0, (self.screen_w - self.window_w) // 2)
        y = max(0, (self.screen_h - self.window_h) // 2)
        self.geometry(f"{self.window_w}x{self.window_h}+{x}+{y}")
        self.minsize(
            min(1080, self.window_w),
            min(680, self.window_h),
        )

        default_family = tkfont.nametofont("TkDefaultFont").actual("family")
        self.font_base = (default_family, base_size)
        self.font_small = (default_family, max(10, base_size - 1))
        self.font_input = (default_family, base_size + 2)
        self.font_button = (default_family, base_size, "bold")
        self.font_title = (default_family, title_size, "bold")
        self.font_result = (default_family, result_size)
        self.font_result_bold = (default_family, result_size, "bold")
        self.font_result_tag = (default_family, max(9, result_size - 2), "bold")
        self.font_result_value = (default_family, max(10, result_size - 1))

        self.ui_pad = max(10, int(base_size * 0.85))
        self.ui_pad_small = max(6, int(base_size * 0.48))
        self.result_row_h = max(46, int(result_size * 3.5))
        self.tag_column_width = max(34, int(result_size * 2.8))

        avg_char_px = max(7.0, result_size * 0.58)
        self.candidate_column_width = int(avg_char_px * 20)

        self.value_column_width = max(94, int(result_size * 7.0))

        style = ttk.Style(self)
        style.configure(".", font=self.font_base)
        style.configure("TButton", font=self.font_button, padding=(8, 3))
        style.configure("TEntry", font=self.font_base)
        style.configure(
            "Candidate.TEntry",
            font=self.font_input,
            padding=(5, 5),
        )
        style.configure("TLabel", font=self.font_base)
        style.configure("TSpinbox", font=self.font_base)

        self.status_colors = {
            "error": ("#ffe3e3", "#8b1111"),
            "attention": ("#fff0cf", "#7b4a00"),
            "ready": ("#e3f5e5", "#175e24"),
            "info": ("#eef3f8", "#253746"),
        }
        self.lock_attention_bg = "#ffd99a"
        self.lock_normal_bg = self.cget("bg")

        self.model = None
        self.processor = None
        self.asset_repo = asset_repo
        self.device = (
            "cuda"
            if (torch is not None and torch.cuda.is_available())
            else "cpu"
        )

        self.current_image_path: Path | None = None
        self.current_config_path: Path | None = None
        self.demo_save_dir: Path = DEMO_DIR
        self.locked_image: Image.Image | None = None
        self.image_unlocked = False
        self.custom_mode = False
        self._image_tk = None
        self._busy = False
        self._busy_label = "WORKING…"

        self.result_view_mode = "confidence"
        self.result_bar_scale = "linear"
        self.last_visual_rows: list[dict[str, Any]] = []
        self.last_text_rows: list[dict[str, Any]] = []
        self.last_source_gate: float | None = None

        self.model_path_var = tk.StringVar(value=initial_model)
        self.text_var = tk.StringVar()
        self.visual_var = tk.StringVar()
        self.status_var = tk.StringVar(value="LOAD MODEL FIRST!")

        self.overlay_word_var = tk.StringVar()
        self.overlay_size_var = tk.StringVar(value="48")
        self.overlay_angle_var = tk.StringVar(value="0")

        self.editor = ImageEditor(self)
        self._build_ui()

        self.overlay_word_var.trace_add(
            "write", self.editor.update_selected_text_from_controls
        )
        self.overlay_size_var.trace_add(
            "write", self.editor.update_selected_text_from_controls
        )
        self.overlay_angle_var.trace_add(
            "write", self.editor.update_selected_text_from_controls
        )

        self.text_var.trace_add(
            "write", lambda *_: self._candidate_fields_changed()
        )
        self.visual_var.trace_add(
            "write", lambda *_: self._candidate_fields_changed()
        )

        self.image_canvas.bind("<ButtonPress-1>", self.editor.on_press)
        self.image_canvas.bind("<B1-Motion>", self.editor.on_motion)
        self.image_canvas.bind("<ButtonRelease-1>", self.editor.on_release)
        self.bind_all("<Control-z>", lambda _e: self.editor.undo())

        self.set_status("LOAD MODEL FIRST!", level="error")
        self.after(100, lambda: self.load_demo(DEFAULT_IMAGE, unlock=False))

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        outer = ttk.Frame(self, padding=self.ui_pad)
        outer.pack(fill="both", expand=True)

        model_frame = ttk.Frame(outer)
        model_frame.pack(fill="x", pady=(0, self.ui_pad_small))
        ttk.Label(model_frame, text="Model").pack(side="left")
        self.model_entry = ttk.Entry(
            model_frame,
            textvariable=self.model_path_var,
            width=76,
        )
        self.model_entry.pack(
            side="left",
            padx=self.ui_pad_small,
            fill="x",
            expand=True,
        )
        self.load_model_button = ttk.Button(
            model_frame,
            text="Load model",
            command=self.load_model,
        )
        self.load_model_button.pack(side="left")

        hint = (
            "comma-separated list of candidate text to read vs. visual object "
            "in the image"
        )
        ttk.Label(outer, text=hint, font=self.font_small).pack(anchor="w")

        candidate_frame = ttk.Frame(outer)
        candidate_frame.pack(
            fill="x",
            pady=(self.ui_pad_small // 2, self.ui_pad_small),
        )

        ttk.Label(candidate_frame, text="text:", width=8).grid(
            row=0, column=0, sticky="w"
        )
        self.text_entry = ttk.Entry(
            candidate_frame,
            textvariable=self.text_var,
            style="Candidate.TEntry",
        )
        self.text_entry.grid(
            row=0,
            column=1,
            sticky="ew",
            pady=max(2, self.ui_pad_small // 3),
        )

        ttk.Label(candidate_frame, text="visual:", width=8).grid(
            row=1, column=0, sticky="w"
        )
        self.visual_entry = ttk.Entry(
            candidate_frame,
            textvariable=self.visual_var,
            style="Candidate.TEntry",
        )
        self.visual_entry.grid(
            row=1,
            column=1,
            sticky="ew",
            pady=max(2, self.ui_pad_small // 3),
        )
        candidate_frame.columnconfigure(1, weight=1)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, self.ui_pad_small))

        self.run_button = ttk.Button(
            controls,
            text="RUN",
            command=self.run_model,
            state="disabled",
        )
        self.run_button.pack(side="left")

        ttk.Separator(controls, orient="vertical").pack(
            side="left",
            fill="y",
            padx=self.ui_pad,
            pady=2,
        )

        self.write_own_button = ttk.Button(
            controls,
            text="Write your own",
            command=self.write_your_own,
        )
        self.write_own_button.pack(side="left")

        self.custom_frame = ttk.Frame(controls)
        # hidden until Write your own
        ttk.Separator(self.custom_frame, orient="vertical").pack(
            side="left",
            fill="y",
            padx=self.ui_pad,
            pady=2,
        )
        self.load_image_button = ttk.Button(
            self.custom_frame,
            text="Load image",
            command=self.load_external_image,
        )
        self.load_image_button.pack(side="left")
        self.save_json_button = ttk.Button(
            self.custom_frame,
            text="Save json",
            command=self.save_demo,
        )
        self.save_json_button.pack(side="left", padx=(self.ui_pad_small, 0))

        ttk.Separator(self.custom_frame, orient="vertical").pack(
            side="left",
            fill="y",
            padx=self.ui_pad,
            pady=2,
        )

        self.edit_image_button = ttk.Button(
            self.custom_frame,
            text="Edit image",
            command=self.begin_editing,
        )
        self.edit_image_button.pack(side="left")

        self.lock_image_button = tk.Button(
            self.custom_frame,
            text="Lock image",
            command=self.lock_image,
            font=self.font_button,
            padx=9,
            pady=4,
            relief="raised",
            bg=self.lock_normal_bg,
            activebackground=self.lock_normal_bg,
        )
        self.lock_image_button.pack(
            side="left",
            padx=(self.ui_pad_small, 0),
        )

        self.view_toggle_button = ttk.Button(
            controls,
            text="Cosine view",
            command=self.toggle_result_view,
        )
        self.view_toggle_button.pack(side="right")

        self.log_bar_button = ttk.Button(
            controls,
            text="Log bars",
            command=self.toggle_bar_scale,
        )
        self.log_bar_button.pack(
            side="right",
            padx=(0, self.ui_pad_small),
        )

        self.editor_frame = ttk.Frame(outer)
        # hidden unless editing
        ttk.Button(
            self.editor_frame,
            text="T",
            width=3,
            command=lambda: self.editor.set_tool("text"),
        ).pack(side="left")

        ttk.Label(self.editor_frame, text="Word").pack(
            side="left",
            padx=(self.ui_pad_small, 2),
        )
        self.overlay_word_entry = ttk.Entry(
            self.editor_frame,
            textvariable=self.overlay_word_var,
            width=18,
        )
        self.overlay_word_entry.pack(side="left")

        ttk.Label(self.editor_frame, text="size").pack(
            side="left",
            padx=(self.ui_pad_small, 2),
        )
        ttk.Spinbox(
            self.editor_frame,
            from_=8,
            to=200,
            increment=2,
            textvariable=self.overlay_size_var,
            width=5,
        ).pack(side="left")

        ttk.Label(self.editor_frame, text="rotate°").pack(
            side="left",
            padx=(self.ui_pad_small, 2),
        )
        ttk.Spinbox(
            self.editor_frame,
            from_=-180,
            to=180,
            increment=5,
            textvariable=self.overlay_angle_var,
            width=6,
        ).pack(side="left")

        ttk.Button(
            self.editor_frame,
            text="✎ Pen",
            command=lambda: self.editor.set_tool("pen"),
        ).pack(side="left", padx=(self.ui_pad, 0))
        ttk.Button(
            self.editor_frame,
            text="◯",
            width=3,
            command=lambda: self.editor.set_tool("circle"),
        ).pack(side="left", padx=(self.ui_pad_small, 0))
        ttk.Button(
            self.editor_frame,
            text="Undo",
            command=self.editor.undo,
        ).pack(side="left", padx=(self.ui_pad, 0))
        ttk.Button(
            self.editor_frame,
            text="Reset blank",
            command=self.editor.reset_blank,
        ).pack(side="left", padx=(self.ui_pad_small, 0))

        self.status_frame = ttk.Frame(outer)
        self.status_frame.pack(side="bottom", fill="x")

        ttk.Separator(self.status_frame).pack(
            fill="x",
            pady=(self.ui_pad_small, self.ui_pad_small // 2),
        )
        self.status_label = tk.Label(
            self.status_frame,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            relief="sunken",
            padx=self.ui_pad,
            pady=max(7, self.ui_pad_small),
            wraplength=max(600, self.window_w - 2 * self.ui_pad),
            font=self.font_button,
        )
        self.status_label.pack(fill="x")

        main = ttk.Frame(outer)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=1)
        main.rowconfigure(0, weight=1)

        image_panel = ttk.Frame(main)
        image_panel.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, self.ui_pad),
        )
        ttk.Label(
            image_panel,
            text="IMAGE",
            font=self.font_title,
        ).pack(anchor="w")
        self.image_name_label = ttk.Label(
            image_panel,
            text="",
            font=self.font_small,
        )
        self.image_name_label.pack(
            anchor="w",
            pady=(0, self.ui_pad_small),
        )

        self.image_canvas = tk.Canvas(
            image_panel,
            width=self.image_display_size,
            height=self.image_display_size,
            highlightthickness=1,
            highlightbackground="#b8b8b8",
            background="#f4f4f4",
            cursor="crosshair",
        )
        self.image_canvas.pack()

        self.visual_results = ResultPane(
            main,
            "VISUAL",
            "what is depicted?",
        )
        self.visual_results.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=self.ui_pad_small,
        )

        self.text_results = ResultPane(
            main,
            "TEXT",
            "what is written?",
        )
        self.text_results.grid(
            row=0,
            column=2,
            sticky="nsew",
            padx=(self.ui_pad_small, 0),
        )

        self.visual_results.clear()
        self.text_results.clear()


    # ------------------------------------------------------------- status/state

    def set_status(
        self,
        message: str,
        error: bool = False,
        level: str | None = None,
    ):
        if level is None:
            level = "error" if error else "info"
        bg, fg = self.status_colors.get(level, self.status_colors["info"])
        self.status_var.set(message)
        self.status_label.configure(
            bg=bg,
            fg=fg,
        )

    def _required_action(self) -> tuple[str, str] | None:
        if self.model is None:
            return "LOAD MODEL FIRST!", "error"
        if self._busy:
            return self._busy_label, "attention"
        if self.locked_image is None or self.image_unlocked:
            return "LOCK IMAGE BEFORE RUN.", "attention"
        if not split_candidates(self.text_var.get()):
            return "FILL TEXT CANDIDATES BEFORE RUN.", "attention"
        if not split_candidates(self.visual_var.get()):
            return "FILL VISUAL CANDIDATES BEFORE RUN.", "attention"
        return None

    def _candidate_fields_changed(self):
        self.update_run_state()

    def _update_lock_button_state(self):
        if not hasattr(self, "lock_image_button"):
            return
        needs_lock = bool(
            self.custom_mode
            and (self.image_unlocked or self.locked_image is None)
        )
        bg = self.lock_attention_bg if needs_lock else self.lock_normal_bg
        self.lock_image_button.configure(
            bg=bg,
            activebackground=bg,
        )

    def mark_image_dirty(
        self,
        message: str = "Image changed.",
    ):
        if self.custom_mode:
            self.image_unlocked = True
            self.locked_image = None
        self.update_run_state(note=message)

    def update_run_state(self, note: str | None = None):
        required = self._required_action()
        ready = required is None

        self.run_button.configure(
            state="normal" if ready else "disabled"
        )
        self._update_lock_button_state()

        if required is not None:
            message, level = required
            self.set_status(message, level=level)
        else:
            message = "READY"
            if note:
                message += f" — {note}"
            self.set_status(message, level="ready")

    def set_busy(
        self,
        busy: bool,
        message: str = "WORKING…",
    ):
        self._busy = busy
        self._busy_label = message
        self.load_model_button.configure(
            state="disabled" if busy or self.model is not None else "normal"
        )
        self.update_run_state()

    def toggle_result_view(self):
        self.result_view_mode = (
            "cosine"
            if self.result_view_mode == "confidence"
            else "confidence"
        )
        if self.result_view_mode == "cosine":
            self.view_toggle_button.configure(text="Confidence view")
            self.visual_results.set_subtitle("content cosine similarity")
            self.text_results.set_subtitle("raw READ cosine similarity")
        else:
            self.view_toggle_button.configure(text="Cosine view")
        self._render_result_views()

    def toggle_bar_scale(self):
        self.result_bar_scale = (
            "log"
            if self.result_bar_scale == "linear"
            else "linear"
        )
        self.log_bar_button.configure(
            text="Linear bars"
            if self.result_bar_scale == "log"
            else "Log bars"
        )
        self._render_result_views()

    def _render_result_views(self):
        if self.result_view_mode == "confidence":
            suffix = " — log bars" if self.result_bar_scale == "log" else ""
            self.visual_results.set_subtitle("what is depicted?" + suffix)
            self.text_results.set_subtitle("what is written?" + suffix)

        if self.last_visual_rows:
            self.visual_results.show_results(
                self.last_visual_rows,
                view_mode=self.result_view_mode,
            )
        if self.last_text_rows:
            self.text_results.show_results(
                self.last_text_rows,
                view_mode=self.result_view_mode,
            )

    def canvas_to_image(
        self,
        x: float,
        y: float,
        *,
        clamp: bool,
    ) -> tuple[int, int] | None:
        size = float(self.image_display_size)
        if not clamp and not (0 <= x < size and 0 <= y < size):
            return None
        x = max(0.0, min(size - 1.0, float(x)))
        y = max(0.0, min(size - 1.0, float(y)))
        scale = CANVAS_SIZE / size
        return (
            max(0, min(CANVAS_SIZE - 1, int(round(x * scale)))),
            max(0, min(CANVAS_SIZE - 1, int(round(y * scale)))),
        )

    # --------------------------------------------------------------- demo/image

    def resolve_demo_asset(self, requested: Path) -> Path:
        """
        Resolve bundled demos from the checkout first, then from Hugging Face.

        A full repo clone therefore stays local/offline. A standalone copied script
        transparently uses the configured asset repo for any missing bundled files.
        """
        requested = Path(requested)
        if requested.is_file():
            return requested

        try:
            in_demo_dir = requested.parent.resolve() == DEMO_DIR.resolve()
        except Exception:
            in_demo_dir = requested.parent == DEMO_DIR
        if not in_demo_dir:
            return requested

        repo_id = self.asset_repo or hf_repo_id_from_reference(
            self.model_path_var.get().strip()
        )
        if not repo_id:
            return requested

        try:
            from huggingface_hub import hf_hub_download
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=f"demo_gui_images/{requested.name}",
            )
            return Path(cached)
        except Exception as exc:
            self.set_status(
                f"Could not fetch demo asset {requested.name!r} from {repo_id}: {exc}",
                error=True,
            )
            return requested

    def load_config(self, json_path: Path):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.set_status(f"Could not load {json_path.name}: {exc}", error=True)
            return

        # Public demo schema uses "text"; apple_none also accepts the requested
        # empty "words" field as a legacy/fallback alias.
        text_value = data.get("text", data.get("words", ""))
        visual_value = data.get("visual", "")
        self.text_var.set(
            ", ".join(text_value) if isinstance(text_value, list) else str(text_value)
        )
        self.visual_var.set(
            ", ".join(visual_value) if isinstance(visual_value, list) else str(visual_value)
        )

    def load_demo(self, image_path: Path, unlock: bool):
        requested_image_path = Path(image_path)
        image_path = self.resolve_demo_asset(requested_image_path)
        if not image_path.is_file():
            self.current_image_path = requested_image_path
            self.image_name_label.configure(text=image_path.name)
            placeholder = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "#eeeeee")
            d = ImageDraw.Draw(placeholder)
            d.text(
                (24, CANVAS_SIZE // 2 - 10),
                f"Missing image:\n{image_path}",
                fill=(60, 60, 60),
                font=resolve_pil_font(18),
            )
            self.editor.set_base(placeholder)
            self.locked_image = None
            repo_id = self.asset_repo or hf_repo_id_from_reference(
                self.model_path_var.get().strip()
            )
            remote_hint = (
                f" or fetch demo_gui_images/{requested_image_path.name} from {repo_id!r}"
                if repo_id
                else ""
            )
            self.set_status(
                f"Image not found locally{remote_hint}.",
                error=True,
            )
            self.image_unlocked = bool(unlock)
            self.update_run_state()
            return

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            self.set_status(f"Could not open image: {exc}", error=True)
            return

        self.current_image_path = requested_image_path
        requested_config = requested_image_path.with_suffix(".json")
        self.current_config_path = self.resolve_demo_asset(requested_config)
        self.image_name_label.configure(text=requested_image_path.name)
        self.editor.set_base(image)

        if self.current_config_path.is_file():
            self.load_config(self.current_config_path)

        self.image_unlocked = bool(unlock)
        if unlock:
            self.locked_image = None
        else:
            self.locked_image = self.editor.rendered_image(show_selection=False)
        self.update_run_state(note=f"Loaded {requested_image_path.name}.")

    def show_image(self, image: Image.Image):
        display = image.resize(
            (self.image_display_size, self.image_display_size),
            RESAMPLE,
        )
        self._image_tk = ImageTk.PhotoImage(display)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self._image_tk,
        )

    def write_your_own(self):
        # First click enters custom mode with the blank demo.  Subsequent clicks
        # only reveal the controls and preserve the currently loaded image,
        # editor actions, candidate fields, and lock state.
        first_entry = not self.custom_mode
        self.custom_mode = True
        if not self.custom_frame.winfo_ismapped():
            self.custom_frame.pack(side="left")
        if first_entry:
            self.load_demo(WRITE_YOUR_OWN_IMAGE, unlock=True)
        else:
            self.update_run_state(note="Write-your-own controls are already active.")

    def load_external_image(self):
        path = filedialog.askopenfilename(
            title="Load image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.custom_mode = True
        image_path = Path(path)
        self.load_demo(image_path, unlock=True)
        self.begin_editing()

    def begin_editing(self):
        if self.editor.base_image is None:
            self.set_status("Load an image first.", error=True)
            return
        self.image_unlocked = True
        self.locked_image = None
        if not self.editor_frame.winfo_ismapped():
            self.editor_frame.pack(fill="x", pady=(0, 10))
        self.update_run_state(note="Image unlocked for editing.")

    def lock_image(self):
        if self.editor.base_image is None:
            self.set_status("No image loaded.", error=True)
            return
        self.editor.commit_selected_text()
        self.locked_image = self.editor.rendered_image(show_selection=False)
        self.image_unlocked = False
        self.editor_frame.pack_forget()
        self.show_image(self.locked_image)
        self.update_run_state(
            note="Image locked; exact edited 448x448 RGB canvas is ready."
        )

    def _check_writable_directory(self, directory: Path) -> tuple[bool, str | None]:
        """Create/test a save directory without leaving a probe file behind."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".modemux_write_test_",
                dir=directory,
                delete=True,
            ):
                pass
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _choose_demo_save_directory(self) -> Path | None:
        """Use the session save folder; offer another one if it is not writable."""
        preferred = Path(getattr(self, "demo_save_dir", DEMO_DIR))
        ok, error = self._check_writable_directory(preferred)
        if ok:
            return preferred

        warning = (
            f"Cannot save to {preferred}: {error}. "
            "Choose another writable folder, or cancel to keep working without saving."
        )
        self.set_status(warning, level="attention")
        messagebox.showwarning("Cannot save beside the script", warning, parent=self)

        chosen = filedialog.askdirectory(
            title="Choose a writable folder for the demo PNG + JSON",
            parent=self,
        )
        if not chosen:
            self.set_status(
                "SAVE CANCELLED — the default demo folder is not writable.",
                level="attention",
            )
            return None

        directory = Path(chosen).expanduser()
        ok, error = self._check_writable_directory(directory)
        if not ok:
            self.set_status(
                f"SAVE FAILED — cannot write to {directory}: {error}",
                level="error",
            )
            messagebox.showerror(
                "Save failed",
                f"Cannot write to {directory}:\n{error}",
                parent=self,
            )
            return None
        self.demo_save_dir = directory
        return directory

    def save_demo(self):
        text_candidates = split_candidates(self.text_var.get())
        visual_candidates = split_candidates(self.visual_var.get())
        if not text_candidates or not visual_candidates:
            self.set_status("Fill both text and visual before saving.", error=True)
            return

        image = self.editor.rendered_image(show_selection=False)
        default_stem = (
            self.current_image_path.stem
            if self.current_image_path is not None
            else "my_demo"
        )
        stem = simpledialog.askstring(
            "Save demo",
            "Demo name (PNG + JSON; default folder: demo_gui_images):",
            initialvalue=default_stem,
            parent=self,
        )
        if stem is None:
            return
        stem = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in stem.strip())
        stem = stem.strip("_") or "my_demo"

        save_dir = self._choose_demo_save_directory()
        if save_dir is None:
            return

        image_path = save_dir / f"{stem}.png"
        json_path = save_dir / f"{stem}.json"

        if (image_path.exists() or json_path.exists()) and not messagebox.askyesno(
            "Overwrite?",
            f"{stem}.png/json already exists in\n{save_dir}\n\nOverwrite?",
            parent=self,
        ):
            self.set_status("SAVE CANCELLED — existing files were left unchanged.", level="info")
            return

        # Serialize both files to temporary siblings before touching the final names.
        # This prevents partial output for the common serialization/permission failures.
        token = f"{os.getpid()}_{threading.get_ident()}"
        tmp_image = save_dir / f".{stem}.{token}.png.tmp"
        tmp_json = save_dir / f".{stem}.{token}.json.tmp"
        payload = {
            "text": ", ".join(text_candidates),
            "visual": ", ".join(visual_candidates),
        }

        try:
            image.save(tmp_image, format="PNG")
            tmp_json.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp_image, image_path)
            os.replace(tmp_json, json_path)
        except Exception as exc:
            for tmp in (tmp_image, tmp_json):
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
            self.set_status(
                f"SAVE FAILED — {type(exc).__name__}: {exc}",
                level="error",
            )
            messagebox.showerror(
                "Save failed",
                f"Could not save the demo to {save_dir}:\n{type(exc).__name__}: {exc}",
                parent=self,
            )
            return

        self.current_image_path = image_path
        self.current_config_path = json_path
        self.image_name_label.configure(text=image_path.name)
        # Update button/lock state first, then leave an explicit success message in
        # the always-visible status bar even when the edited image still needs locking.
        self.update_run_state()
        self.set_status(
            f"SAVED OK — {image_path.name} + {json_path.name} → {save_dir}",
            level="ready",
        )

    # ------------------------------------------------------------------- model

    def load_model(self):
        if self.model is not None:
            self.set_status(
                "Model is already loaded and will remain resident until the GUI closes."
            )
            return
        if TORCH_IMPORT_ERROR is not None:
            self.set_status(f"PyTorch import failed: {TORCH_IMPORT_ERROR}", error=True)
            return

        model_reference = self.model_path_var.get().strip()
        if not model_reference:
            self.set_status("Model path / Hugging Face repo ID is empty.", error=True)
            return

        self.set_busy(True, message="LOAD MODEL FIRST! — loading model…")
        self.set_status(
            "LOAD MODEL FIRST! — loading model…",
            level="error",
        )

        def worker():
            try:
                from transformers import AutoModel, AutoProcessor

                model = AutoModel.from_pretrained(
                    model_reference,
                    trust_remote_code=True,
                )
                processor = AutoProcessor.from_pretrained(
                    model_reference,
                    trust_remote_code=True,
                )

                # Reference/demo path: keep the entire checkpoint in FP32.
                model = model.float().eval().to(self.device)

                if str(getattr(model.config, "model_type", "")) != "xattn_clip":
                    raise RuntimeError(
                        f"Expected model_type='xattn_clip', got "
                        f"{getattr(model.config, 'model_type', None)!r}."
                    )
                if getattr(model, "read_implant", None) is None:
                    raise RuntimeError("Loaded model has no read_implant.")

                architecture = str(
                    getattr(model.config, "read_attention_architecture", "")
                )
                if architecture != "sigmoid_all":
                    raise RuntimeError(
                        f"Expected final sigmoid_all reader, got {architecture!r}."
                    )

                self.after(
                    0,
                    lambda: self._model_loaded(
                        model=model,
                        processor=processor,
                        model_reference=model_reference,
                    ),
                )
            except Exception as exc:
                # Python clears the exception name after the except block, so bind
                # it now for the later Tk callback.
                self.after(0, lambda exc=exc: self._model_failed(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _model_loaded(self, model, processor, model_reference: str):
        self.model = model
        self.processor = processor
        self.model_entry.configure(state="disabled")
        self.load_model_button.configure(text="Model loaded ✓", state="disabled")
        self.set_busy(False)
        self.update_run_state(
            note=(
                f"Model loaded from {model_reference!r} on {self.device}; "
                "it stays resident until the GUI closes."
            )
        )

    def _model_failed(self, exc: Exception):
        self.set_busy(False)
        self.set_status(f"Model load failed: {type(exc).__name__}: {exc}", error=True)

    # ---------------------------------------------------------------- inference

    def run_model(self):
        if self.model is None or self.processor is None:
            self.set_status("Load model first.", error=True)
            return
        if self.locked_image is None or self.image_unlocked:
            self.set_status("Lock the image before RUN.", error=True)
            return

        text_candidates = split_candidates(self.text_var.get())
        visual_candidates = split_candidates(self.visual_var.get())
        if not text_candidates or not visual_candidates:
            self.set_status(
                "Both text and visual candidate fields must be filled.",
                error=True,
            )
            return

        candidates, origins = union_candidates(text_candidates, visual_candidates)
        if not candidates:
            self.set_status("No usable candidates.", error=True)
            return

        image = self.locked_image.copy().convert("RGB")
        self.set_busy(
            True,
            message=f"RUNNING {len(candidates)} CANDIDATES…",
        )
        self.visual_results.clear("Running…")
        self.text_results.clear("Running…")

        def worker():
            try:
                prompts = [
                    PROMPT.format(candidate=candidate)
                    for candidate in candidates
                ]
                inputs = self.processor(
                    text=prompts,
                    images=image,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"].to(self.device)
                pixel_values = inputs["pixel_values"].to(self.device)

                with torch.inference_mode():
                    # User-facing VISUAL is the robust automatic semantic mode.
                    visual_output = self.model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        mode="any",
                        correction=True,
                        return_details=True,
                        pieces_fp32=True,
                    )
                    # User-facing TEXT is the exported model's read mode.  It
                    # automatically appends the internal <text><null> candidate.
                    text_output = self.model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        mode="read",
                        correction=True,
                        return_details=True,
                        pieces_fp32=True,
                    )

                visual_logits = (
                    visual_output.logits_per_image[0].detach().float().cpu()
                )
                text_logits = (
                    text_output.logits_per_image[0].detach().float().cpu()
                )

                # Cosine view deliberately exposes two different underlying
                # similarities:
                #   VISUAL -> corrected content/image semantic cosine
                #   TEXT   -> raw PIECES READ cosine before calibration / NULL policy
                visual_cosines = (
                    visual_output.image_embeds[0].detach().float().cpu()
                    @ visual_output.text_embeds.detach().float().cpu().T
                )
                text_details = getattr(text_output, "details", None)
                if not isinstance(text_details, dict):
                    raise RuntimeError(
                        "TEXT cosine view requires return_details=True."
                    )
                raw_read_logits = text_details.get("raw_read_logits")
                if raw_read_logits is None:
                    raise RuntimeError(
                        "TEXT output did not expose raw_read_logits."
                    )
                logit_scale = float(
                    self.model.logit_scale.detach().float().exp().cpu()
                )
                text_cosines = (
                    raw_read_logits[0].detach().float().cpu() / logit_scale
                )

                n = len(candidates)
                if visual_logits.numel() != n:
                    raise RuntimeError(
                        f"VISUAL returned {visual_logits.numel()} columns for "
                        f"{n} candidates."
                    )

                null_index = text_output.null_candidate_index
                if null_index is None:
                    raise RuntimeError(
                        'HF mode="read" did not expose a null candidate.'
                    )
                null_index = int(null_index)
                if text_logits.numel() != n + 1 or null_index != n:
                    raise RuntimeError(
                        f"Unexpected TEXT/null layout: logits={text_logits.numel()}, "
                        f"candidate_count={n}, null_candidate_index={null_index}."
                    )

                visual_probs = torch.softmax(visual_logits, dim=0)
                text_probs = torch.softmax(text_logits, dim=0)

                visual_rows = []
                for i, candidate in enumerate(candidates):
                    key = " ".join(candidate.casefold().split())
                    visual_rows.append(
                        {
                            "candidate": candidate,
                            "origin": origins.get(key, ""),
                            "prob": float(visual_probs[i]),
                            "logit": float(visual_logits[i]),
                            "cosine": float(visual_cosines[i]),
                            "is_null": False,
                        }
                    )

                text_rows = []
                for i, candidate in enumerate(candidates):
                    key = " ".join(candidate.casefold().split())
                    text_rows.append(
                        {
                            "candidate": candidate,
                            "origin": origins.get(key, ""),
                            "prob": float(text_probs[i]),
                            "logit": float(text_logits[i]),
                            "cosine": float(text_cosines[i]),
                            "is_null": False,
                        }
                    )
                text_rows.append(
                    {
                        "candidate": "abstain",
                        "origin": "",
                        "prob": float(text_probs[null_index]),
                        "logit": float(text_logits[null_index]),
                        "cosine": float(text_cosines[null_index]),
                        "is_null": True,
                    }
                )

                source_gate = None
                details = getattr(visual_output, "details", None)
                if isinstance(details, dict) and details.get("source_gate") is not None:
                    source_gate = float(
                        details["source_gate"][0].detach().float().cpu()
                    )

                self.after(
                    0,
                    lambda: self._show_run_results(
                        visual_rows, text_rows, source_gate
                    ),
                )
            except Exception as exc:
                self.after(0, lambda exc=exc: self._run_failed(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _show_run_results(self, visual_rows, text_rows, source_gate):
        self.last_visual_rows = list(visual_rows)
        self.last_text_rows = list(text_rows)
        self.last_source_gate = source_gate
        self._render_result_views()
        self.set_busy(False)

        visual_winner = (
            max(visual_rows, key=lambda row: row["logit"])["candidate"]
            if visual_rows
            else "?"
        )
        text_winner = (
            max(text_rows, key=lambda row: row["logit"])["candidate"]
            if text_rows
            else "?"
        )
        extra = (
            f"  SOURCE={source_gate:.3f}"
            if source_gate is not None and math.isfinite(source_gate)
            else ""
        )
        self.update_run_state(
            note=f"VISUAL → {visual_winner}   |   TEXT → {text_winner}{extra}"
        )

    def _run_failed(self, exc: Exception):
        self.set_busy(False)
        self.visual_results.clear("Run failed.")
        self.text_results.clear("Run failed.")
        self.set_status(
            f"RUN failed: {type(exc).__name__}: {exc}",
            level="error",
        )



def parse_args():
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Local HF export directory or Hugging Face repo ID. If this script lives "
            "inside a complete ModeMUX repo clone, that local checkout is the default; "
            "otherwise the public HF repo is used. Remote repositories load with "
            "trust_remote_code=True."
        ),
    )
    parser.add_argument(
        "--asset-repo",
        default=DEFAULT_ASSET_REPO,
        help=(
            "HF repo used only when a bundled demo_gui_images/* file is missing locally. "
            "A complete clone therefore needs no network. Defaults to the official "
            "ModeMUX repo; pass an empty string to fall back to --model when --model "
            "itself is an HF repo ID."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = PiecesDemoApp(
        initial_model=args.model,
        asset_repo=(args.asset_repo or None),
    )
    app.mainloop()


if __name__ == "__main__":
    main()
