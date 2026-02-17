#!/usr/bin/env python3
"""
Combine multiple images into one figure with simple labels.

Usage example (vertical stack):
python scripts/combine_images.py \
  --input "train (45,15):assets/eigen_ind49_train(45,15).png" \
  --input "train (60,30):assets/eigen_ind49_train(60,30).png" \
  --input "fx (30,15):assets/eigen_fx_train(30,15).png" \
  --layout vertical \
  --output assets/eigen_combined.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="Format: LABEL:PATH (PATH to image). LABEL is optional; if omitted, uses filename stem.",
    )
    ap.add_argument("--output", type=Path, required=True, help="Output image path.")
    ap.add_argument(
        "--output-format",
        default="png",
        choices=["png", "pdf"],
        help="Output format (default: png).",
    )
    ap.add_argument("--layout", choices=["vertical", "horizontal"], default="vertical", help="Stack direction.")
    ap.add_argument("--label-pad", type=int, default=30, help="Pixels reserved above each image for the label.")
    ap.add_argument("--label-font-size", type=int, default=18, help="Font size for labels.")
    ap.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saving (and PDF rasterization if inputs are PDFs).",
    )
    ap.add_argument("--no-labels", action="store_true", help="Disable label banners and stacking padding.")
    ap.add_argument(
        "--label",
        action="append",
        help="Optional explicit label for each input, same order as --input when labels are omitted in the input strings.",
    )
    return ap.parse_args()


def _parse_input(entry: str) -> Tuple[str, Path]:
    if ":" in entry:
        label, path = entry.split(":", 1)
        label = label.strip() or Path(path).stem
        return label, Path(path).expanduser()
    p = Path(entry).expanduser()
    return p.stem, p


def _load_pdf_with_pillow(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
            return img.convert("RGB").copy()
    except Exception:
        return None


def _load_pdf_with_pymupdf(path: Path, dpi: int) -> Image.Image | None:
    try:
        import fitz  # type: ignore
    except Exception:
        return None
    doc = None
    try:
        doc = fitz.open(str(path))
        if doc.page_count < 1:
            return None
        page = doc.load_page(0)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=True)
        if pix.alpha:
            img = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples).convert("RGB")
        else:
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img
    except Exception:
        return None
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def _load_pdf_with_pdf2image(path: Path, dpi: int) -> Image.Image | None:
    try:
        from pdf2image import convert_from_path  # type: ignore
    except Exception:
        return None
    try:
        images = convert_from_path(str(path), dpi=dpi, first_page=1, last_page=1)
        if images:
            return images[0].convert("RGB")
    except Exception:
        return None
    return None


def _load_image(path: Path, dpi: int) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        img = _load_pdf_with_pillow(path)
        if img is None:
            img = _load_pdf_with_pymupdf(path, dpi)
        if img is None:
            img = _load_pdf_with_pdf2image(path, dpi)
        if img is None:
            raise SystemExit(
                "Unable to read PDF inputs. Install PyMuPDF (`pip install pymupdf`) or "
                "pdf2image + poppler, or enable Ghostscript support in Pillow."
            )
        return img
    try:
        with Image.open(path) as img:
            return img.convert("RGB").copy()
    except UnidentifiedImageError as exc:
        raise SystemExit(f"Unsupported image format: {path}") from exc


def main() -> None:
    args = parse_args()
    fmt = args.output_format.lstrip(".").lower()
    out_path = args.output.with_suffix(f".{fmt}")
    items: List[Tuple[str, Path]] = []
    for idx, entry in enumerate(args.input):
        label, path = _parse_input(entry)
        if args.label and idx < len(args.label) and args.label[idx]:
            label = args.label[idx]
        items.append((label, path))

    images = []
    for label, path in items:
        if not path.exists():
            raise SystemExit(f"Input image does not exist: {path}")
        im = _load_image(path, args.dpi)
        if args.no_labels:
            images.append(im)
            continue
        # Choose font
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", args.label_font_size)
        except Exception:
            font = ImageFont.load_default()
        try:
            bbox = font.getbbox(label)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = font.getsize(label)
        pad = max(args.label_pad, text_h + 10)
        canvas = Image.new("RGB", (im.width, im.height + pad), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        # label banner
        banner_height = pad
        draw.rectangle([0, 0, im.width, banner_height], fill=(245, 245, 245))
        draw.text((10, (banner_height - text_h) // 2), label, fill=(0, 0, 0), font=font)
        canvas.paste(im, (0, banner_height))
        images.append(canvas)

    if args.layout == "vertical":
        width = max(im.width for im in images)
        height = sum(im.height for im in images)
        out = Image.new("RGB", (width, height), (255, 255, 255))
        y = 0
        for im in images:
            out.paste(im, (0, y))
            y += im.height
    else:
        height = max(im.height for im in images)
        width = sum(im.width for im in images)
        out = Image.new("RGB", (width, height), (255, 255, 255))
        x = 0
        for im in images:
            out.paste(im, (x, 0))
            x += im.width

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pdf":
        out.save(out_path, resolution=args.dpi)
    else:
        out.save(out_path, dpi=(args.dpi, args.dpi))
    print(f"Saved combined image to {out_path}")


if __name__ == "__main__":
    main()
