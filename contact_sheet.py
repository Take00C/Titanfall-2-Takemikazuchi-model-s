"""Stage 31 - tile the review renders into one image.

    python -m tsfronin.contact_sheet [unit]   -> build/renders/<unit>/contact_sheet.png
"""
import argparse

from PIL import Image, ImageDraw, ImageFont

from . import config

ORDER = [("front", "front"), ("threequarter_front", "3/4 front"), ("side_left", "left side"),
         ("threequarter_back", "3/4 back"), ("back", "back"), ("upper_body", "upper body"),
         ("clay_threequarter_front", "geometry only")]


def sheet(unit, cell_h=900):
    src = config.RENDER_DIR / unit
    tiles = [(label, Image.open(src / f"{tag}.png").convert("RGB"))
             for tag, label in ORDER if (src / f"{tag}.png").exists()]
    if not tiles:
        raise SystemExit(f"no renders in {src}")
    cols = 4
    rows = -(-len(tiles) // cols)
    cw = round(cell_h * tiles[0][1].width / tiles[0][1].height)
    out = Image.new("RGB", (cols * cw, rows * cell_h), (24, 26, 30))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for i, (label, im) in enumerate(tiles):
        x, y = (i % cols) * cw, (i // cols) * cell_h
        out.paste(im.resize((cw, cell_h), Image.LANCZOS), (x, y))
        draw.rectangle([x, y, x + cw, y + 46], fill=(24, 26, 30))
        draw.text((x + 14, y + 7), label, fill=(235, 235, 235), font=font)
    dst = src / "contact_sheet.png"
    out.save(dst)
    print(f"wrote {dst}  ({out.width}x{out.height})")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("unit", nargs="?", default=config.UNIT)
    sheet(ap.parse_args().unit)


if __name__ == "__main__":
    main()
