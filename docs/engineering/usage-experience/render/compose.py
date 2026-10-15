import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if len(sys.argv) != 4:
    raise SystemExit(
        "Usage: compose.py BEFORE_DIRECTORY AFTER_DIRECTORY OUTPUT_DIRECTORY"
    )
BEFORE, AFTER, OUT = [Path(p).resolve() for p in sys.argv[1:]]
OUT.mkdir(exist_ok=True)
full_output = os.environ.get("USAGE_COMPARISON_FULL_OUTPUT")
FULL_OUT = Path(full_output).resolve() if full_output else None
if FULL_OUT:
    FULL_OUT.mkdir(parents=True, exist_ok=True)
cases = json.loads((Path(__file__).parent / "comparison-cases.json").read_text())
font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F = lambda size, heavy=False: ImageFont.truetype(bold if heavy else font, size)


def wrap(draw, text, font, width):
    lines = []
    line = ""
    for word in text.split():
        candidate = (line + " " + word).strip()
        if line and draw.textlength(candidate, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def write(draw, text, pos, font, width, color):
    x, y = pos
    for line in wrap(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=color)
        y += int(font.size * 1.45)
    return y


report = []
for case in cases:
    identifier = case["id"]
    improvement, note = (case["improvement"], case.get("note"))
    before = BEFORE / f"{identifier}.png"
    after = AFTER / f"{identifier}-1280.png"
    if not after.exists():
        raise FileNotFoundError(after)
    a = Image.open(after).convert("RGB")
    b = Image.open(before).convert("RGB") if before.exists() else None
    # The production capture uses DPR 2. Final implementation captures use DPR 1.
    bw, bh = (b.width / 2, b.height / 2) if b else (560, 300)
    aw, ah = a.size
    column = int(max(bw, aw, 560))
    gap = 28
    pad = 28
    width = 2 * column + gap + 2 * pad
    measure = Image.new("RGB", (width, 10))
    d = ImageDraw.Draw(measure)
    titleh = len(wrap(d, case["title"], F(24, True), width - 2 * pad)) * 35
    summaryh = len(wrap(d, improvement, F(17), width - 2 * pad)) * 25
    top = pad + titleh + 9 + summaryh + 28 + 34
    footer = (
        (note + " " if note else "")
        + "Both sides use sample account data and the same UI scale. Left: deployed source 4473fbbc (verified Oct 12, 2026). Right: actual PR components, not deployed."
    )
    footerh = len(wrap(d, footer, F(13), width - 2 * pad)) * 19
    height = top + int(max(bh, ah, 300)) + 22 + footerh + pad
    canvas = Image.new("RGB", (width, height), "#f4f4f5")
    d = ImageDraw.Draw(canvas)
    y = write(d, case["title"], (pad, pad), F(24, True), width - 2 * pad, "#252529")
    write(d, improvement, (pad, y + 9), F(17), width - 2 * pad, "#5b5b65")
    d.text(
        (pad, top - 34),
        "Before · production components",
        font=F(16, True),
        fill="#333338",
    )
    d.text(
        (pad + column + gap, top - 34),
        "After · actual PR components",
        font=F(16, True),
        fill="#6031b7",
    )
    if b:
        b = b.resize((round(bw), round(bh)), Image.Resampling.LANCZOS)
        canvas.paste(b, (pad + (column - b.width) // 2, top))
    else:
        box = (pad + (column - 560) // 2, top, pad + (column + 560) // 2, top + 300)
        d.rounded_rectangle(box, 16, fill="white", outline="#d4d4db", width=1)
        x = box[0] + 28
        y = write(d, "NO EXISTING SCREEN", (x, top + 25), F(12, True), 504, "#74747c")
        y = write(d, case["absence"], (x, y + 15), F(20, True), 504, "#333338")
        write(
            d,
            case["absenceDetail"].replace("proposed", "new"),
            (x, y + 16),
            F(15),
            504,
            "#666670",
        )
    canvas.paste(a, (pad + column + gap + (column - a.width) // 2, top))
    fy = top + int(max(bh, ah, 300)) + 22
    d.line((pad, fy - 10, width - pad, fy - 10), fill="#dedee3", width=1)
    write(d, footer, (pad, fy), F(13), width - 2 * pad, "#666670")
    if FULL_OUT:
        canvas.save(FULL_OUT / f"{identifier}.png", optimize=True)
    scale = min(1, 1000 / width, 760 / height)
    compact = canvas.resize(
        (round(width * scale), round(height * scale)), Image.Resampling.LANCZOS
    )
    compact.save(OUT / f"{identifier}.png", optimize=True)
    report.append(
        {
            "id": identifier,
            "title": case["title"],
            "improvement": improvement,
            "note": note,
            "before": before.name if b else None,
            "after": after.name,
            "dimensions": [compact.width, compact.height],
            "beforeSha256": (
                hashlib.sha256(before.read_bytes()).hexdigest() if b else None
            ),
            "afterSha256": hashlib.sha256(after.read_bytes()).hexdigest(),
        }
    )
(OUT / "comparison-manifest.json").write_text(
    json.dumps(
        {
            "productionRelease": "4473fbbcb1951471df0a05374d47c93c4e39bea6",
            "source": "Actual components with illustrative data; no independently drawn UI",
            "pairs": report,
        },
        indent=2,
    )
    + "\n"
)
print(f"Composed {len(report)} actual-component comparison pairs.")
