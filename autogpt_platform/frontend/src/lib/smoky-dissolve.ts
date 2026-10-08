function readNum(name: string, fallback: number): number {
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  if (!raw) return fallback;
  if (raw.endsWith("ms")) return Number.parseFloat(raw);
  if (raw.endsWith("s")) return Number.parseFloat(raw) * 1000;
  const n = Number.parseFloat(raw);
  return Number.isNaN(n) ? fallback : n;
}

function readStr(name: string): string {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}

function makeEaseSampler(raw: string): (x: number) => number {
  const kw: Record<string, [number, number, number, number]> = {
    linear: [0, 0, 1, 1],
    ease: [0.25, 0.1, 0.25, 1],
    "ease-in": [0.42, 0, 1, 1],
    "ease-out": [0, 0, 0.58, 1],
    "ease-in-out": [0.42, 0, 0.58, 1],
  };
  let c = kw[raw.trim()];
  if (!c) {
    const m =
      /cubic-bezier\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)/.exec(
        raw,
      );
    if (m) c = [Number(m[1]), Number(m[2]), Number(m[3]), Number(m[4])];
  }
  const [x1, y1, x2, y2] = c ?? kw.ease!;
  const cx = 3 * x1;
  const bx = 3 * (x2 - x1) - cx;
  const ax = 1 - cx - bx;
  const cy = 3 * y1;
  const by = 3 * (y2 - y1) - cy;
  const ay = 1 - cy - by;
  const sx = (t: number) => ((ax * t + bx) * t + cx) * t;
  const dxf = (t: number) => (3 * ax * t + 2 * bx) * t + cx;
  const sy = (t: number) => ((ay * t + by) * t + cy) * t;
  return (x) => {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    let t = x;
    for (let i = 0; i < 6; i++) {
      const e = sx(t) - x;
      if (Math.abs(e) < 1e-4) break;
      const d = dxf(t);
      if (Math.abs(d) < 1e-6) break;
      t -= e / d;
    }
    return sy(t);
  };
}

const TRANSPARENT = /^rgba?\(.*,\s*0\)$/;

function radiusOf(style: CSSStyleDeclaration, w: number, h: number): number {
  const raw = style.borderTopLeftRadius;
  const value = raw.endsWith("%")
    ? (Number.parseFloat(raw) / 100) * Math.min(w, h)
    : Number.parseFloat(raw);
  return Math.min(Number.isNaN(value) ? 0 : value, Math.min(w, h) / 2);
}

function paintSvg(
  ctx: CanvasRenderingContext2D,
  svg: SVGSVGElement,
  x: number,
  y: number,
) {
  const box = svg.getBoundingClientRect();
  const view = svg.viewBox.baseVal;
  const vw = view.width || box.width;
  const vh = view.height || box.height;
  if (!vw || !vh) return;
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(box.width / vw, box.height / vh);
  ctx.translate(-view.x, -view.y);
  for (const path of svg.querySelectorAll("path")) {
    const d = path.getAttribute("d");
    if (!d) continue;
    const style = getComputedStyle(path);
    const shape = new Path2D(d);
    if (style.fill && style.fill !== "none" && !TRANSPARENT.test(style.fill)) {
      ctx.fillStyle = style.fill;
      ctx.fill(shape);
    }
    if (
      style.stroke &&
      style.stroke !== "none" &&
      !TRANSPARENT.test(style.stroke)
    ) {
      ctx.strokeStyle = style.stroke;
      ctx.lineWidth = Number.parseFloat(style.strokeWidth) || 1;
      ctx.lineCap = style.strokeLinecap as CanvasLineCap;
      ctx.lineJoin = style.strokeLinejoin as CanvasLineJoin;

      const dash = style.strokeDasharray;
      if (dash && dash !== "none") {
        ctx.setLineDash(
          dash
            .split(/[\s,]+/)
            .map(Number.parseFloat)
            .filter(Number.isFinite),
        );
        ctx.lineDashOffset = Number.parseFloat(style.strokeDashoffset) || 0;
      }
      ctx.stroke(shape);
      ctx.setLineDash([]);
    }
  }
  ctx.restore();
}

function fillBox(
  ctx: CanvasRenderingContext2D,
  style: CSSStyleDeclaration,
  x: number,
  y: number,
  w: number,
  h: number,
) {
  const r = radiusOf(style, w, h);
  if (
    !TRANSPARENT.test(style.backgroundColor) &&
    style.backgroundColor !== "transparent"
  ) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    ctx.fillStyle = style.backgroundColor;
    ctx.fill();
  }
  const bw = Number.parseFloat(style.borderTopWidth);
  if (bw > 0 && !TRANSPARENT.test(style.borderTopColor)) {
    ctx.beginPath();
    ctx.roundRect(
      x + bw / 2,
      y + bw / 2,
      w - bw,
      h - bw,
      Math.max(0, r - bw / 2),
    );
    ctx.strokeStyle = style.borderTopColor;
    ctx.lineWidth = bw;
    ctx.stroke();
  }
}

function paintPseudo(
  ctx: CanvasRenderingContext2D,
  node: Element,
  which: "::before" | "::after",
  x: number,
  y: number,
) {
  const style = getComputedStyle(node, which);
  if (style.content === "none" || style.position !== "absolute") return;
  const w = Number.parseFloat(style.width);
  const h = Number.parseFloat(style.height);
  const top = Number.parseFloat(style.top);
  const left = Number.parseFloat(style.left);
  if (![w, h, top, left].every(Number.isFinite)) return;
  fillBox(ctx, style, x + left, y + top, w, h);
}

function paint(
  ctx: CanvasRenderingContext2D,
  root: Element,
  ox: number,
  oy: number,
) {
  const origin = root.getBoundingClientRect();
  const walk = (node: Element) => {
    const style = getComputedStyle(node);
    const alpha = Number(style.opacity);
    if (
      style.visibility === "hidden" ||
      style.display === "none" ||
      alpha === 0
    )
      return;
    const box = node.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) return;

    if (style.overflow === "hidden" && box.width <= 1 && box.height <= 1)
      return;
    const x = box.left - origin.left + ox;
    const y = box.top - origin.top + oy;

    ctx.save();
    ctx.globalAlpha = Number.isNaN(alpha) ? 1 : alpha;
    if (node instanceof SVGSVGElement) {
      paintSvg(ctx, node, x, y);
      ctx.restore();
      return;
    }

    fillBox(ctx, style, x, y, box.width, box.height);
    paintPseudo(ctx, node, "::before", x, y);

    ctx.fillStyle = style.color;
    ctx.font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
    ctx.textBaseline = "middle";

    ctx.direction = style.direction === "rtl" ? "rtl" : "ltr";
    ctx.textAlign = "left";
    for (const child of node.childNodes) {
      if (child.nodeType !== Node.TEXT_NODE || !child.textContent?.trim())
        continue;

      const range = document.createRange();
      range.selectNodeContents(child);
      if (range.getClientRects().length > 1) {
        for (const word of child.textContent.matchAll(/\S+/g)) {
          range.setStart(child, word.index);
          range.setEnd(child, word.index + word[0].length);
          const at = range.getBoundingClientRect();
          ctx.fillText(
            word[0],
            at.left - origin.left + ox,
            at.top - origin.top + oy + at.height / 2,
          );
        }
        continue;
      }
      const line = range.getBoundingClientRect();
      ctx.fillText(
        child.textContent,
        line.left - origin.left + ox,
        line.top - origin.top + oy + line.height / 2,
      );
    }
    paintPseudo(ctx, node, "::after", x, y);
    ctx.restore();

    for (const child of node.children) walk(child);
  };
  walk(root);
}

function boxH(
  srcA: Uint8ClampedArray,
  dstA: Uint8ClampedArray,
  w: number,
  h: number,
  r: number,
) {
  const inv = 1 / (2 * r + 1);
  for (let y = 0; y < h; y++) {
    const base = y * w * 4;
    let sr = 0;
    let sg = 0;
    let sb = 0;
    let sa = 0;
    for (let x = 0; x <= r && x < w; x++) {
      const i = base + x * 4;
      sr += srcA[i]!;
      sg += srcA[i + 1]!;
      sb += srcA[i + 2]!;
      sa += srcA[i + 3]!;
    }
    for (let x = 0; x < w; x++) {
      const o = base + x * 4;
      dstA[o] = sr * inv;
      dstA[o + 1] = sg * inv;
      dstA[o + 2] = sb * inv;
      dstA[o + 3] = sa * inv;
      const xa = x + r + 1;
      if (xa < w) {
        const ia = base + xa * 4;
        sr += srcA[ia]!;
        sg += srcA[ia + 1]!;
        sb += srcA[ia + 2]!;
        sa += srcA[ia + 3]!;
      }
      const xs = x - r;
      if (xs >= 0) {
        const is = base + xs * 4;
        sr -= srcA[is]!;
        sg -= srcA[is + 1]!;
        sb -= srcA[is + 2]!;
        sa -= srcA[is + 3]!;
      }
    }
  }
}

function boxV(
  srcA: Uint8ClampedArray,
  dstA: Uint8ClampedArray,
  w: number,
  h: number,
  r: number,
  straight: boolean,
) {
  const stride = w * 4;
  const inv = 1 / (2 * r + 1);
  for (let x = 0; x < w; x++) {
    const base = x * 4;
    let sr = 0;
    let sg = 0;
    let sb = 0;
    let sa = 0;
    for (let y = 0; y <= r && y < h; y++) {
      const i = base + y * stride;
      sr += srcA[i]!;
      sg += srcA[i + 1]!;
      sb += srcA[i + 2]!;
      sa += srcA[i + 3]!;
    }
    for (let y = 0; y < h; y++) {
      const o = base + y * stride;
      const aOut = sa * inv;
      if (straight && aOut > 0.5 && aOut < 254.6) {
        const k = 255 / aOut;
        dstA[o] = sr * inv * k;
        dstA[o + 1] = sg * inv * k;
        dstA[o + 2] = sb * inv * k;
      } else {
        dstA[o] = sr * inv;
        dstA[o + 1] = sg * inv;
        dstA[o + 2] = sb * inv;
      }
      dstA[o + 3] = aOut;
      const ya = y + r + 1;
      if (ya < h) {
        const ia = base + ya * stride;
        sr += srcA[ia]!;
        sg += srcA[ia + 1]!;
        sb += srcA[ia + 2]!;
        sa += srcA[ia + 3]!;
      }
      const ys = y - r;
      if (ys >= 0) {
        const is = base + ys * stride;
        sr -= srcA[is]!;
        sg -= srcA[is + 1]!;
        sb -= srcA[is + 2]!;
        sa -= srcA[is + 3]!;
      }
    }
  }
}

export type DissolveOptions = { onComplete?: () => void };

export function dissolve(
  el: HTMLElement,
  { onComplete }: DissolveOptions = {},
) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    onComplete?.();
    return;
  }

  const durMs = readNum("--smoky-dur", 550);
  const g = readNum("--smoky-gravity", 150);
  const warp = readNum("--smoky-warp", 30);
  const warpDurRaw = readNum("--smoky-warp-dur", 0);
  const warpDurMs = warpDurRaw > 0 ? warpDurRaw : durMs;
  const maxBlur = readNum("--smoky-blur", 12);
  const sway = readNum("--smoky-sway", 0);
  const spin = readNum("--smoky-spin", 3);
  const churn = readNum("--smoky-churn", 30);
  const spread = readNum("--smoky-spread", 0);
  const grain = readNum("--smoky-grain", 28);
  const warpEase = makeEaseSampler(readStr("--smoky-warp-ease"));
  const blurEase = makeEaseSampler(readStr("--smoky-blur-ease"));
  const gravityEase = makeEaseSampler(readStr("--smoky-gravity-ease"));
  const dissolveEase = makeEaseSampler(readStr("--smoky-dissolve-ease"));

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = el.getBoundingClientRect();
  const cardW = rect.width;
  const cardH = rect.height;
  const pad = Math.min(Math.ceil(warp / 2 + maxBlur * 2) + 6, 80);
  const padDev = Math.round(pad * dpr);
  const snapW = Math.max(1, Math.round(cardW * dpr));
  const snapH = Math.max(1, Math.round(cardH * dpr));
  const workW = snapW + padDev * 2;
  const workH = snapH + padDev * 2;

  const canvas = document.createElement("canvas");
  canvas.width = workW;
  canvas.height = workH;
  canvas.setAttribute("aria-hidden", "true");
  canvas.style.cssText = `position:fixed;left:${rect.left - pad}px;top:${rect.top - pad}px;width:${cardW + pad * 2}px;height:${cardH + pad * 2}px;pointer-events:none;z-index:200;will-change:transform,opacity`;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    onComplete?.();
    return;
  }
  ctx.imageSmoothingQuality = "high";

  const snap = document.createElement("canvas");
  snap.width = snapW;
  snap.height = snapH;
  const sc = snap.getContext("2d");
  if (!sc) {
    onComplete?.();
    return;
  }
  sc.setTransform(dpr, 0, 0, dpr, 0, 0);
  paint(sc, el, 0, 0);

  const sd = sc.getImageData(0, 0, snapW, snapH).data;
  const snapPre = new Uint8ClampedArray(sd.length);
  for (let i = 0; i < sd.length; i += 4) {
    const aP = sd[i + 3]!;
    snapPre[i] = (sd[i]! * aP) / 255;
    snapPre[i + 1] = (sd[i + 1]! * aP) / 255;
    snapPre[i + 2] = (sd[i + 2]! * aP) / 255;
    snapPre[i + 3] = aP;
  }

  const workHalf = document.createElement("canvas");
  const wW2 = Math.ceil(workW / 2);
  const wH2 = Math.ceil(workH / 2);
  workHalf.width = wW2;
  workHalf.height = wH2;
  const whctx = workHalf.getContext("2d")!;
  const workData = ctx.createImageData(workW, workH);
  const halfData = whctx.createImageData(wW2, wH2);
  const blurTmp = new Uint8ClampedArray(halfData.data.length);

  const LAT = 4;
  const latW = Math.ceil(workW / LAT) + 2;
  const latH = Math.ceil(workH / LAT) + 2;
  const latDX = new Float32Array(latW * latH);
  const latDY = new Float32Array(latW * latH);

  function remap(
    dstArr: Uint8ClampedArray,
    dw: number,
    dh: number,
    s: number,
    straight: boolean,
  ) {
    const src = snapPre;
    const invLat = 1 / LAT;
    let di = 0;
    for (let yD = 0; yD < dh; yD++) {
      const yF = yD * s;
      const gy = yF * invLat;
      const gy0 = gy | 0;
      const fy = gy - gy0;
      const row0 = gy0 * latW;
      const row1 = row0 + latW;
      for (let xD = 0; xD < dw; xD++, di += 4) {
        const xF = xD * s;
        const gx = xF * invLat;
        const gx0 = gx | 0;
        const fx = gx - gx0;
        const a = row0 + gx0;
        const b = row1 + gx0;
        const dxv =
          (latDX[a]! + (latDX[a + 1]! - latDX[a]!) * fx) * (1 - fy) +
          (latDX[b]! + (latDX[b + 1]! - latDX[b]!) * fx) * fy;
        const dyv =
          (latDY[a]! + (latDY[a + 1]! - latDY[a]!) * fx) * (1 - fy) +
          (latDY[b]! + (latDY[b + 1]! - latDY[b]!) * fx) * fy;
        const sxf = xF - padDev + dxv;
        const syf = yF - padDev + dyv;
        const sx0 = Math.floor(sxf);
        const sy0 = Math.floor(syf);
        if (sx0 < -1 || sx0 > snapW - 1 || sy0 < -1 || sy0 > snapH - 1) {
          dstArr[di] = 0;
          dstArr[di + 1] = 0;
          dstArr[di + 2] = 0;
          dstArr[di + 3] = 0;
          continue;
        }
        const u = sxf - sx0;
        const v = syf - sy0;
        let rC = 0;
        let gC = 0;
        let bC = 0;
        let aC = 0;
        for (const [ox2, oy2, w] of [
          [0, 0, (1 - u) * (1 - v)],
          [1, 0, u * (1 - v)],
          [0, 1, (1 - u) * v],
          [1, 1, u * v],
        ] as const) {
          const sx = sx0 + ox2;
          const sy = sy0 + oy2;
          if (sx < 0 || sx >= snapW || sy < 0 || sy >= snapH) continue;
          const si = (sy * snapW + sx) * 4;
          rC += src[si]! * w;
          gC += src[si + 1]! * w;
          bC += src[si + 2]! * w;
          aC += src[si + 3]! * w;
        }
        if (straight && aC > 0.5 && aC < 254.6) {
          const k = 255 / aC;
          rC *= k;
          gC *= k;
          bC *= k;
        }
        dstArr[di] = rC;
        dstArr[di + 1] = gC;
        dstArr[di + 2] = bC;
        dstArr[di + 3] = aC;
      }
    }
  }

  const N = 64;
  const noiseR = new Float32Array(N * N);
  const noiseG = new Float32Array(N * N);
  let seed = 4;
  for (let i = 0; i < N * N; i++) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    noiseR[i] = seed / 2147483648 - 1;
    seed = (seed * 1664525 + 1013904223) >>> 0;
    noiseG[i] = seed / 2147483648 - 1;
  }
  const gridAt = (gArr: Float32Array, x: number, y: number) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    let fx = x - xi;
    let fy = y - yi;
    fx = fx * fx * (3 - 2 * fx);
    fy = fy * fy * (3 - 2 * fy);
    const x0 = ((xi % N) + N) % N;
    const x1 = (x0 + 1) % N;
    const y0 = ((yi % N) + N) % N;
    const y1 = (y0 + 1) % N;
    const top =
      gArr[y0 * N + x0]! + (gArr[y0 * N + x1]! - gArr[y0 * N + x0]!) * fx;
    const bot =
      gArr[y1 * N + x0]! + (gArr[y1 * N + x1]! - gArr[y1 * N + x0]!) * fx;
    return top + (bot - top) * fy;
  };
  const noise2 = (gArr: Float32Array, px: number, py: number) =>
    (gridAt(gArr, px / grain, py / grain) +
      0.5 * gridAt(gArr, (px / grain) * 2 + 37.7, (py / grain) * 2 + 11.3)) /
    1.5;

  el.style.visibility = "hidden";
  document.body.appendChild(canvas);
  const t0 = performance.now();
  let frame = 0;

  const tick = (now: number) => {
    const t = (now - t0) / 1000;

    const rawP = Math.min((now - t0) / Math.max(durMs, 1), 1);
    const p = dissolveEase(rawP);

    const teff = gravityEase(p) * (durMs / 1000);
    const fallY = 0.5 * g * teff * teff;
    const swayX = Math.sin(t * 5) * sway * p;
    const pw = Math.min((now - t0) / Math.max(warpDurMs, 1), 1);
    const dispScale = warp * warpEase(pw);
    const blur = maxBlur * blurEase(p);
    const alpha = 1 - p ** 1.6;
    const driftY = churn * t;
    const driftX = Math.sin(t * 3.2) * churn * 0.3;

    if (frame % 2 === 0 || rawP >= 1) {
      const halfDev = (dispScale / 2) * dpr;
      for (let ly = 0; ly < latH; ly++) {
        const py = (ly * LAT - padDev) / dpr;
        for (let lx = 0; lx < latW; lx++) {
          const px = (lx * LAT - padDev) / dpr;
          const li = ly * latW + lx;
          latDX[li] = halfDev * noise2(noiseR, px - driftX, py - driftY);
          latDY[li] = halfDev * noise2(noiseG, px - driftX, py - driftY);
        }
      }
      const rBox = blur > 0.3 ? Math.round(blur * dpr * 1.22) : 0;
      if (rBox >= 2) {
        remap(halfData.data, wW2, wH2, 2, false);
        const rHalf = Math.max(1, Math.round(rBox / 2));
        boxH(halfData.data, blurTmp, wW2, wH2, rHalf);
        boxH(blurTmp, halfData.data, wW2, wH2, rHalf);
        boxV(halfData.data, blurTmp, wW2, wH2, rHalf, false);
        boxV(blurTmp, halfData.data, wW2, wH2, rHalf, true);
        whctx.putImageData(halfData, 0, 0);
        ctx.clearRect(0, 0, workW, workH);
        ctx.drawImage(workHalf, 0, 0, wW2, wH2, 0, 0, workW, workH);
      } else {
        remap(workData.data, workW, workH, 1, true);
        ctx.putImageData(workData, 0, 0);
      }
    }
    frame++;

    canvas.style.transform = `translate(${swayX.toFixed(1)}px, ${fallY.toFixed(1)}px) rotate(${(spin * p).toFixed(2)}deg) scale(${(1 + (spread / 100) * p).toFixed(3)})`;
    canvas.style.opacity = alpha.toFixed(3);

    if (rawP < 1) {
      requestAnimationFrame(tick);
      return;
    }
    canvas.remove();
    onComplete?.();
  };
  requestAnimationFrame(tick);
}
