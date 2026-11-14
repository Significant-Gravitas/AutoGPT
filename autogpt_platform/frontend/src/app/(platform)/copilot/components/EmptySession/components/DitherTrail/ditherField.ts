/* The trail is a grid of squares. Each square holds how recently the
 * pointer was near it, from 1 down to 0, and is drawn only while that
 * number is above its own threshold in a 4x4 ordered-dither pattern. So the
 * trail is solid at the pointer, breaks into a dither towards its edge, and
 * thins out square by square as it fades. */

/** One square of the grid, gap included, in CSS pixels. */
const CELL = 14;
const GAP = 3;
/** How far from the pointer squares light up, and how fast that falls off. */
const RADIUS = 120;
const FALLOFF = 1.6;
/** How quickly the trail fades, per second: higher is faster. */
const FADE = 1.9;
/** The most opaque a square ever gets. The page's content sits on top of
 *  the trail, so it stays well under what would compete with text. */
const ALPHA = 0.07;
/** Below the lowest threshold nothing is drawn, so the square is dropped. */
const FLOOR = 1 / 34;
/** Points filled in between two pointer events, so a fast move leaves a
 *  line and not a row of dots. */
const MAX_STEPS = 24;

const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(
  (rank) => (rank + 0.5) / 16,
);

/** Draws the pointer's dither trail on the canvas until the returned
 *  function is called. It listens on the window, since the canvas sits
 *  behind the page and never gets the pointer itself, and it draws only
 *  while there is something left to fade. */
export function runDitherTrail(
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
) {
  const color = getComputedStyle(canvas).color;
  let cols = 0;
  let rows = 0;
  let field = new Float32Array(0);
  let frame = 0;
  let lastTime = 0;
  let lastPoint: { x: number; y: number } | null = null;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    cols = Math.ceil(rect.width / CELL);
    rows = Math.ceil(rect.height / CELL);
    field = new Float32Array(cols * rows);
  }

  function light(x: number, y: number) {
    const reach = Math.ceil(RADIUS / CELL);
    const centreCol = Math.floor(x / CELL);
    const centreRow = Math.floor(y / CELL);
    for (let row = centreRow - reach; row <= centreRow + reach; row++) {
      if (row < 0 || row >= rows) continue;
      for (let col = centreCol - reach; col <= centreCol + reach; col++) {
        if (col < 0 || col >= cols) continue;
        const distance = Math.hypot(
          (col + 0.5) * CELL - x,
          (row + 0.5) * CELL - y,
        );
        if (distance >= RADIUS) continue;
        const value = (1 - distance / RADIUS) ** FALLOFF;
        const index = row * cols + col;
        if (value > field[index]) field[index] = value;
      }
    }
  }

  function draw(now: number) {
    const keep = Math.exp(-Math.min(0.05, (now - lastTime) / 1000) * FADE);
    lastTime = now;
    let isAlive = false;
    context.clearRect(0, 0, cols * CELL, rows * CELL);
    context.fillStyle = color;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const index = row * cols + col;
        const value = field[index] * keep;
        if (value < FLOOR) {
          field[index] = 0;
          continue;
        }
        field[index] = value;
        isAlive = true;
        if (value <= BAYER[(row & 3) * 4 + (col & 3)]) continue;
        context.globalAlpha = ALPHA * Math.min(1, value * 1.5);
        context.fillRect(
          col * CELL + GAP / 2,
          row * CELL + GAP / 2,
          CELL - GAP,
          CELL - GAP,
        );
      }
    }
    frame = isAlive ? requestAnimationFrame(draw) : 0;
  }

  function onMove(event: PointerEvent) {
    if (event.pointerType !== "mouse") return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const isNear =
      x > -RADIUS &&
      y > -RADIUS &&
      x < rect.width + RADIUS &&
      y < rect.height + RADIUS;
    if (!isNear) {
      lastPoint = null;
      return;
    }
    const from = lastPoint ?? { x, y };
    const steps = Math.min(
      MAX_STEPS,
      Math.max(1, Math.ceil(Math.hypot(x - from.x, y - from.y) / CELL)),
    );
    for (let step = 1; step <= steps; step++) {
      const part = step / steps;
      light(from.x + (x - from.x) * part, from.y + (y - from.y) * part);
    }
    lastPoint = { x, y };
    if (frame) return;
    lastTime = performance.now();
    frame = requestAnimationFrame(draw);
  }

  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  resize();
  window.addEventListener("pointermove", onMove, { passive: true });

  return function stop() {
    cancelAnimationFrame(frame);
    observer.disconnect();
    window.removeEventListener("pointermove", onMove);
  };
}
