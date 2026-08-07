const COLS = 16;
const ROWS = 7;

// Deterministic pseudo-random per cell (Knuth multiplicative hash) so the
// server and client render the identical mosaic — Math.random here would
// be a hydration mismatch.
function cellOpacity(index: number) {
  const noise = ((index + 1) * 2654435761) % 1000;
  return 0.02 + (noise / 1000) * 0.14;
}

// A "glass pixel" mosaic over the stage gradient: a grid of frosted white
// tiles at varying opacities, letting the same violet gradient glow
// through each pixel differently.
export function GlassPixelBackdrop() {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 grid"
      style={{
        gridTemplateColumns: `repeat(${COLS}, 1fr)`,
        gridTemplateRows: `repeat(${ROWS}, 1fr)`,
      }}
    >
      {Array.from({ length: COLS * ROWS }, (_, index) => (
        <span
          key={index}
          className="border-[0.5px] border-white/10 bg-white"
          style={{ opacity: cellOpacity(index) }}
        />
      ))}
    </div>
  );
}
