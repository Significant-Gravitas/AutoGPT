// The grid classes below spell these out (Tailwind needs static classes).
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
      data-slot="glass-pixel-backdrop"
      aria-hidden
      className="pointer-events-none absolute inset-0 grid grid-cols-[repeat(16,1fr)] grid-rows-[repeat(7,1fr)]"
    >
      {Array.from({ length: COLS * ROWS }, (_, index) => (
        <span
          key={index}
          data-slot="glass-pixel-backdrop-cell"
          // TODO(tokens): no border width is 0.5px. The opacity differs per
          // cell, so it is a custom property read by an arbitrary value.
          className="opacity-(--cell-opacity) border-[0.5px] border-surface/10 bg-surface"
          style={
            { "--cell-opacity": cellOpacity(index) } as React.CSSProperties
          }
        />
      ))}
    </div>
  );
}
