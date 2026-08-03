"use client";

// Aave-style glass pane (aave.com/design/building-glass-for-the-web):
// frosted backdrop blur + saturation, gradient tint, rim highlights and a
// heavier frost band around the edge. All knobs exposed for the design lab.
export interface GlassParams {
  frost: number;
  saturation: number;
  tint: number;
  edge: number;
  distortion: number;
  ringWidth: number;
  ringDepth: number;
  ringDark: number;
}

export const DEFAULT_GLASS_PARAMS: GlassParams = {
  frost: 8,
  saturation: 1.4,
  tint: 0.35,
  edge: 0.8,
  distortion: 40,
  ringWidth: 2,
  ringDepth: 6,
  ringDark: 0.45,
};

export function GlassSurface({ params }: { params: GlassParams }) {
  const { frost, saturation, tint, edge } = params;
  const backdropFilter = `blur(${frost}px) saturate(${saturation})`;

  return (
    <>
      <div
        className="pointer-events-none absolute inset-0 rounded-full"
        style={{
          backdropFilter,
          WebkitBackdropFilter: backdropFilter,
          backgroundImage: `linear-gradient(155deg, rgba(255,255,255,${tint}), rgba(255,255,255,${tint * 0.2}) 48%, rgba(255,255,255,${tint * 0.6}))`,
          border: `1px solid rgba(255,255,255,${Math.min(edge, 1)})`,
          boxShadow: `inset 0 1px 3px rgba(255,255,255,${Math.min(edge * 1.2, 1)}), inset 0 -12px 28px rgba(255,255,255,${edge * 0.5}), 0 12px 40px rgba(96,64,224,0.18)`,
        }}
      />
      <div
        className="pointer-events-none absolute inset-0 rounded-full [mask-image:radial-gradient(closest-side,transparent_50%,black_80%)]"
        style={{
          backdropFilter: `blur(${frost * 2.5}px)`,
          WebkitBackdropFilter: `blur(${frost * 2.5}px)`,
        }}
      />
    </>
  );
}
