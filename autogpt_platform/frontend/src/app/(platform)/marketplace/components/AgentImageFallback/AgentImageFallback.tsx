import BoringAvatar from "boring-avatars";
import { cn } from "@/lib/utils";

interface Props {
  // Stable per-agent seed (graph id / slug) so each agent always gets the same
  // gradient. Do not seed with user/creator id — these are per-agent thumbnails.
  seed: string;
  className?: string;
}

const FALLBACK_COLORS = ["#92A1C6", "#146A7C", "#F0AB3D", "#C271B4", "#C20D90"];

// Deterministic gradient shown when an agent has no thumbnail. Rendered as an
// inline SVG (boring-avatars, already a dependency) so there is no network
// request and nothing to fail — unlike the external placeholder it replaces.
export function AgentImageFallback({ seed, className }: Props) {
  return (
    <BoringAvatar
      size="100%"
      name={seed}
      variant="marble"
      colors={FALLBACK_COLORS}
      square
      preserveAspectRatio="none"
      className={cn("absolute inset-0 h-full w-full", className)}
    />
  );
}
