import { cn } from "@/lib/utils";
import { BADGE_FILL, BADGE_GLYPH } from "./statusBadge";
import type { AvatarStatus } from "./status";

interface Props {
  status: AvatarStatus;
  size: number;
  className?: string;
}

/** The status badge as an overlay, for the avatar rendered as an image. The
 *  in-artwork badge draws the same glyphs at the same radius. */
export function StatusDot({ status, size, className }: Props) {
  if (status === "idle") return null;

  return (
    <svg
      viewBox="-10 -10 20 20"
      width={size}
      height={size}
      aria-hidden
      data-testid="status-dot"
      data-status={status}
      className={cn("pointer-events-none shrink-0", className)}
      dangerouslySetInnerHTML={{
        __html: `<circle r="8" fill="${BADGE_FILL[status]}" stroke="#fff" stroke-width="2.5"/>${BADGE_GLYPH[status]}`,
      }}
    />
  );
}
