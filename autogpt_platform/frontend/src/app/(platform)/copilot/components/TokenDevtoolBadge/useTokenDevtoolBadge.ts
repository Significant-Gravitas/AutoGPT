import { useTokenDevtoolStore } from "../../tokenDevtool/store";
import { breakdownTotal, displayContext } from "../../tokenDevtool/tokenMath";

export function useTokenDevtoolBadge(sessionId: string) {
  const turns = useTokenDevtoolStore((s) => s.turnsBySession[sessionId]);
  const breakdown = useTokenDevtoolStore(
    (s) => s.breakdownBySession[sessionId],
  );
  const liveContext = useTokenDevtoolStore(
    (s) => s.liveContextBySession[sessionId],
  );
  const compacted = useTokenDevtoolStore(
    (s) => s.compactedBySession[sessionId] ?? false,
  );
  const seed = breakdown ? breakdownTotal(breakdown) : undefined;
  const context = displayContext(liveContext ?? null, compacted, seed);
  // The rows describe the loaded-history seed, so showing them next to a
  // headline sourced from the live cache-write sum would read as one total
  // that does not add up. Only render them while the seed IS the headline.
  const visibleBreakdown =
    breakdown !== undefined && context === seed ? breakdown : undefined;

  return { turns, breakdown: visibleBreakdown, context };
}
