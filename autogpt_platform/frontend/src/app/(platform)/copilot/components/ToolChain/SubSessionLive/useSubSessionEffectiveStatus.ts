import { isSessionLive } from "./helpers";
import { useLiveSubSession } from "./useLiveSubSession";

/** A sub-session tool output freezes the status it had when the tool
 *  returned — a "running" card would say running forever after the work
 *  lands. While the frozen status is running/queued, read the truth off the
 *  polled session and flip to completed once it goes idle.
 *
 *  Owns the poll through `useLiveSubSession` rather than piggybacking on a
 *  mounted live view: a minimal delegate card renders no live view, so this
 *  is the only thing left that can flip running → completed. On a full card
 *  it is the same query key, so the two share one poll. */
export function useSubSessionEffectiveStatus(
  subSessionId: string | null,
  status: string | null,
) {
  const stale = ["running", "queued"].includes(status?.toLowerCase() ?? "");
  const { session, isError, isPaused } = useLiveSubSession(
    subSessionId ?? "",
    stale && !!subSessionId,
  );
  if (!stale) return status;
  // The frozen status is only trustworthy while the poll can refute it. A
  // minimal card has no "Live updates paused" notice to fall back on, so a
  // dead poll has to show up in the pill or it reads as fact.
  if (isError || isPaused) return "unknown";
  if (!session) return status;
  return isSessionLive(session) ? status : "completed";
}
