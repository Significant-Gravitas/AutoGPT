import { useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { useRef } from "react";
import { POLL_CAP_MS, POLL_MS } from "./helpers";

/** Find the delegated run's session while the tool output (which would name
 *  it) doesn't exist yet: the expert's currently-processing session. If the
 *  expert happens to be busy elsewhere this shows that work instead — still
 *  their live status, and the exact session takes over once the tool
 *  returns. */
export function useDelegatedSessionId(expertId: string | null) {
  const mountedAtRef = useRef(Date.now());
  const { data } = useGetV2ListSessions(
    // Strict recency: the default puts pinned sessions first, so an expert
    // with a handful of pinned threads would push the running one out of
    // the window and the live view would never find it.
    { expert_id: expertId ?? undefined, limit: 5, pinned_first: false },
    {
      query: {
        enabled: !!expertId,
        refetchInterval: (query) => {
          if (query.state.status === "error") return false;
          if (Date.now() - mountedAtRef.current > POLL_CAP_MS) return false;
          const raw = query.state.data;
          const sessions = raw && raw.status === 200 ? raw.data.sessions : [];
          const hasLive = sessions.some(
            (s) => s.is_processing || s.chat_status === "running",
          );
          // Once a live session is found, SubSessionLive takes over polling
          // it directly by id — this list poll has nothing left to watch for.
          return hasLive ? false : POLL_MS;
        },
        select: (res) => (res.status === 200 ? res.data.sessions : []),
      },
    },
  );
  const live = (data ?? []).find(
    (s) => s.is_processing || s.chat_status === "running",
  );
  return live?.id ?? null;
}
