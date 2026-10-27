import { useState } from "react";
import { useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";

/** A failed hand-off tells Otto to send the task again himself when that can
 *  help, so by the time the user can click, he may already have done it. A
 *  hire is brand new and has no chat of its own until someone starts one, so
 *  any thread with them means the task may already be there: the card then
 *  points at it instead of offering a second copy. Read fresh once the turn
 *  is over (this mounts only then), never from a cache filled mid-turn. */
export function useHandoffRetry(expertId: string) {
  const [asked, setAsked] = useState(false);
  const { data, isFetchedAfterMount, isError } = useGetV2ListSessions(
    { expert_id: expertId, limit: 1, pinned_first: false },
    {
      query: {
        select: (res) => (res.status === 200 ? res.data.sessions : null),
      },
    },
  );
  const known = isFetchedAfterMount && !isError && !!data;
  return {
    asked,
    markAsked: () => setAsked(true),
    known,
    threadId: known ? (data[0]?.id ?? null) : null,
  };
}
