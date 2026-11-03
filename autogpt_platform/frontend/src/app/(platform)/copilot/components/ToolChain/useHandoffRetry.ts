import { useState } from "react";
import {
  getGetV2ListSessionsQueryKey,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { getCopilotHref } from "@/services/org-team/builder";
import { useCopilotTenantScope } from "../../CopilotTenantScopeContext";

/** Try again is the one retry path: Otto is told not to resend a failed
 *  hand-off himself. Someone may still have started work with the hire since
 *  (the user, by asking Otto in text). A hire is brand new and has no chat of
 *  its own until someone starts one, so any thread with them means the task
 *  may already be there: the card then points at it instead of offering a
 *  second copy. Read fresh once the turn is over (this mounts only then),
 *  never from a cache filled mid-turn. */
export function useHandoffRetry(expertId: string) {
  const scope = useCopilotTenantScope();
  const [asked, setAsked] = useState(false);
  const filters = { expert_id: expertId, limit: 1, pinned_first: false };
  const { data, isFetchedAfterMount, isError } = useGetV2ListSessions(filters, {
    query: {
      queryKey: getTeamScopedQueryKey(
        getGetV2ListSessionsQueryKey(filters),
        scope.organizationId,
        scope.teamId,
      ),
      select: (res) => (res.status === 200 ? res.data.sessions : null),
    },
    request: getTenantRequestInit(scope.organizationId, scope.teamId),
  });
  const known = isFetchedAfterMount && !isError && !!data;
  const threadId = known ? (data[0]?.id ?? null) : null;
  return {
    asked,
    markAsked: () => setAsked(true),
    known,
    threadId,
    // The thread was found in this chat's scope, so it opens in it.
    threadHref: threadId
      ? getCopilotHref(threadId, scope.organizationId, scope.teamId)
      : null,
  };
}
