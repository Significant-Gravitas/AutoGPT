import {
  useGetDelegationSettings,
  useListDelegations,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { getSentFromDisplayName, type SentFrom } from "../../../../sentFrom";
import { useExpertMap } from "../../../../useExpertMap";

interface Args {
  sentFrom: SentFrom;
  sessionId: string;
}

/** Who delegated this thread, and the hand-off record Otto's chat keeps
 *  for it: its title, brief, status and what it has cost so far. */
export function useDelegatedThreadNotice({ sentFrom, sessionId }: Args) {
  const { expertsById } = useExpertMap();
  const delegator = sentFrom.expertId
    ? (expertsById.get(sentFrom.expertId) ?? null)
    : null;
  const from = getSentFromDisplayName(sentFrom, delegator?.name);

  const listQuery = useListDelegations(
    { parent_session_id: sentFrom.sessionId },
    { query: { select: (res) => okData(res) ?? null } },
  );
  const settingsQuery = useGetDelegationSettings({
    query: { select: (res) => okData(res) ?? null },
  });
  const delegation =
    listQuery.data?.delegations.find((d) => d.sub_session_id === sessionId) ??
    null;

  return {
    from,
    isOtto: from === AUTOPILOT_NAME,
    delegator,
    delegation,
    capUsd: settingsQuery.data?.per_delegation_cap_usd ?? null,
  };
}
