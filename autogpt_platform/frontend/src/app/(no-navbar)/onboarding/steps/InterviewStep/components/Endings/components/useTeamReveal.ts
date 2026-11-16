import { useGetBrainDumpPlan } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";

export const TEAM_POLL_MS = 2_000;
// Reads of the plan before the plain ending stays in the team's place.
const TEAM_READS = 15;

// The team the hand-off proposed, as the plan step reads it, read until it
// is ready.
export function useTeamReveal() {
  const query = useGetBrainDumpPlan({
    query: {
      refetchOnWindowFocus: false,
      refetchInterval: (q) => {
        const data = q.state.data;
        const isReady = data?.status === 200 && data.data.ready;
        return isReady || q.state.dataUpdateCount >= TEAM_READS
          ? false
          : TEAM_POLL_MS;
      },
    },
  });
  const plan = query.data?.status === 200 ? query.data.data : null;
  return { team: plan?.ready ? (plan.team ?? []) : [] };
}
