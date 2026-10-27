import { useListExpertFirstJobs } from "@/app/api/__generated__/endpoints/experts/experts";
import { shouldPoll } from "./helpers";

const POLL_MS = 5000;

export function useTeamAtWork() {
  const { data: response, isLoading } = useListExpertFirstJobs({
    query: {
      // Installs and first runs finish on their own; poll until they do.
      refetchInterval: (query) => {
        const data = query.state.data;
        const items = data?.status === 200 ? data.data.items : [];
        return shouldPoll(items) ? POLL_MS : false;
      },
    },
  });
  const items = response?.status === 200 ? response.data.items : [];

  return { items, isLoading };
}
