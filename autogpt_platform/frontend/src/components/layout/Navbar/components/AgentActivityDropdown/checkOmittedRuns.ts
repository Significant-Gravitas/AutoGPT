import { getGetV1GetExecutionDetailsQueryOptions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { okData } from "@/app/api/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { QueryClient } from "@tanstack/react-query";

// Each full list reads at most this many of the runs it leaves out.
const MAX_CHECKS_PER_LIST = 10;

interface Args {
  runs: GraphExecutionMeta[];
  queryClient: QueryClient;
  log: {
    mark(): number;
    updatedSince(runID: string, mark: number): boolean;
  };
  checking: Set<string>;
  onRead: (run: GraphExecutionMeta) => void;
  onGone: (runID: string) => void;
}

/**
 * A full run list holds only the newest runs, so it can't tell whether an
 * older run shown as going has ended. One that ended while live updates were
 * lost would show as going for good, so read each such run by ID. A live
 * update that arrives during the read is newer than it and stands.
 */
export async function checkOmittedRuns({
  runs,
  queryClient,
  log,
  checking,
  onRead,
  onGone,
}: Args) {
  const due = runs
    .filter((run) => !checking.has(run.id))
    .slice(0, MAX_CHECKS_PER_LIST);
  await Promise.all(
    due.map(async (run) => {
      checking.add(run.id);
      const mark = log.mark();
      try {
        const response = await queryClient.fetchQuery({
          ...getGetV1GetExecutionDetailsQueryOptions(run.graph_id, run.id),
          staleTime: 0,
        });
        const read = okData(response);
        if (read && !log.updatedSince(run.id, mark)) onRead(read);
      } catch (error) {
        const gone = error instanceof ApiError && error.status === 404;
        if (gone && !log.updatedSince(run.id, mark)) onGone(run.id);
        // Any other failure keeps the run as shown; the next list reads it.
      } finally {
        checking.delete(run.id);
      }
    }),
  );
}
