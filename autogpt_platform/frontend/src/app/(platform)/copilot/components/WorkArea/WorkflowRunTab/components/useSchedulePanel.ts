import {
  getGetV1ListExecutionSchedulesForAGraphQueryKey,
  useDeleteV1DeleteExecutionSchedule,
  useGetV1ListExecutionSchedulesForAGraph,
  usePostV1CreateExecutionSchedule,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { browserTimezone, dailyCron, DEFAULT_CUSTOM_TIME } from "../schedule";

interface Args {
  graphID: string;
  agent: LibraryAgent | undefined;
  /** The run whose inputs a new schedule repeats. */
  run: GraphExecutionMeta | undefined;
}

/** The workflow's schedules: the ones it has, adding one at a quick pick or
 *  at a time of day, and removing one. A new schedule runs with the inputs
 *  of the run on show. */
export function useSchedulePanel({ graphID, agent, run }: Args) {
  const queryClient = useQueryClient();
  const [time, setTime] = useState(DEFAULT_CUSTOM_TIME);
  const [error, setError] = useState<string | null>(null);
  const listKey = getGetV1ListExecutionSchedulesForAGraphQueryKey(graphID);

  const schedules =
    useGetV1ListExecutionSchedulesForAGraph(graphID, {
      query: { select: okData },
    }).data ?? [];

  function refresh() {
    return queryClient.invalidateQueries({ queryKey: listKey });
  }

  const create = usePostV1CreateExecutionSchedule({
    mutation: {
      onSuccess: () => {
        setError(null);
        refresh();
      },
      onError: (failure) =>
        setError(
          failure instanceof Error
            ? failure.message
            : "The schedule could not be saved.",
        ),
    },
  });
  const remove = useDeleteV1DeleteExecutionSchedule({
    mutation: { onSuccess: () => refresh() },
  });

  function add(cron: string, label: string) {
    create.mutate({
      graphId: graphID,
      data: {
        name: `${agent?.name ?? "Workflow"}: ${label}`,
        cron,
        graph_version: agent?.graph_version ?? run?.graph_version,
        inputs: run?.inputs ?? {},
        credentials: run?.credential_inputs ?? {},
        timezone: browserTimezone(),
      },
    });
  }

  function addAtTime() {
    const cron = dailyCron(time);
    if (!cron) {
      setError("Pick a time of day, like 09:00.");
      return;
    }
    add(cron, `every day ${time}`);
  }

  return {
    schedules,
    time,
    setTime,
    error,
    isSaving: create.isPending,
    removingID: remove.isPending ? remove.variables?.scheduleId : null,
    add,
    addAtTime,
    remove: (scheduleID: string) => remove.mutate({ scheduleId: scheduleID }),
  };
}
