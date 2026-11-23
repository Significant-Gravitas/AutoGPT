"use client";

import { useMemo } from "react";

import {
  useGetV1ListExecutionSchedulesForAUser,
  useListUpcomingScheduleRuns,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";

import { buildSchedules } from "./helpers";
import { TASK_LIST_PARAMS } from "./scheduleCache";
import { useMinuteClock } from "./useMinuteClock";
import { useScheduleTabs } from "./useScheduleTabs";

/** Every schedule the user has: repeating, watching and once-only tasks, and
 *  workflow schedules, with their coming run times. */
export function useSchedulesPage() {
  // "Now" for how long a task at work has been running: a minute's clock,
  // resting while the tab is hidden, so it never freezes at page load.
  const now = useMinuteClock();
  const tabs = useScheduleTabs();

  const tasksQuery = useListTasks(TASK_LIST_PARAMS, {
    query: { select: okData },
  });
  const workflowsQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: okData },
  });
  const upcomingQuery = useListUpcomingScheduleRuns(undefined, {
    query: { select: okData },
  });

  const tasks = tasksQuery.data?.tasks;
  const workflows = workflowsQuery.data;
  const upcoming = upcomingQuery.data?.schedules;
  // The Affect UI block keeps its own copy of the list and resets it whenever
  // the array changes, so it has to be the same array until the data does.
  const schedules = useMemo(
    () =>
      buildSchedules({
        tasks: tasks ?? [],
        workflows: workflows ?? [],
        upcoming: upcoming ?? [],
        now,
      }),
    [tasks, workflows, upcoming, now],
  );

  // The coming times only add to the list: it shows without them, each
  // schedule with its next run, and fills in when they arrive.
  const isLoading = tasksQuery.isLoading || workflowsQuery.isLoading;
  const error = tasksQuery.error ?? workflowsQuery.error ?? null;

  return {
    schedules,
    tasks: tasks ?? [],
    workflows: workflows ?? [],
    tabs,
    isLoading,
    error,
    isEmpty: !isLoading && !error && schedules.length === 0,
    retry: () => {
      tasksQuery.refetch();
      workflowsQuery.refetch();
      upcomingQuery.refetch();
    },
  };
}
