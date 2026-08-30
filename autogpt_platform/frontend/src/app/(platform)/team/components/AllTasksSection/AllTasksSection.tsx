"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { DelegatedTasksBoard } from "./components/DelegatedTasksBoard";
import { ExpertRunsBoard } from "./components/ExpertRunsBoard";

interface Props {
  experts: Expert[];
  enabled: boolean;
}

/** Forks per flag so only the active board's queries ever fire: the spine
 *  board is one `/api/tasks` read, the legacy board fans out one
 *  `list_expert_runs` call per hired expert. Rides the experts flag — the
 *  fail-closed default keeps LD-less environments on the legacy board. */
export function AllTasksSection({ experts, enabled }: Props) {
  const isTaskSpineEnabled = useGetFlag(Flag.HIRE_EXPERTS);

  if (isTaskSpineEnabled) {
    return <DelegatedTasksBoard enabled={enabled} />;
  }
  return <ExpertRunsBoard experts={experts} enabled={enabled} />;
}
