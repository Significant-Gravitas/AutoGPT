"use client";

import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { EditScheduleModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/SelectedScheduleView/components/EditScheduleModal/EditScheduleModal";
import { GraphScheduleListItem } from "@/components/contextual/SchedulesPanel/components/GraphScheduleListItem/GraphScheduleListItem";

interface Props {
  expertName: string;
  schedules: GraphExecutionJobInfo[];
  lastRunLabel: string | null;
}

export function ExpertSchedulesSection({
  expertName,
  schedules,
  lastRunLabel,
}: Props) {
  return (
    <section>
      <div className="mb-2.5 flex items-center justify-end gap-2 empty:mb-0">
        {lastRunLabel ? (
          <span className="text-xs text-zinc-500">{lastRunLabel}</span>
        ) : null}
      </div>
      {schedules.length === 0 ? (
        <p className="text-sm text-zinc-500">
          No schedules yet. Workflows with a schedule will run {expertName}{" "}
          automatically and show up here.
        </p>
      ) : (
        <ul className="flex flex-col gap-3" aria-label="Expert schedules">
          {schedules.map((schedule) => (
            <li key={schedule.id}>
              <GraphScheduleListItem
                schedule={schedule}
                editAction={
                  <EditScheduleModal
                    graphId={schedule.graph_id}
                    schedule={schedule}
                    triggerClassName="shrink-0"
                  />
                }
              />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
