"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { EditScheduleModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/SelectedScheduleView/components/EditScheduleModal/EditScheduleModal";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { GraphScheduleListItem } from "@/components/contextual/SchedulesPanel/components/GraphScheduleListItem/GraphScheduleListItem";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import {
  filterExpertSchedules,
  SCHEDULE_FILTERS,
  ScheduleFilter,
  ACTION_BUTTON_CLASS,
} from "../../helpers";
import { CreateScheduleDialog } from "./CreateScheduleDialog";
import { FilterIconMenu } from "./FilterIconMenu";

interface Props {
  title: string;
  expertName: string;
  expertId?: string;
  workflows?: ExpertWorkflowRef[];
  accentClassName?: string;
  schedules: GraphExecutionJobInfo[];
  lastRunLabel: string | null;
}

export function ExpertSchedulesSection({
  title,
  expertName,
  expertId,
  workflows,
  accentClassName,
  schedules,
  lastRunLabel,
}: Props) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<ScheduleFilter>("all");
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const visible = filterExpertSchedules(schedules, query, filter);

  return (
    <section>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-baseline gap-3">
          <Text variant="large-medium">{title}</Text>
          {lastRunLabel ? (
            <span className="text-xs text-zinc-500">{lastRunLabel}</span>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          {expertId && workflows ? (
            <Button
              variant="secondary"
              size="small"
              className={ACTION_BUTTON_CLASS}
              leftIcon={<Icon icon={PlusSignIcon} size={14} />}
              onClick={() => setIsCreateOpen(true)}
            >
              Create schedule
            </Button>
          ) : null}
          <SearchInput
            size="xsmall"
            value={query}
            onChange={setQuery}
            placeholder="Search schedules"
            className="w-48"
          />
          <FilterIconMenu
            label="Filter schedules"
            value={filter}
            defaultValue="all"
            options={SCHEDULE_FILTERS}
            onChange={setFilter}
          />
        </div>
      </div>
      {schedules.length === 0 ? (
        <p className="pt-4 text-sm text-zinc-500">
          No schedules yet. Workflows with a schedule will run {expertName}{" "}
          automatically and show up here.
        </p>
      ) : visible.length === 0 ? (
        <p className="pt-4 text-sm text-zinc-500">No schedules match.</p>
      ) : (
        <ul className="flex flex-col gap-3 pt-4" aria-label="Expert schedules">
          {visible.map((schedule) => (
            <li key={schedule.id}>
              <GraphScheduleListItem
                schedule={schedule}
                iconClassName={accentClassName}
                actionClassName={ACTION_BUTTON_CLASS}
                editAction={
                  <EditScheduleModal
                    graphId={schedule.graph_id}
                    schedule={schedule}
                    triggerClassName={`${ACTION_BUTTON_CLASS} shrink-0`}
                  />
                }
              />
            </li>
          ))}
        </ul>
      )}
      {expertId && workflows ? (
        <CreateScheduleDialog
          expertId={expertId}
          workflows={workflows}
          open={isCreateOpen}
          onClose={() => setIsCreateOpen(false)}
        />
      ) : null}
    </section>
  );
}
