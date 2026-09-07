"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { EditScheduleModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/SelectedScheduleView/components/EditScheduleModal/EditScheduleModal";
import { Button } from "@/components/atoms/Button/Button";
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
          <Text variant="body-medium" tone="primary">
            {title}
          </Text>
          {lastRunLabel ? (
            <Text variant="small" as="span" tone="muted">
              {lastRunLabel}
            </Text>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          {expertId && workflows ? (
            <Button
              variant="secondary"
              size="xs"
              leadingIcon={PlusSignIcon}
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
        <Text variant="body" tone="muted" className="pt-4">
          No schedules yet. Workflows with a schedule will run {expertName}{" "}
          automatically and show up here.
        </Text>
      ) : visible.length === 0 ? (
        <Text variant="body" tone="muted" className="pt-4">
          No schedules match.
        </Text>
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
