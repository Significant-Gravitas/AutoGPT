"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { EditScheduleModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/SelectedScheduleView/components/EditScheduleModal/EditScheduleModal";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { GraphScheduleListItem } from "@/components/contextual/SchedulesPanel/components/GraphScheduleListItem/GraphScheduleListItem";
import {
  Calendar03Icon,
  Clock01Icon,
  PlusSignIcon,
  RepeatIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SectionEmptyState } from "../../components/SectionEmptyState";
import { useState } from "react";
import {
  filterExpertSchedules,
  SCHEDULE_FILTERS,
  ScheduleFilter,
  ACTION_BUTTON_CLASS,
} from "../../helpers";
import { CreateScheduleDialog } from "./CreateScheduleDialog";
import { ToolbarSearch } from "../../components/ToolbarSearch";
import { PillFilterMenu } from "../../components/PillFilterMenu";

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
        <div className="flex min-w-0 flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-3">
          <Text variant="large-medium" tone="primary">
            {title}
          </Text>
          {lastRunLabel ? (
            <Text variant="small" as="span" tone="muted">
              {lastRunLabel}
            </Text>
          ) : null}
        </div>
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
          {expertId && workflows ? (
            <Button
              variant="secondary"
              size="small"
              textSize="body-md"
              leadingIcon={PlusSignIcon}
              onClick={() => setIsCreateOpen(true)}
            >
              Create schedule
            </Button>
          ) : null}
          <ToolbarSearch
            textSize="body-md"
            id="expert-schedules-search"
            label="Search schedules"
            value={query}
            onChange={setQuery}
          />
          <PillFilterMenu
            textSize="body-md"
            label="Filter schedules"
            value={filter}
            defaultValue="all"
            options={SCHEDULE_FILTERS}
            onChange={setFilter}
          />
        </div>
      </div>
      {schedules.length === 0 ? (
        <SectionEmptyState
          tiles={[
            // oxlint-disable-next-line design-system/no-raw-colors -- sky has no role; info blue would change the hue (deferred)
            <span key="clock" className="inline-flex text-sky-500">
              <Icon icon={Clock01Icon} size={22} />
            </span>,
            <span key="repeat" className="inline-flex text-success">
              <Icon icon={RepeatIcon} size={22} />
            </span>,
            <span key="calendar" className="inline-flex text-brand">
              <Icon icon={Calendar03Icon} size={22} />
            </span>,
          ]}
          title="No schedules yet"
          description={`Put a workflow on a schedule and ${expertName} runs it for you, on time, every time.`}
        >
          {expertId && workflows ? (
            <Button
              variant="primary"
              size="small"
              textSize="body-md"
              leadingIcon={PlusSignIcon}
              onClick={() => setIsCreateOpen(true)}
            >
              Create schedule
            </Button>
          ) : null}
        </SectionEmptyState>
      ) : visible.length === 0 ? (
        <Text variant="body-md" tone="muted" className="mt-4">
          No schedules match.
        </Text>
      ) : (
        <ul className="flex flex-col gap-3 pt-4" aria-label="Expert schedules">
          {visible.map((schedule) => (
            <li key={schedule.id}>
              <GraphScheduleListItem
                textSize="body-md"
                schedule={schedule}
                className="rounded-2xl border-0 smooth-shadow-ring-sm"
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
