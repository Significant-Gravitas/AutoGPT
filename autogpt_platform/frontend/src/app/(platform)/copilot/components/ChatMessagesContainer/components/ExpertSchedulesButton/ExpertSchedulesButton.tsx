"use client";

import { EditScheduleModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/SelectedScheduleView/components/EditScheduleModal/EditScheduleModal";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { GraphScheduleListItem } from "@/components/contextual/SchedulesPanel/components/GraphScheduleListItem/GraphScheduleListItem";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Clock01Icon } from "@hugeicons/core-free-icons";
import { useExpertSchedulesButton } from "./useExpertSchedulesButton";

interface Props {
  expertId: string;
  expertName: string;
}

export function ExpertSchedulesButton({ expertId, expertName }: Props) {
  const { isOpen, setIsOpen, schedules } = useExpertSchedulesButton(expertId);

  if (schedules.length === 0) return null;

  const scheduledLabel = `${schedules.length === 1 ? "workflow" : "workflows"} scheduled`;

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            data-testid="expert-schedules-button"
            onClick={() => setIsOpen(true)}
            className="flex shrink-0 items-center gap-1 rounded-full px-1.5 py-0.5 text-xs font-medium tabular-nums text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
          >
            <Icon icon={Clock01Icon} className="size-3.5" />
            {schedules.length}
            <span className="sr-only">{` ${scheduledLabel}`}</span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          {schedules.length} {scheduledLabel}
        </TooltipContent>
      </Tooltip>
      <Sheet open={isOpen} onOpenChange={setIsOpen}>
        <SheetContent
          side="right"
          className="w-full overflow-y-auto sm:max-w-xl"
        >
          <SheetTitle>{expertName}&apos;s scheduled workflows</SheetTitle>
          <ul
            className="mt-6 flex flex-col gap-3"
            aria-label="Expert schedules"
          >
            {schedules.map((schedule) => (
              <li key={schedule.id}>
                <GraphScheduleListItem
                  schedule={schedule}
                  className="sm:flex-col sm:items-stretch sm:justify-start"
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
        </SheetContent>
      </Sheet>
    </>
  );
}
