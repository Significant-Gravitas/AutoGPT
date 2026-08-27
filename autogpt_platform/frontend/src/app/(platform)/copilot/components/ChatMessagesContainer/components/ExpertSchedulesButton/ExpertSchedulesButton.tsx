"use client";

import { EditScheduleModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/SelectedScheduleView/components/EditScheduleModal/EditScheduleModal";
import { Button } from "@/components/atoms/Button/Button";
import { GraphScheduleListItem } from "@/components/contextual/SchedulesPanel/components/GraphScheduleListItem/GraphScheduleListItem";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { useExpertSchedulesButton } from "./useExpertSchedulesButton";

interface Props {
  expertId: string;
  expertName: string;
}

export function ExpertSchedulesButton({ expertId, expertName }: Props) {
  const { isOpen, setIsOpen, schedules } = useExpertSchedulesButton(expertId);

  if (schedules.length === 0) return null;

  return (
    <>
      <Button
        variant="secondary"
        size="small"
        className="ml-auto shrink-0"
        data-testid="expert-schedules-button"
        onClick={() => setIsOpen(true)}
      >
        {schedules.length} {schedules.length === 1 ? "workflow" : "workflows"}{" "}
        scheduled
      </Button>
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
