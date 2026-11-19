"use client";

import { useRouter } from "next/navigation";

import { AgentSchedulesScreen } from "./affect/agent-schedules";
import { ScheduleSidePanel } from "./components/ScheduleSidePanel/ScheduleSidePanel";
import { useScheduleDetail } from "./components/ScheduleSidePanel/useScheduleDetail";
import { askOttoHref, ERROR_LABELS, EXAMPLE_SCHEDULE, LABELS } from "./helpers";
import { useScheduleActions } from "./useScheduleActions";
import { useSchedulesPage } from "./useSchedulesPage";

/** Every schedule in one place: repeating, watching and once-only tasks, and
 *  workflow schedules, on the Affect UI Schedules block. */
export default function SchedulesPage() {
  const router = useRouter();
  const page = useSchedulesPage();
  const actions = useScheduleActions({
    tasks: page.tasks,
    workflows: page.workflows,
  });
  const detail = useScheduleDetail({
    row: page.tabs.isOpen ? page.tabs.active : null,
    schedules: page.schedules,
    tasks: page.tasks,
    workflows: page.workflows,
    onRemove: (row) => {
      page.tabs.close(row);
      actions.remove(row);
    },
    onOpen: (href) => router.push(href),
  });

  return (
    <div className="flex h-(--inset-height) min-h-0 w-full overflow-hidden">
      <div className="h-full min-w-0 flex-1">
        <AgentSchedulesScreen
          schedules={page.schedules}
          selected={page.tabs.isOpen ? page.tabs.active : null}
          onSelect={page.tabs.open}
          onPause={actions.setPaused}
          labels={page.error ? ERROR_LABELS : LABELS}
          loading={page.isLoading}
          empty={page.isEmpty || Boolean(page.error)}
          onEmptyAction={
            page.error
              ? page.retry
              : () => router.push(askOttoHref(EXAMPLE_SCHEDULE))
          }
        />
      </div>
      <ScheduleSidePanel
        tabs={page.tabs}
        schedules={page.schedules}
        detail={detail}
        onPause={actions.setPaused}
        onRunNow={actions.runNow}
      />
    </div>
  );
}
