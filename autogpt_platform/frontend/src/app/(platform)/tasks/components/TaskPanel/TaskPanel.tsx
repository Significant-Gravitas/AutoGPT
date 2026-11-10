"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { OwnerChip } from "../OwnerChip/OwnerChip";
import { StatusPill } from "../StatusPill/StatusPill";
import { RenameField } from "../TaskMenu/components/RenameField/RenameField";
import { TaskMenu } from "../TaskMenu/TaskMenu";
import { PanelSection } from "./components/PanelSection/PanelSection";
import { PanelSkeleton } from "./components/PanelSkeleton/PanelSkeleton";
import { TaskChats } from "./components/TaskChats/TaskChats";
import { TaskTimes } from "./components/TaskTimes/TaskTimes";
import { PANEL_TITLE_ID, useTaskPanel } from "./useTaskPanel";

interface Props {
  taskID: string;
  experts: Expert[];
  onArchived: () => void;
}

/** Top to bottom: the title with its status and owner, what was asked,
 *  when it ran, and the chats it was worked on in. */
export function TaskPanel({ taskID, experts, onArchived }: Props) {
  const panel = useTaskPanel(taskID, experts);
  const { task, owner } = panel;

  if (panel.isLoading) return <PanelSkeleton />;
  if (panel.isError || !task || !owner) {
    return (
      <div className="p-4">
        <ErrorCard
          context="task"
          httpError={{ message: "This task could not be loaded." }}
          onRetry={panel.retry}
        />
      </div>
    );
  }

  return (
    <div className="flex animate-inbox-enter flex-col divide-y divide-border-hairline">
      <div className="flex flex-col gap-3 px-1.5 py-5">
        <div className="flex min-w-0 items-start gap-2">
          {panel.isRenaming ? (
            <RenameField
              taskID={task.id}
              title={task.title}
              onDone={panel.stopRename}
            />
          ) : (
            <Text
              id={PANEL_TITLE_ID}
              variant="large-medium"
              as="h2"
              tone="ink"
              tabIndex={-1}
              className="min-w-0 flex-1 text-pretty break-words"
            >
              {task.title}
            </Text>
          )}
          <TaskMenu
            task={task}
            onRename={panel.startRename}
            onArchived={onArchived}
            className="shrink-0"
          />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <StatusPill status={task.status} />
          <span className="inline-flex h-6 items-center rounded-control bg-surface-wash px-2">
            <OwnerChip owner={owner} />
          </span>
        </div>
      </div>
      <PanelSection label="Description">
        <Text
          variant="inbox"
          tone={task.description ? "text" : "faint"}
          className="text-pretty break-words whitespace-pre-wrap"
        >
          {task.description || "No description"}
        </Text>
      </PanelSection>
      <PanelSection label="Timeline">
        <TaskTimes task={task} />
      </PanelSection>
      <PanelSection label="Chats" meta={task.session_ids?.length || null}>
        <TaskChats sessionIDs={task.session_ids ?? []} />
      </PanelSection>
    </div>
  );
}
