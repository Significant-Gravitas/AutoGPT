"use client";

import { StateIcon } from "@/app/(platform)/tasks/components/StateIcon/StateIcon";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { CARD } from "../../components/ToolChain/ResultCards";
import { useExpertMap } from "../../useExpertMap";
import { useWorkAreaStore } from "../../workArea";
import { taskStateWords, type TaskLine, type TaskToolView } from "./helpers";

interface Props {
  view: TaskToolView;
}

/** A task step opened up in the chat: each task's title, who has it and
 *  where it stands, with Open beside it; or one line saying what was done. */
export function TaskToolCard({ view }: Props) {
  if (view.kind === "archived") return <Note>Task archived.</Note>;
  if (view.kind === "finished") return <Note>{finishedWords(view)}</Note>;
  if (view.tasks.length === 0) return <Note>No tasks yet.</Note>;
  return <TaskLines tasks={view.tasks} />;
}

function finishedWords(view: Extract<TaskToolView, { kind: "finished" }>) {
  if (view.notDone !== null) {
    return view.notDone
      ? `Saved as not done: ${view.notDone}`
      : "Saved as not done.";
  }
  return view.document
    ? `Saved the result in ${view.document}.`
    : "Saved the result.";
}

function TaskLines({ tasks }: { tasks: TaskLine[] }) {
  const { expertsById } = useExpertMap();
  const openTask = useWorkAreaStore((s) => s.openTask);
  const [now] = useState(Date.now);

  function ownerName(task: TaskLine) {
    if (!task.ownerExpertID) return AUTOPILOT_NAME;
    return expertsById.get(task.ownerExpertID)?.name ?? "An expert";
  }

  return (
    <ul className={`${CARD} flex flex-col p-1`}>
      {tasks.map((task) => (
        <li
          key={task.id}
          className="flex min-h-11 items-center gap-2.5 rounded-lg px-2 py-1.5"
        >
          <StateIcon
            status={task.status}
            queued={task.isQueued && task.status === "IN_PROGRESS"}
            className="shrink-0"
          />
          <span className="flex min-w-0 flex-1 flex-col">
            <Text
              variant="inbox"
              as="span"
              tone="ink"
              dir="auto"
              className="truncate"
            >
              {task.title}
            </Text>
            <Text variant="small" as="span" tone="quiet" className="truncate">
              {`${ownerName(task)} · ${taskStateWords(task, now)}`}
            </Text>
          </span>
          <button
            type="button"
            onClick={() => openTask(task.id, task.title)}
            aria-label={`Open task ${task.title}`}
            className="inline-flex h-7 shrink-0 items-center gap-1 rounded-control px-2 transition-colors duration-100 ease-out hover:bg-surface-tray focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
          >
            <Text variant="inbox-medium" as="span" tone="ink">
              Open
            </Text>
            <Icon icon={ArrowRight01Icon} size={12} tone="quiet" aria-hidden />
          </button>
        </li>
      ))}
    </ul>
  );
}

function Note({ children }: { children: string }) {
  return (
    <div className={`${CARD} px-3 py-2.5`}>
      <Text variant="inbox" as="p" tone="text">
        {children}
      </Text>
    </div>
  );
}
