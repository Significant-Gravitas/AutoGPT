"use client";

import { Text } from "@/components/atoms/Text/Text";
import { BriefButton } from "../../HomeToday/BriefButton";
import { BoardViewSwitch } from "../BoardViewSwitch";
import { CreateTask } from "../CreateTask";
import type { BoardView } from "../useBoardView";
import type { TaskLauncher } from "../useTaskLauncher";

/** Below this row width the brief button shows only its icon. */
const NARROW_BRIEF = "[@container(max-width:34rem)]:sr-only";

interface Props {
  view: BoardView;
  onView: (view: BoardView) => void;
  launcher: TaskLauncher;
}

/** "Tasks", the layout switch, the day's brief and Create task. On a narrow
 *  board (the sidebar open beside it) the brief button drops its label and
 *  the controls wrap under the heading; Create task is never cut off. */
export function BoardHeading({ view, onView, launcher }: Props) {
  return (
    <div className="@container flex flex-wrap items-center gap-2 px-1">
      <Text id="home-board-heading" variant="inbox-medium" as="h2" tone="ink">
        Tasks
      </Text>
      <span className="ml-auto flex flex-wrap items-center justify-end gap-2">
        <BoardViewSwitch value={view} onChange={onView} />
        <BriefButton labelClassName={NARROW_BRIEF} />
        <CreateTask
          onOpen={() => launcher.open()}
          buttonRef={launcher.openerRef}
        />
      </span>
    </div>
  );
}
