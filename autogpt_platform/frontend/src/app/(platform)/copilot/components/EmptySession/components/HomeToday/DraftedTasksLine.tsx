"use client";

import { Button } from "@/components/atoms/Button/Button";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { useState } from "react";
import { useIsMobile } from "../../../../useIsMobile";
import { BRIEF_TAB_KEY, useWorkAreaStore } from "../../../../workArea";
import { briefName } from "./helpers";
import { useTaskSuggestions } from "./useTaskSuggestions";

/** One quiet line under Home's composer while Otto has drafted tasks and
 *  the brief is not in front; it opens the brief. */
export function DraftedTasksLine() {
  const [now] = useState(() => Date.now());
  const { suggestions } = useTaskSuggestions();
  const isMobile = useIsMobile();
  const isBriefShown = useWorkAreaStore(
    (s) => s.activeKey === BRIEF_TAB_KEY && !s.isFolded,
  );
  const openBrief = useWorkAreaStore((s) => s.openBrief);
  const count = suggestions.length;
  if (isMobile || isBriefShown || count === 0) return null;

  return (
    <div className="mt-3 flex justify-center">
      <Button
        variant="ghost-muted"
        size="xs"
        onClick={() => openBrief(briefName(now))}
      >
        {AUTOPILOT_NAME} drafted {count === 1 ? "a task" : `${count} tasks`} for
        you
      </Button>
    </div>
  );
}
