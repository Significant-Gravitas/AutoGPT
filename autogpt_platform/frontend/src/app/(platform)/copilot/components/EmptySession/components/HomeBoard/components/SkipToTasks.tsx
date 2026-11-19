"use client";

import type { MouseEvent } from "react";
import { TASKS_REGION_ID } from "../useBoardShortcuts";

/** For keyboard users: straight to the tasks, past the page around them.
 *  It shows only when focused. */
export function SkipToTasks() {
  function skip(e: MouseEvent<HTMLAnchorElement>) {
    e.preventDefault();
    const region = document.getElementById(TASKS_REGION_ID);
    const first = region?.querySelector<HTMLElement>(
      "[data-task-card][tabindex='0']",
    );
    (first ?? region)?.focus();
  }

  return (
    <a
      href={`#${TASKS_REGION_ID}`}
      onClick={skip}
      className="sr-only rounded-lg bg-surface px-3 py-1.5 text-body-sm font-medium text-foreground-ink shadow-(--inbox-shadow-card) focus:not-sr-only focus:self-start focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
    >
      Skip to tasks
    </a>
  );
}
