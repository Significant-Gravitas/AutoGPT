"use client";

import {
  actorName,
  subtaskTitle,
} from "@/app/(platform)/tasks/components/TaskPanel/components/TaskRecordView/helpers";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskActivity } from "@/app/api/__generated__/models/taskActivity";
import { cn } from "@/lib/utils";
import { useEffect, useId, useState } from "react";
import {
  CARD,
  InfoGroup,
} from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { type TrailStep, trailOf } from "./trailSteps";
import { TrailStepRow } from "./TrailStepRow";

const SECOND = 1000;

interface Props {
  task: Task;
  subtasks: Task[];
  activity: TaskActivity[];
  experts: Expert[];
}

/** What happened, as one trail: each step the worker named, the one it is
 *  on turning with a live timer, and the tool calls made inside a step
 *  branching off under it. The working step is open; any other opens on a
 *  click. */
export function ActivityTrail({ task, subtasks, activity, experts }: Props) {
  const steps = trailOf(task, activity);
  const panelID = useId();
  const [toggled, setToggled] = useState<Record<string, boolean>>({});
  const now = useNow(steps.some((step) => step.state === "running"));
  if (steps.length === 0) return null;

  function toggleKey(step: TrailStep) {
    return `${step.id}:${step.state}`;
  }

  function isOpen(step: TrailStep | undefined) {
    if (!step || step.lines.length === 0) return false;
    return toggled[toggleKey(step)] ?? step.state === "running";
  }

  function hintOf(step: TrailStep) {
    const who = step.line ? actorName(step.line, experts) : null;
    const part = step.line
      ? subtaskTitle(step.line.task_id, task, subtasks)
      : null;
    return [step.title, who, part].filter(Boolean).join(" · ");
  }

  return (
    <InfoGroup
      label="Activity"
      count={steps.filter((step) => step.state !== "next").length}
    >
      <ol
        aria-label="Activity"
        className={cn(CARD, "flex flex-col px-1 pt-1 pb-1.5")}
      >
        {steps.map((step, index) => {
          const previous = steps[index - 1];
          const next = steps[index + 1];
          return (
            <TrailStepRow
              key={step.id}
              step={step}
              panelID={`${panelID}-${index}`}
              hint={hintOf(step)}
              now={now}
              isOpen={isOpen(step)}
              hasPrevious={!!previous}
              hasNext={!!next}
              isWalkedIn={previous?.state === "done" && step.state !== "next"}
              isWalkedOut={
                step.state === "done" && !!next && next.state !== "next"
              }
              isAfterBranch={isOpen(previous)}
              onToggle={() =>
                setToggled({ ...toggled, [toggleKey(step)]: !isOpen(step) })
              }
            />
          );
        })}
      </ol>
    </InfoGroup>
  );
}

/** The time now, once a second, only while a step's timer needs it. */
function useNow(isActive: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!isActive) return;
    const timer = setInterval(() => setNow(Date.now()), SECOND);
    return () => clearInterval(timer);
  }, [isActive]);
  return now;
}
