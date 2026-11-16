"use client";

import { HIDDEN_FROM_REPLAY } from "@/lib/replay-privacy";
import { cn } from "@/lib/utils";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import { useState } from "react";
import { firstTaskCorrection } from "../helpers";

interface Props {
  firstTask: string;
  owner: string | null;
  isLeaving: boolean;
  onConfirm: (corrections: string | null) => void;
}

// The first thing the team starts on, as Otto put it, for the person to
// start or change.
export function FirstTaskEnding({
  firstTask,
  owner,
  isLeaving,
  onConfirm,
}: Props) {
  const [isEditing, setIsEditing] = useState(false);
  const [task, setTask] = useState(firstTask);

  function handleStart() {
    onConfirm(firstTaskCorrection(firstTask, task));
  }

  return (
    <section
      aria-label="Your first task"
      className={cn(
        HIDDEN_FROM_REPLAY,
        "flex w-full max-w-md flex-col gap-4 rounded-large border border-border-base bg-surface p-5",
      )}
    >
      <Text variant="small-medium" tone="muted">
        First, your team starts on
      </Text>
      {isEditing ? (
        <Textarea
          aria-label="Your first task"
          value={task}
          maxLength={500}
          autoFocus
          onChange={(event) => setTask(event.target.value)}
        />
      ) : (
        <Text variant="large-medium">{task}</Text>
      )}
      {owner && (
        <Text variant="small" tone="muted">
          {owner} would take this on.
        </Text>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="primary"
          size="large"
          loading={isLeaving}
          disabled={isLeaving || !task.trim()}
          onClick={handleStart}
        >
          Start this
        </Button>
        {!isEditing && (
          <Button
            variant="ghost"
            size="large"
            disabled={isLeaving}
            onClick={() => setIsEditing(true)}
          >
            Change it
          </Button>
        )}
      </div>
    </section>
  );
}
