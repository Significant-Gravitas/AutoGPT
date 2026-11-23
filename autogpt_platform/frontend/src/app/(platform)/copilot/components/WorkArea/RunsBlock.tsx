"use client";

import { chatHref } from "@/app/(platform)/tasks/helpers";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskArtifact } from "@/app/api/__generated__/models/taskArtifact";
import type { TaskRun } from "@/app/api/__generated__/models/taskRun";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { useNowMinute } from "@/hooks/useNowMinute";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import { Fragment, useState } from "react";
import {
  InfoGroup,
  InfoList,
  InfoMeta,
} from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockRow } from "./blocks";
import { RunResult } from "./RunResult";
import { documentOf } from "./taskBlocks";
import { runTitle, runViewOf, showsRuns } from "./taskRuns";

interface Props {
  task: Task;
  runs: TaskRun[];
  artifacts: TaskArtifact[];
  onOpenArtifact: (artifact: TaskArtifact) => void;
}

function keyOf(run: TaskRun) {
  return `${run.chat_session_id}-${String(run.started_at)}`;
}

/** Each run of a repeating task, newest first: when it ran (and what
 *  started it, for an event) and how it ended, a failed one with a red
 *  mark; the header says when several failed in a row. A click opens the run in
 *  place: its result, the document it wrote and a way into its chat. */
export function RunsBlock({ task, runs, artifacts, onOpenArtifact }: Props) {
  const isShown = showsRuns(task, runs);
  const now = useNowMinute(isShown);
  const [openKey, setOpenKey] = useState<string | null>(null);
  if (!isShown) return null;
  const waiting = task.events_waiting ?? 0;
  const failedInARow = task.repeat?.failed_in_a_row ?? 0;

  return (
    <InfoGroup
      label="Runs"
      count={runs.length}
      action={
        failedInARow > 0 ? (
          <Text variant="small" as="span" tone="error">
            {failedInARow} failed in a row
          </Text>
        ) : null
      }
    >
      <InfoList>
        {waiting > 0 ? (
          <BlockRow
            tile={<Icon icon={Clock01Icon} size={14} tone="info" />}
            title={`${waiting} more in line`}
            subtitle="They run one after another, oldest first"
            trailing={<InfoMeta>Waiting</InfoMeta>}
          />
        ) : null}
        {runs.map((run, index) => {
          const key = keyOf(run);
          const isOpen = openKey === key;
          const isNewest = index === 0;
          const view = runViewOf(run, isNewest, task);
          const short = run.failure_reason ?? run.result ?? null;
          // The task holds the newest run's result in full; older runs keep
          // only the short one.
          const full =
            isNewest && run.outcome
              ? (task.failure_reason ?? task.result ?? short)
              : short;
          return (
            <Fragment key={key}>
              <BlockRow
                tile={
                  <Icon
                    icon={view.icon}
                    size={14}
                    tone={view.tone}
                    spin={view.spin}
                  />
                }
                title={runTitle(run, now)}
                subtitle={isOpen ? null : short}
                trailing={
                  <span className="flex shrink-0 items-center gap-1">
                    <InfoMeta>{view.word}</InfoMeta>
                    <span
                      aria-hidden
                      className={cn(
                        "flex text-foreground-faint transition-transform duration-150 ease-out motion-reduce:transition-none",
                        isOpen && "rotate-180",
                      )}
                    >
                      <Icon icon={ArrowDown01Icon} size={14} tone="inherit" />
                    </span>
                  </span>
                }
                onClick={() => setOpenKey(isOpen ? null : key)}
              />
              {isOpen ? (
                <RunResult
                  text={full}
                  isFailure={!!run.failure_reason}
                  document={
                    isNewest && !run.failure_reason
                      ? documentOf(artifacts)
                      : null
                  }
                  chatHref={chatHref(run.chat_session_id)}
                  onOpenArtifact={onOpenArtifact}
                />
              ) : null}
            </Fragment>
          );
        })}
      </InfoList>
    </InfoGroup>
  );
}
