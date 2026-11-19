"use client";

import { StateIcon } from "@/app/(platform)/tasks/components/StateIcon/StateIcon";
import {
  answeredAsks,
  openAsks,
} from "@/app/(platform)/tasks/components/TaskPanel/components/TaskRecordView/helpers";
import { chatHref, ownerOf } from "@/app/(platform)/tasks/helpers";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { TaskArtifact } from "@/app/api/__generated__/models/taskArtifact";
import type { TaskRecord } from "@/app/api/__generated__/models/taskRecord";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowRight01Icon,
  MessageQuestionIcon,
} from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import {
  InfoGroup,
  InfoList,
  InfoMeta,
} from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { ActivityTrail } from "./ActivityTrail";
import { BlockNote, BlockRow, NeedsYouTag } from "./blocks";
import { RunsBlock } from "./RunsBlock";
import { ArtifactsBlock, EvidenceBlock, ResultBlock } from "./TaskTabBlocks";

interface Props {
  record: TaskRecord;
  experts: Expert[];
  onOpenArtifact: (artifact: TaskArtifact) => void;
}

/** What only the task's full record holds, under its details: its parts,
 *  the outcome, its runs, what happened, what it made, the proof and what
 *  you decided. A block with nothing in it is left out. */
export function TaskRecordBlocks({ record, experts, onOpenArtifact }: Props) {
  const { task } = record;
  const subtasks = record.subtasks ?? [];
  const asks = record.asks ?? [];
  const decided = answeredAsks(asks);
  const artifacts = record.artifacts ?? [];

  return (
    <>
      {subtasks.length > 0 ? (
        <InfoGroup
          label="Subtasks"
          action={
            <InfoMeta>
              {task.subtasks_done ?? 0} of {subtasks.length} done
            </InfoMeta>
          }
        >
          <InfoList>
            {subtasks.map((subtask) => (
              <BlockRow
                key={subtask.id}
                tile={<StateIcon status={subtask.status} />}
                title={subtask.title}
                subtitle={subtask.status_text}
                trailing={<InfoMeta>{ownerOf(subtask, experts).name}</InfoMeta>}
              />
            ))}
          </InfoList>
        </InfoGroup>
      ) : null}

      <ResultBlock
        task={task}
        artifacts={artifacts}
        isUnverified={!!record.is_unverified}
        onOpenArtifact={onOpenArtifact}
      />

      <RunsBlock
        task={task}
        runs={record.runs ?? []}
        artifacts={artifacts}
        onOpenArtifact={onOpenArtifact}
      />

      <ActivityTrail
        task={task}
        subtasks={subtasks}
        activity={record.activity ?? []}
        experts={experts}
      />

      <ArtifactsBlock artifacts={artifacts} onOpenArtifact={onOpenArtifact} />

      <EvidenceBlock
        evidence={record.evidence ?? []}
        artifacts={artifacts}
        onOpenArtifact={onOpenArtifact}
      />

      {decided.length > 0 ? (
        <InfoGroup label="What you decided" count={decided.length}>
          <InfoList>
            {decided.map((ask) => (
              <BlockRow
                key={ask.id}
                tile={
                  <Icon icon={MessageQuestionIcon} size={14} tone="inherit" />
                }
                title={ask.answer ?? ""}
                subtitle={ask.question}
                wrap
              />
            ))}
          </InfoList>
        </InfoGroup>
      ) : null}
    </>
  );
}

/** The questions the task waits on you for, each with a way to answer. */
export function NeedsYouBlock({ asks }: { asks: TaskRecord["asks"] }) {
  const waiting = openAsks(asks ?? []);
  if (waiting.length === 0) return null;
  return (
    <InfoGroup label="Needs you" count={waiting.length}>
      {waiting.map((ask) => (
        <BlockNote key={ask.id}>
          <NeedsYouTag>Waiting on you</NeedsYouTag>
          <Text variant="inbox" tone="ink" className="text-pretty">
            {ask.question}
          </Text>
          {ask.chat_session_id ? (
            <NextLink
              href={chatHref(ask.chat_session_id)}
              className="mt-1 flex h-7 w-fit items-center gap-1 rounded-control bg-foreground-ink px-2.5 text-body-sm font-medium text-surface outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox"
            >
              Answer
              <Icon icon={ArrowRight01Icon} size={13} tone="inherit" />
            </NextLink>
          ) : null}
        </BlockNote>
      ))}
    </InfoGroup>
  );
}
