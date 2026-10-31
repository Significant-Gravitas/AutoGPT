"use client";

import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { tourStartingPoint } from "@/lib/tour-targets";
import { cn } from "@/lib/utils";
import { ArrowRight01Icon, Cancel01Icon } from "@hugeicons/core-free-icons";
import { motion, useReducedMotion } from "framer-motion";
import NextLink from "next/link";
import { useId } from "react";
import {
  getChatHref,
  getOwnerName,
  getTaskState,
  TASK_CONTROL_ATTRIBUTE,
  type TaskOwner,
} from "../helpers";
import { OwnerAvatar, OwnerLabel } from "./TaskOwner";

interface Props {
  task: SuggestedTask;
  owner: TaskOwner;
  isStartHere: boolean;
  /** The "more tasks" rows are one line: no why. */
  showWhy: boolean;
  /** Starting a task is off: another is starting, or the composer is locked. */
  isBusy: boolean;
  isSending: boolean;
  /** Removing is off while a task is starting. */
  isRemoveDisabled: boolean;
  onSend: (task: SuggestedTask) => void;
  onRemove: (task: SuggestedTask) => void;
}

const ROW_CLASSES =
  "flex min-w-0 flex-1 items-start gap-3 py-3 pl-4 pr-2 text-left outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus sm:items-center";

export function TaskRow({
  task,
  owner,
  isStartHere,
  showWhy,
  isBusy,
  isSending,
  isRemoveDisabled,
  onSend,
  onRemove,
}: Props) {
  const prefersReducedMotion = useReducedMotion();
  const pillID = useId();
  const whyID = useId();
  const name = getOwnerName(owner);
  const isSent = getTaskState(task) === "sent" && Boolean(task.session_id);
  const hasWhy = showWhy && Boolean(task.why);
  // What the row's own label leaves out: the pill and the why.
  const describedIDs: string[] = [];
  if (isStartHere) describedIDs.push(pillID);
  if (hasWhy) describedIDs.push(whyID);
  const describedBy = describedIDs.join(" ") || undefined;
  const control = { [TASK_CONTROL_ATTRIBUTE]: task.id };

  return (
    <motion.li
      exit={prefersReducedMotion ? undefined : { height: 0, opacity: 0 }}
      transition={{ duration: 0.15 }}
      className={cn(
        "group/row flex items-start transition-colors duration-150 sm:items-center",
        isStartHere ? "bg-accent-surface" : "hover:bg-surface-muted",
      )}
      data-testid="suggested-task"
    >
      {isSent ? (
        <NextLink
          href={getChatHref(task.session_id ?? "")}
          aria-label={`Open the chat with ${name}: ${task.title}`}
          className={ROW_CLASSES}
          {...control}
        >
          <OwnerAvatar owner={owner} />
          <RowText task={task} showWhy={showWhy} isMuted>
            <Text
              variant="small"
              as="span"
              tone="success-strong"
              className="sm:whitespace-nowrap"
            >
              In progress with {name}
            </Text>
          </RowText>
        </NextLink>
      ) : (
        <button
          type="button"
          {...tourStartingPoint()}
          {...control}
          aria-label={`Start with ${name}: ${task.title}`}
          aria-describedby={describedBy}
          aria-busy={isSending || undefined}
          disabled={isBusy}
          onClick={() => onSend(task)}
          className={cn(
            ROW_CLASSES,
            "cursor-pointer disabled:cursor-default disabled:opacity-60",
          )}
        >
          <OwnerAvatar owner={owner} />
          <RowText
            task={task}
            showWhy={showWhy}
            isStartHere={isStartHere}
            pillID={pillID}
            whyID={whyID}
          >
            <OwnerLabel owner={owner} />
          </RowText>
          <span
            className={cn(
              "shrink-0 self-center text-foreground-tertiary",
              isSending ? "flex" : "hidden sm:flex",
            )}
          >
            {isSending ? (
              <LoadingSpinner size="small" />
            ) : (
              <Icon icon={ArrowRight01Icon} size={15} />
            )}
          </span>
        </button>
      )}
      {!isSent ? (
        // Shown on hover or focus, and always on touch screens.
        <span className="mt-3 mr-2 flex shrink-0 opacity-0 transition-opacity duration-150 group-hover/row:opacity-100 focus-within:opacity-100 sm:mt-0 [@media(hover:none)]:opacity-100">
          <Button
            type="button"
            variant="toggle"
            size="icon-xs"
            withTooltip={false}
            aria-label={`Remove: ${task.title}`}
            disabled={isRemoveDisabled}
            onClick={() => onRemove(task)}
          >
            <Icon icon={Cancel01Icon} size={14} />
          </Button>
        </span>
      ) : (
        // Keeps the titles of sent rows aligned with the rest.
        <span aria-hidden className="mr-2 size-6 shrink-0" />
      )}
    </motion.li>
  );
}

interface RowTextProps {
  task: SuggestedTask;
  showWhy: boolean;
  isStartHere?: boolean;
  isMuted?: boolean;
  pillID?: string;
  whyID?: string;
  children: React.ReactNode;
}

// The title (with "Start here" in its flow) and the why, then the owner or
// the state: right-aligned on a wide row, under the title on a phone.
function RowText({
  task,
  showWhy,
  isStartHere,
  isMuted,
  pillID,
  whyID,
  children,
}: RowTextProps) {
  return (
    <span className="flex min-w-0 flex-1 flex-col gap-1 sm:flex-row sm:items-center sm:gap-3">
      <span className="flex min-w-0 flex-1 flex-col">
        <Text
          variant="body-medium"
          as="span"
          tone={isMuted ? "secondary" : "primary"}
          className="text-pretty"
        >
          {isStartHere ? (
            <span
              id={pillID}
              className="mr-2 inline-block -translate-y-px rounded-full bg-accent-strong/10 px-2 py-0.5 align-middle text-caption-sm font-semibold tracking-eyebrow text-accent-strong uppercase"
            >
              Start here
            </span>
          ) : null}
          {task.title}
        </Text>
        {showWhy && task.why ? (
          <Text
            variant="small"
            as="span"
            tone="secondary"
            id={whyID}
            className="mt-0.5"
          >
            {task.why}
          </Text>
        ) : null}
      </span>
      <span className="shrink-0 sm:ml-auto">{children}</span>
    </span>
  );
}
