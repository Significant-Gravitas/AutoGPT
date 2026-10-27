"use client";

import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  ArrowRight01Icon,
  Cancel01Icon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import NextLink from "next/link";
import {
  getChatHref,
  getKickoffLabel,
  getPromptIcon,
  getTaskState,
  trackDraftOpened,
} from "./helpers";

interface Props {
  tasks: SuggestedTask[];
  sendingId: string | null;
  disabled?: boolean;
  onSend: (task: SuggestedTask) => void;
  onHide: (task: SuggestedTask) => void;
  className?: string;
}

const ROW_CLASSES =
  "flex min-w-0 flex-1 items-center gap-3 px-4 py-3 text-left";

export function SuggestedTaskList({
  tasks,
  sendingId,
  disabled = false,
  onSend,
  onHide,
  className,
}: Props) {
  const prefersReducedMotion = useReducedMotion();
  const isBusy = disabled || sendingId !== null;

  return (
    <ol
      className={cn(
        "divide-y divide-separator overflow-hidden rounded-lg border border-border-base bg-surface",
        className,
      )}
    >
      <AnimatePresence initial={false}>
        {tasks.map((task) => (
          <motion.li
            key={task.id}
            exit={prefersReducedMotion ? undefined : { height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="group/row flex items-center transition-colors duration-150 hover:bg-surface-muted"
            data-testid="suggested-task"
          >
            <RowBody
              task={task}
              isSending={sendingId === task.id}
              isBusy={isBusy}
              onSend={onSend}
            />
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              withTooltip={false}
              aria-label={`Hide: ${task.title}`}
              disabled={sendingId !== null}
              onClick={() => onHide(task)}
              className="mr-2 shrink-0 text-foreground-subtle opacity-0 transition-opacity duration-150 hover:text-foreground-body focus-visible:opacity-100 group-hover/row:opacity-100 [@media(hover:none)]:opacity-100"
            >
              <Icon icon={Cancel01Icon} size={14} />
            </Button>
          </motion.li>
        ))}
      </AnimatePresence>
    </ol>
  );
}

interface RowBodyProps {
  task: SuggestedTask;
  isSending: boolean;
  isBusy: boolean;
  onSend: (task: SuggestedTask) => void;
}

function RowBody({ task, isSending, isBusy, onSend }: RowBodyProps) {
  const promptIcon = getPromptIcon(task.icon);

  if (task.kickoff && task.session_id) {
    return (
      <NextLink
        href={getChatHref(task.session_id)}
        onClick={() => trackDraftOpened(task)}
        className={ROW_CLASSES}
      >
        <Icon
          icon={promptIcon}
          size={15}
          className="shrink-0 text-foreground-subtle"
        />
        <Text variant="body-medium" tone="primary" className="text-pretty">
          {task.title}
        </Text>
        <Text
          variant="small-medium"
          className="ml-auto shrink-0 whitespace-nowrap rounded-full bg-accent-surface px-2 py-0.5 text-brand"
        >
          {getKickoffLabel(task)}
        </Text>
      </NextLink>
    );
  }

  if (getTaskState(task) === "sent" && task.session_id) {
    return (
      <NextLink href={getChatHref(task.session_id)} className={ROW_CLASSES}>
        <Icon
          icon={promptIcon}
          size={15}
          className="shrink-0 text-foreground-disabled"
        />
        <Text variant="body-medium" tone="muted" className="text-pretty">
          {task.title}
        </Text>
        <Text
          variant="small"
          tone="muted"
          className="ml-auto shrink-0 whitespace-nowrap"
        >
          In progress
        </Text>
      </NextLink>
    );
  }

  return (
    <button
      type="button"
      disabled={isBusy}
      onClick={() => onSend(task)}
      className={cn(
        ROW_CLASSES,
        "cursor-pointer disabled:cursor-default disabled:opacity-60",
      )}
    >
      <Icon
        icon={promptIcon}
        size={15}
        className="shrink-0 text-foreground-subtle"
      />
      <Text variant="body-medium" tone="primary" className="text-pretty">
        {task.title}
      </Text>
      <Icon
        icon={isSending ? Loading03Icon : ArrowRight01Icon}
        size={15}
        className={cn(
          "ml-auto shrink-0 text-foreground-subtle",
          isSending && "animate-spin",
        )}
        aria-label={isSending ? "Starting" : undefined}
      />
    </button>
  );
}
