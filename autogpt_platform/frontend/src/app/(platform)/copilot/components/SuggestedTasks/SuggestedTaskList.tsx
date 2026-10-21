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
import { getChatHref, getPromptIcon, getTaskState } from "./helpers";

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
        "divide-y divide-zinc-100 overflow-hidden rounded-lg border border-zinc-200 bg-white",
        className,
      )}
    >
      <AnimatePresence initial={false}>
        {tasks.map((task) => {
          const promptIcon = getPromptIcon(task.icon);
          const isSent = getTaskState(task) === "sent";
          return (
            <motion.li
              key={task.id}
              exit={
                prefersReducedMotion ? undefined : { height: 0, opacity: 0 }
              }
              transition={{ duration: 0.15 }}
              className="group/row flex items-center transition-colors duration-150 hover:bg-zinc-50"
              data-testid="suggested-task"
            >
              {isSent && task.session_id ? (
                <NextLink
                  href={getChatHref(task.session_id)}
                  className={ROW_CLASSES}
                >
                  <Icon
                    icon={promptIcon}
                    size={15}
                    className="shrink-0 text-zinc-300"
                  />
                  <Text
                    variant="body-medium"
                    tone="muted"
                    className="text-pretty"
                  >
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
              ) : (
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
                    className="shrink-0 text-zinc-400"
                  />
                  <Text
                    variant="body-medium"
                    tone="primary"
                    className="text-pretty"
                  >
                    {task.title}
                  </Text>
                  {sendingId === task.id ? (
                    <Icon
                      icon={Loading03Icon}
                      size={15}
                      className="ml-auto shrink-0 animate-spin text-zinc-400"
                      aria-label="Starting"
                    />
                  ) : (
                    <Icon
                      icon={ArrowRight01Icon}
                      size={15}
                      className="ml-auto shrink-0 text-zinc-400"
                    />
                  )}
                </button>
              )}
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                withTooltip={false}
                aria-label={`Hide: ${task.title}`}
                disabled={sendingId !== null}
                onClick={() => onHide(task)}
                className="mr-2 shrink-0 text-zinc-400 opacity-0 transition-opacity duration-150 group-hover/row:opacity-100 hover:text-zinc-700 focus-visible:opacity-100 [@media(hover:none)]:opacity-100"
              >
                <Icon icon={Cancel01Icon} size={14} />
              </Button>
            </motion.li>
          );
        })}
      </AnimatePresence>
    </ol>
  );
}
