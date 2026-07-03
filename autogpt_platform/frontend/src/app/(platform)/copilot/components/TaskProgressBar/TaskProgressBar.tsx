"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import {
  CaretDownIcon,
  CheckIcon,
  CircleIcon,
  ListChecksIcon,
  PauseCircleIcon,
  SealCheckIcon,
} from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import {
  getCurrentLabel,
  getCurrentTask,
  isAllComplete,
  type TodoItem,
} from "./helpers";

// ease-out-quad — Emil's default for entering elements.
const EASE_OUT = [0.25, 0.46, 0.45, 0.94] as const;

// A newly in-progress loader waits for the completing task's tick + strikethrough
// to play first, so each step reads as a sequence.
const LOADER_DELAY = 0.3;

// Blur-bridge for the status icon swap (Emil's copy-button pattern).
const ICON_EXIT = 0.12;
const ICON_ENTER = 0.18;

// A task that was in_progress when the agent went idle (user Stop, end of turn,
// or error) is "stopped" — shown with a distinct amber icon instead of a spinner
// or a plain pending circle.
type DisplayStatus = TodoItem["status"] | "stopped";

function toDisplayStatus(
  status: TodoItem["status"],
  isStreaming: boolean,
): DisplayStatus {
  return !isStreaming && status === "in_progress" ? "stopped" : status;
}

// BlurText-style swap for the collapsed header's cycling current task only.
const BLUR_SWAP = {
  initial: { filter: "blur(10px)", opacity: 0, y: -10 },
  animate: {
    filter: ["blur(10px)", "blur(4px)", "blur(0px)"],
    opacity: [0, 0.6, 1],
    y: [-10, 3, 0],
    transition: { duration: 0.22, times: [0, 0.5, 1], ease: EASE_OUT },
  },
  exit: {
    filter: ["blur(0px)", "blur(4px)", "blur(10px)"],
    opacity: [1, 0.6, 0],
    y: [0, 4, 10],
    transition: { duration: 0.14, times: [0, 0.5, 1], ease: EASE_OUT },
  },
};

const BLUR_SWAP_REDUCED = {
  initial: false,
  animate: { opacity: 1 },
  exit: { opacity: 0, transition: { duration: 0 } },
};

interface Props {
  todos: TodoItem[];
  isStreaming?: boolean;
  defaultExpanded?: boolean;
}

export function TaskProgressBar({
  todos,
  isStreaming = false,
  defaultExpanded = false,
}: Props) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const reduceMotion = useReducedMotion();

  if (!todos || todos.length === 0) return null;

  const allDone = isAllComplete(todos);
  const current = getCurrentTask(todos, isStreaming);
  const currentIndex = current ? todos.indexOf(current) : -1;

  const reveal = reduceMotion
    ? { duration: 0 }
    : { duration: 0.22, ease: EASE_OUT };

  return (
    <div className="mx-auto w-[95%] overflow-hidden rounded-t-3xl border border-b-0 border-zinc-200 bg-neutral-100 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.9),inset_0_5px_6px_-4px_rgba(255,255,255,0.7)]">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="flex w-full items-center gap-2 px-3 py-3 text-left"
      >
        <div className="flex min-w-0 flex-1 items-center gap-2">
          {allDone ? (
            <>
              <SealCheckIcon
                size={22}
                weight="fill"
                className="flex-shrink-0 text-[#00a656]"
              />
              <Text
                variant="body-medium"
                className="min-w-0 flex-1 truncate text-sm text-zinc-800"
              >
                All tasks complete
              </Text>
            </>
          ) : !expanded && current ? (
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={`current-${currentIndex}`}
                {...(reduceMotion ? BLUR_SWAP_REDUCED : BLUR_SWAP)}
                className="flex min-w-0 flex-1 items-center gap-2"
              >
                <StatusIcon
                  status={toDisplayStatus(current.status, isStreaming)}
                />
                <Text
                  variant="body-medium"
                  className="min-w-0 flex-1 truncate text-sm text-zinc-800"
                >
                  {toDisplayStatus(current.status, isStreaming) === "stopped"
                    ? current.content
                    : getCurrentLabel(current)}
                </Text>
              </motion.div>
            </AnimatePresence>
          ) : (
            <>
              <ListChecksIcon
                size={16}
                weight="bold"
                className="flex-shrink-0 text-zinc-500"
              />
              <Text
                variant="body-medium"
                className="min-w-0 flex-1 text-sm text-zinc-800"
              >
                Task Progress
              </Text>
            </>
          )}
        </div>

        <span className="flex-shrink-0 text-sm tabular-nums text-zinc-900">
          {allDone ? todos.length : currentIndex + 1}/{todos.length}
        </span>
        <motion.span
          animate={{ rotate: expanded ? 180 : 0 }}
          transition={
            reduceMotion ? { duration: 0 } : { duration: 0.2, ease: EASE_OUT }
          }
          className="flex-shrink-0 text-zinc-400"
        >
          <CaretDownIcon size={14} weight="bold" />
        </motion.span>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            key="list"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={reveal}
            className="overflow-hidden"
          >
            <ul className="flex max-h-56 flex-col gap-1.5 overflow-y-auto px-3 pb-2.5">
              {todos.map((todo, i) => (
                <TaskRow
                  key={i}
                  todo={todo}
                  isStreaming={isStreaming}
                  reduceMotion={!!reduceMotion}
                />
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TaskRow({
  todo,
  isStreaming,
  reduceMotion,
}: {
  todo: TodoItem;
  isStreaming: boolean;
  reduceMotion: boolean;
}) {
  const status = toDisplayStatus(todo.status, isStreaming);
  const active = status === "in_progress";
  const completed = status === "completed";
  const label = active && todo.activeForm ? todo.activeForm : todo.content;
  const textClass = completed
    ? `text-zinc-400 ${reduceMotion ? "line-through" : ""}`
    : active
      ? "font-medium text-zinc-900"
      : "text-zinc-600";

  return (
    <li className="flex items-start gap-2 text-sm">
      <span className="mt-0.5 flex-shrink-0">
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={status}
            className="block"
            initial={reduceMotion ? false : { opacity: 0, filter: "blur(6px)" }}
            animate={{ opacity: 1, filter: "blur(0px)" }}
            exit={
              reduceMotion
                ? { opacity: 0 }
                : {
                    opacity: 0,
                    filter: "blur(6px)",
                    transition: {
                      duration: ICON_EXIT,
                      ease: EASE_OUT,
                      // A pending circle only leaves when its task becomes active
                      // — hold it until the previous step's tick/strike finishes.
                      delay: status === "pending" ? LOADER_DELAY : 0,
                    },
                  }
            }
            transition={{ duration: ICON_ENTER, ease: EASE_OUT }}
          >
            <StatusIcon status={status} />
          </motion.span>
        </AnimatePresence>
      </span>
      <span className={`min-w-0 flex-1 ${textClass}`}>
        <span className="relative inline">
          {label}
          {completed && !reduceMotion && (
            <motion.span
              aria-hidden
              className="pointer-events-none absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-zinc-400"
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              style={{ transformOrigin: "left" }}
              transition={{ duration: 0.25, ease: EASE_OUT, delay: ICON_EXIT }}
            />
          )}
        </span>
      </span>
    </li>
  );
}

function StatusIcon({ status }: { status: DisplayStatus }) {
  if (status === "completed") {
    return (
      <CheckIcon
        size={14}
        weight="bold"
        className="text-emerald-500"
        aria-label="completed"
      />
    );
  }
  if (status === "in_progress") {
    return (
      <LoadingSpinner
        size="small"
        className="h-3.5 w-3.5 text-purple-500 [animation-duration:0.5s]"
        aria-label="in progress"
      />
    );
  }
  if (status === "stopped") {
    return (
      <PauseCircleIcon
        size={15}
        weight="fill"
        className="text-amber-500"
        aria-label="stopped"
      />
    );
  }
  return (
    <CircleIcon
      size={14}
      weight="regular"
      className="text-zinc-400"
      aria-label="pending"
    />
  );
}
