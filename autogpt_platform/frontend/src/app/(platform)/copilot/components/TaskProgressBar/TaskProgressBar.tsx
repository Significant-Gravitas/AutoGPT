"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  CaretDownIcon,
  ListChecksIcon,
  SealCheckIcon,
} from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import { EASE_OUT } from "./animations";
import { StatusIcon } from "./components/StatusIcon/StatusIcon";
import { TaskRow } from "./components/TaskRow/TaskRow";
import {
  getCurrentLabel,
  getCurrentTask,
  isAllComplete,
  toDisplayStatus,
  type TodoItem,
} from "./helpers";

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
