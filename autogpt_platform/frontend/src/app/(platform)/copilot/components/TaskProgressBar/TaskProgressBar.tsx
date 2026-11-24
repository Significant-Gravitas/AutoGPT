"use client";

import { Text } from "@/components/atoms/Text/Text";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { memo, useState } from "react";
import { StatusIcon } from "./components/StatusIcon/StatusIcon";
import { TaskRow } from "./components/TaskRow/TaskRow";
import {
  getCurrentLabel,
  getCurrentTask,
  haveSameTodos,
  isAllComplete,
  toDisplayStatus,
  type TodoItem,
} from "./helpers";
import {
  ArrowDown01Icon,
  CheckListIcon,
  CheckmarkBadge01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { duration, ease } from "@/components/styles/motion";

// BlurText-style swap for the collapsed header's cycling current task only.
const BLUR_SWAP = {
  initial: { filter: "blur(10px)", opacity: 0, y: -10 },
  animate: {
    filter: ["blur(10px)", "blur(4px)", "blur(0px)"],
    opacity: [0, 0.6, 1],
    y: [-10, 3, 0],
    transition: { duration: duration.base, times: [0, 0.5, 1], ease: ease.out },
  },
  exit: {
    filter: ["blur(0px)", "blur(4px)", "blur(10px)"],
    opacity: [1, 0.6, 0],
    y: [0, 4, 10],
    transition: {
      duration: duration.instant,
      times: [0, 0.5, 1],
      ease: ease.out,
    },
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

// memo-boundary: the chat derives `todos` afresh on every streamed flush;
// the comparator below renders the bar again only when the tasks change.
export const TaskProgressBar = memo(function TaskProgressBar({
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
    : { duration: duration.base, ease: ease.out };

  return (
    <div className="mx-auto w-[95%] overflow-hidden rounded-t-3xl border border-b-0 border-border-base bg-surface-hover shadow-sheen">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="flex w-full items-center gap-2 px-3 py-3 text-left"
      >
        <div className="flex min-w-0 flex-1 items-center gap-2">
          {allDone ? (
            <>
              <Icon
                icon={CheckmarkBadge01Icon}
                size={22}
                tone="success"
                className="flex-shrink-0"
              />
              <Text
                variant="body"
                weight="medium"
                leading="5"
                tone="emphasis"
                className="min-w-0 flex-1 truncate"
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
                  variant="body"
                  weight="medium"
                  leading="5"
                  tone="emphasis"
                  className="min-w-0 flex-1 truncate"
                >
                  {toDisplayStatus(current.status, isStreaming) === "stopped"
                    ? current.content
                    : getCurrentLabel(current)}
                </Text>
              </motion.div>
            </AnimatePresence>
          ) : (
            <>
              <Icon
                icon={CheckListIcon}
                size={16}
                tone="muted"
                className="flex-shrink-0"
              />
              <Text
                variant="body"
                weight="medium"
                leading="5"
                tone="emphasis"
                className="min-w-0 flex-1"
              >
                Task Progress
              </Text>
            </>
          )}
        </div>

        <Text
          variant="body"
          leading="5"
          as="span"
          tone="primary"
          tabular
          className="flex-shrink-0"
        >
          {allDone ? todos.length : currentIndex + 1}/{todos.length}
        </Text>
        <motion.span
          animate={{ rotate: expanded ? 180 : 0 }}
          transition={
            reduceMotion
              ? { duration: 0 }
              : { duration: duration.base, ease: ease.out }
          }
          className="flex-shrink-0 text-foreground-subtle"
        >
          <Icon icon={ArrowDown01Icon} size={14} />
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
}, areSameTaskBarProps);

function areSameTaskBarProps(prev: Props, next: Props) {
  return (
    prev.isStreaming === next.isStreaming &&
    prev.defaultExpanded === next.defaultExpanded &&
    haveSameTodos(prev.todos, next.todos)
  );
}
