"use client";

import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { tourTarget } from "@/lib/tour-targets";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { SuggestedTaskCard } from "./SuggestedTaskCard";

interface Props {
  tasks: SuggestedTask[];
  sendingId: string | null;
  disabled?: boolean;
  onSend: (task: SuggestedTask) => void;
  onHide: (task: SuggestedTask) => void;
  className?: string;
}

// The greeting's suggestions as a grid of cards: the same rows as
// SuggestedTaskList, laid out to be scanned rather than read.
export function SuggestedTaskCards({
  tasks,
  sendingId,
  disabled = false,
  onSend,
  onHide,
  className,
}: Props) {
  const prefersReducedMotion = useReducedMotion();

  return (
    <ol
      {...tourTarget("first-tasks")}
      className={cn("grid grid-cols-1 gap-3 sm:grid-cols-2", className)}
    >
      <AnimatePresence initial={false}>
        {tasks.map((task) => (
          <motion.li
            key={task.id}
            layout={!prefersReducedMotion}
            exit={
              prefersReducedMotion ? undefined : { scale: 0.96, opacity: 0 }
            }
            transition={{ duration: 0.15 }}
            className="group/card relative flex"
            data-testid="suggested-task"
          >
            <SuggestedTaskCard
              task={task}
              isSending={sendingId === task.id}
              isBusy={disabled || sendingId !== null}
              canHide={sendingId === null}
              onSend={onSend}
              onHide={onHide}
            />
          </motion.li>
        ))}
      </AnimatePresence>
    </ol>
  );
}
