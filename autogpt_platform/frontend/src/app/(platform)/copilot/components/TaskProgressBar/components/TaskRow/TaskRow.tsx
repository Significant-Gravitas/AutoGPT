import { AnimatePresence, motion } from "framer-motion";
import {
  EASE_OUT,
  ICON_ENTER,
  ICON_EXIT,
  LOADER_DELAY,
} from "../../animations";
import { toDisplayStatus, type TodoItem } from "../../helpers";
import { StatusIcon } from "../StatusIcon/StatusIcon";

interface Props {
  todo: TodoItem;
  isStreaming: boolean;
  reduceMotion: boolean;
}

export function TaskRow({ todo, isStreaming, reduceMotion }: Props) {
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
