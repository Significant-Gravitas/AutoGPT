import { Icon } from "@/components/atoms/Icon/Icon";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { motion } from "framer-motion";

export type ChecklistState = "done" | "active" | "pending";

interface Props {
  state: ChecklistState;
}

export function ChecklistMark({ state }: Props) {
  if (state === "done") {
    return (
      <motion.span
        initial={{ scale: 0.4, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 500, damping: 22 }}
        className="flex size-6 shrink-0 items-center justify-center rounded-full bg-brand text-action-foreground"
      >
        <Icon icon={Tick02Icon} size={14} aria-hidden />
        <span className="sr-only">Completed</span>
      </motion.span>
    );
  }

  if (state === "active") {
    return (
      <span className="size-6 shrink-0 animate-spin rounded-full border-2 border-brand/20 border-t-brand">
        <span className="sr-only">In progress</span>
      </span>
    );
  }

  return (
    <span className="size-6 shrink-0 rounded-full border-2 border-border-base">
      <span className="sr-only">Pending</span>
    </span>
  );
}
