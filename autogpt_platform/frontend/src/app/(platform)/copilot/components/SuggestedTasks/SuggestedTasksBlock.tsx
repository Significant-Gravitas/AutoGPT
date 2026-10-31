"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, ArrowUp01Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useId, useState } from "react";
import { YourTeamRow } from "./components/YourTeamRow";
import { countOpenTasks } from "./helpers";
import { SuggestedTaskList } from "./SuggestedTaskList";
import type { useCollapsedSuggestions } from "./useCollapsedSuggestions";
import type { useSuggestedTasks } from "./useSuggestedTasks";

interface Props {
  suggestions: ReturnType<typeof useSuggestedTasks>;
  collapse: ReturnType<typeof useCollapsedSuggestions>;
  disabled?: boolean;
  /** The first visit: who is on the team, above the list. */
  showTeam?: boolean;
  className?: string;
}

// The suggested tasks under one header, in the greeting and on Home after
// it: Hide folds the list away, Show brings it back.
export function SuggestedTasksBlock({
  suggestions,
  collapse,
  disabled,
  showTeam = false,
  className,
}: Props) {
  const prefersReducedMotion = useReducedMotion();
  const listID = useId();
  const toggleID = useId();
  const [moreChoice, setMoreChoice] = useState<boolean | null>(null);
  const count = suggestions.tasks.length;
  // Owners are shown, and tasks sent, only once the roster has answered.
  if (count === 0 || !suggestions.isRosterLoaded) return null;
  const isOpen = !collapse.isCollapsed;

  return (
    <section
      className={cn("w-full text-left", className)}
      aria-label="Suggested first tasks"
    >
      <div className="mb-2 flex items-center justify-between px-1">
        <Text variant="eyebrow">Suggested first tasks</Text>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          withTooltip={false}
          id={toggleID}
          aria-expanded={isOpen}
          aria-controls={listID}
          onClick={() => collapse.toggle(countOpenTasks(suggestions.allTasks))}
          className="gap-1 text-foreground-secondary hover:text-foreground-body"
        >
          {isOpen
            ? "Hide"
            : `${count} ${count === 1 ? "task" : "tasks"} · Show`}
          <Icon icon={isOpen ? ArrowUp01Icon : ArrowDown01Icon} size={14} />
        </Button>
      </div>
      <div id={listID}>
        <AnimatePresence initial={false}>
          {isOpen ? (
            <motion.div
              key="list"
              initial={prefersReducedMotion ? false : { height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={
                prefersReducedMotion ? undefined : { height: 0, opacity: 0 }
              }
              transition={{ duration: prefersReducedMotion ? 0 : 0.2 }}
              className="overflow-hidden"
            >
              {showTeam ? <YourTeamRow experts={suggestions.experts} /> : null}
              <SuggestedTaskList
                tasks={suggestions.allTasks}
                experts={suggestions.experts}
                expertsById={suggestions.expertsById}
                sendingId={suggestions.sendingId}
                disabled={disabled}
                moreChoice={moreChoice}
                onMoreChoice={setMoreChoice}
                fallbackFocusID={toggleID}
                onSend={suggestions.sendTask}
                onRemove={suggestions.removeTask}
              />
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </section>
  );
}
