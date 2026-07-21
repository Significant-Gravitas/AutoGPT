import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  ChatCircleIcon,
  CheckCircleIcon,
  CircleNotchIcon,
  HourglassIcon,
} from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { MutableRefObject } from "react";
import { shouldShowSessionProcessingIndicator } from "../../sessionActivity";
import { ChatOriginIcon } from "../ChatOriginIcon/ChatOriginIcon";
import { getSessionTitle, highlightMatch, type SearchSession } from "./helpers";

const indicatorTransition = {
  type: "spring" as const,
  damping: 30,
  stiffness: 520,
  mass: 0.8,
};

interface Props {
  results: SearchSession[];
  query: string;
  highlightedIndex: number;
  highlightedResultRef: MutableRefObject<HTMLButtonElement | null>;
  currentSessionId: string | null;
  completedSessionIDs: Set<string>;
  sessionNeedsReload: Record<string, boolean>;
  onHighlight: (index: number) => void;
  onSelect: (id: string) => void;
}

export function ChatSearchResults({
  results,
  query,
  highlightedIndex,
  highlightedResultRef,
  currentSessionId,
  completedSessionIDs,
  sessionNeedsReload,
  onHighlight,
  onSelect,
}: Props) {
  const reduceMotion = useReducedMotion();
  return (
    <div role="listbox" aria-label="Chat sessions" className="flex flex-col">
      {results.map((session, index) => {
        const isHighlighted = index === highlightedIndex;
        const title = getSessionTitle(session);

        return (
          <Button
            key={session.id}
            ref={isHighlighted ? highlightedResultRef : undefined}
            id={`chat-search-result-${session.id}`}
            type="button"
            variant="ghost"
            role="option"
            aria-selected={isHighlighted}
            onMouseEnter={() => onHighlight(index)}
            onClick={() => onSelect(session.id)}
            className={cn(
              "relative h-auto w-full justify-start rounded-md px-3 py-2 text-left hover:bg-transparent",
            )}
          >
            {isHighlighted && (
              <>
                <motion.div
                  layoutId="chat-search-highlight"
                  aria-hidden="true"
                  className="absolute inset-0 z-0 rounded-md bg-zinc-100"
                  transition={
                    reduceMotion ? { duration: 0 } : indicatorTransition
                  }
                />
                <motion.div
                  layoutId="chat-search-highlight-bar"
                  aria-hidden="true"
                  className="absolute inset-y-0 left-0 z-[1] my-auto h-5 w-[3px] rounded-full bg-zinc-900"
                  transition={
                    reduceMotion ? { duration: 0 } : indicatorTransition
                  }
                />
              </>
            )}
            <div className="relative z-10 flex min-w-0 flex-1 items-center gap-2.5">
              <ChatCircleIcon
                aria-hidden="true"
                className={cn(
                  "h-4 w-4 shrink-0 transition-colors duration-150",
                  isHighlighted ? "text-zinc-900" : "text-zinc-500",
                )}
              />
              <ChatOriginIcon sourcePlatform={session.source_platform} />
              <div className="min-w-0 flex-1">
                <div
                  className={cn(
                    "truncate text-sm font-normal transition-colors duration-150",
                    isHighlighted ? "text-zinc-950" : "text-zinc-800",
                    session.id === currentSessionId && "text-zinc-600",
                  )}
                >
                  {highlightMatch(title, query).map((part, partIndex) => (
                    <span
                      key={`${part.text}-${partIndex}`}
                      className={part.isMatch ? "font-semibold" : undefined}
                    >
                      {part.text}
                    </span>
                  ))}
                </div>
              </div>
              {session.chat_status === "running" && (
                <span
                  aria-label="Session running"
                  className="inline-flex h-4 w-4 shrink-0 items-center justify-center"
                >
                  <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
                </span>
              )}
              {session.chat_status === "queued" && (
                <span
                  aria-label="Session queued"
                  className="inline-flex h-4 w-4 shrink-0 items-center justify-center text-purple-600"
                >
                  <HourglassIcon className="h-3.5 w-3.5" weight="bold" />
                </span>
              )}
              {session.chat_status !== "running" &&
                session.is_processing &&
                shouldShowSessionProcessingIndicator({
                  sessionId: session.id,
                  currentSessionId,
                  isProcessing: session.is_processing,
                  hasCompletedIndicator: completedSessionIDs.has(session.id),
                  needsReload: !!sessionNeedsReload[session.id],
                }) && (
                  <CircleNotchIcon
                    aria-label="Session processing"
                    className="h-4 w-4 shrink-0 animate-spin text-zinc-400"
                    weight="bold"
                  />
                )}
              {completedSessionIDs.has(session.id) &&
                session.id !== currentSessionId && (
                  <CheckCircleIcon
                    aria-label="Session completed"
                    className="h-4 w-4 shrink-0 text-green-500"
                    weight="fill"
                  />
                )}
              <AnimatePresence initial={false}>
                {isHighlighted && (
                  <motion.span
                    aria-hidden="true"
                    initial={
                      reduceMotion
                        ? { opacity: 0 }
                        : { opacity: 0, scale: 0.95, x: 80 }
                    }
                    animate={
                      reduceMotion
                        ? { opacity: 1 }
                        : { opacity: 1, scale: 1, x: 0 }
                    }
                    exit={
                      reduceMotion
                        ? { opacity: 0 }
                        : { opacity: 0, scale: 0.95, x: 80 }
                    }
                    transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
                    style={{ willChange: "transform, opacity" }}
                    className="inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-md border border-zinc-200 bg-white px-1 font-sans text-[11px] text-zinc-600 shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]"
                  >
                    ↵
                  </motion.span>
                )}
              </AnimatePresence>
            </div>
          </Button>
        );
      })}
    </div>
  );
}
