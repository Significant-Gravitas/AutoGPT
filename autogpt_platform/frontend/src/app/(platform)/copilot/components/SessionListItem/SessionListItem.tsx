"use client";

import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { CheckCircle } from "@phosphor-icons/react";
import { AnimatePresence, motion } from "framer-motion";
import type { ReactNode } from "react";
import {
  formatSessionDate,
  getSessionStartTypeLabel,
  isNonManualSessionStartType,
} from "../../helpers";
import { PulseLoader } from "../PulseLoader/PulseLoader";

interface Props {
  actionSlot?: ReactNode;
  currentSessionId: string | null;
  isCompleted: boolean;
  onSelect: (sessionId: string) => void;
  session: SessionSummaryResponse;
  variant?: "sidebar" | "drawer";
}

export function SessionListItem({
  actionSlot,
  currentSessionId,
  isCompleted,
  onSelect,
  session,
  variant = "sidebar",
}: Props) {
  const isActive = session.id === currentSessionId;
  const showProcessing = session.is_processing && !isCompleted && !isActive;
  const showCompleted = isCompleted && !isActive;
  const startTypeLabel = isNonManualSessionStartType(session.start_type)
    ? getSessionStartTypeLabel(session.start_type)
    : null;

  if (variant === "drawer") {
    return (
      <button
        onClick={() => onSelect(session.id)}
        className={cn(
          "w-full rounded-lg px-3 py-2.5 text-left transition-colors",
          isActive ? "bg-zinc-100" : "hover:bg-zinc-50",
        )}
      >
        <div className="flex min-w-0 max-w-full flex-col overflow-hidden">
          <div className="flex min-w-0 max-w-full items-center gap-1.5">
            <Text
              variant="body"
              className={cn(
                "truncate font-normal",
                isActive ? "text-zinc-600" : "text-zinc-800",
              )}
            >
              {session.title || "Untitled chat"}
            </Text>
            {showProcessing ? (
              <PulseLoader size={8} className="shrink-0" />
            ) : null}
            {showCompleted ? (
              <CheckCircle
                className="h-4 w-4 shrink-0 text-green-500"
                weight="fill"
              />
            ) : null}
          </div>
          {startTypeLabel ? (
            <div className="mt-1">
              <Badge variant="info" size="small">
                {startTypeLabel}
              </Badge>
            </div>
          ) : null}
          <Text variant="small" className="text-neutral-400">
            {formatSessionDate(session.updated_at)}
          </Text>
        </div>
      </button>
    );
  }

  return (
    <div
      className={cn(
        "group relative w-full rounded-lg transition-colors",
        isActive ? "bg-zinc-100" : "hover:bg-zinc-50",
      )}
    >
      <button
        onClick={() => onSelect(session.id)}
        className="w-full px-3 py-2.5 pr-10 text-left"
      >
        <div className="flex min-w-0 max-w-full items-center gap-2">
          <div className="min-w-0 flex-1">
            <Text
              variant="body"
              className={cn(
                "truncate font-normal",
                isActive ? "text-zinc-600" : "text-zinc-800",
              )}
            >
              <AnimatePresence mode="wait" initial={false}>
                <motion.span
                  key={session.title || "untitled"}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.2 }}
                  className="block truncate"
                >
                  {session.title || "Untitled chat"}
                </motion.span>
              </AnimatePresence>
            </Text>
            {startTypeLabel ? (
              <div className="mt-1">
                <Badge variant="info" size="small">
                  {startTypeLabel}
                </Badge>
              </div>
            ) : null}
            <Text variant="small" className="text-neutral-400">
              {formatSessionDate(session.updated_at)}
            </Text>
          </div>
          {showProcessing ? (
            <PulseLoader size={16} className="shrink-0" />
          ) : null}
          {showCompleted ? (
            <CheckCircle
              className="h-4 w-4 shrink-0 text-green-500"
              weight="fill"
            />
          ) : null}
        </div>
      </button>
      {actionSlot ? (
        <div className="absolute right-2 top-1/2 -translate-y-1/2">
          {actionSlot}
        </div>
      ) : null}
    </div>
  );
}
