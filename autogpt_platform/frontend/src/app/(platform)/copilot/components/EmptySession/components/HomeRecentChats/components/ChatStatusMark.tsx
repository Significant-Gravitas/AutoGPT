"use client";

import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { Eclipse } from "loading-dev";
import { useReducedMotion } from "motion/react";
import { chatStatus } from "../helpers";

const MARK_SIZE = 14;

interface Props {
  session: SessionSummaryResponse;
}

/** A chat's state: the turning mark a working expert has in the sidebar, or
 *  a tick once it is done. The word is read out ahead of the title. */
export function ChatStatusMark({ session }: Props) {
  const prefersReducedMotion = useReducedMotion();
  const status = chatStatus(session);
  return (
    <span
      title={status.label}
      data-state={status.isWorking ? "working" : "done"}
      className="flex shrink-0 items-center text-foreground-text"
    >
      {status.isWorking ? (
        <span aria-hidden className="flex">
          <Eclipse
            size={MARK_SIZE}
            playState={prefersReducedMotion ? "paused" : "running"}
          />
        </span>
      ) : (
        <Icon
          icon={CheckmarkCircle02Icon}
          size={MARK_SIZE}
          tone="success"
          aria-hidden
        />
      )}
      <span className="sr-only">{status.label}: </span>
    </span>
  );
}
