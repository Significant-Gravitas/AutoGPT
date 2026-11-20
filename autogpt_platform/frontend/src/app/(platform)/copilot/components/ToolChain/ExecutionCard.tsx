"use client";

import {
  LinkSquare01Icon,
  PlayIcon,
  Robot01Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useLiveRunStatus } from "../../tools/RunAgent/components/ExecutionStartedCard/useLiveRunStatus";
import { CARD, HALF, StatusPill } from "./ResultCards";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  name: string;
  status?: string;
  href?: string;
  variant?: "run" | "agent";
  /** A run started from this chat: the card follows it until it ends. */
  run?: { graphID: string; executionID: string };
}

export function ExecutionCard({
  name,
  status,
  href,
  variant = "run",
  run,
}: Props) {
  const live = useLiveRunStatus(
    run
      ? { graph_id: run.graphID, execution_id: run.executionID, status }
      : null,
  );
  const shownStatus = live.status ?? status;

  return (
    <div
      data-testid="execution-card"
      className={`${CARD} ${HALF} flex flex-col gap-1.5 p-2.5`}
    >
      <div className="flex items-center gap-2.5">
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
          {variant === "agent" ? (
            <Icon icon={Robot01Icon} size={15} tone="secondary" />
          ) : (
            <Icon icon={PlayIcon} size={13} tone="secondary" />
          )}
        </div>
        <Text
          variant="body-sm-medium-inherit"
          tone="emphasis"
          className="min-w-0 flex-1 truncate"
        >
          {name}
        </Text>
        {shownStatus && <StatusPill status={shownStatus} />}
        {href && (
          <Link
            href={href}
            aria-label={
              variant === "agent" ? "Open workflow" : "Open execution"
            }
            className="shrink-0 rounded-full p-1 text-foreground-subtle transition-colors hover:bg-surface-hover hover:text-foreground-body"
          >
            <Icon icon={LinkSquare01Icon} size={14} />
          </Link>
        )}
      </div>
      {live.error && (
        <Text variant="small" tone="error" className="break-words">
          {live.error}
        </Text>
      )}
    </div>
  );
}
