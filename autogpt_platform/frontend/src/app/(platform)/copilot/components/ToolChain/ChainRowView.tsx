"use client";

import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { memo, useEffect, useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { HIDDEN_FROM_REPLAY } from "@/lib/replay-privacy";
import { cn } from "@/lib/utils";
import { useCopilotUIStore } from "../../store";
import { ACCORDION_PANEL, accordionState, PANEL_REVEAL } from "./accordion";
import { areChainRowPropsEqual } from "./chainRowEquality";
import type { ChainRow } from "./helpers";
import { ReasoningStream } from "./ReasoningStream";
import { isLiveSubSessionRow, SUB_SESSION_TOOLS } from "./subSessionRow";
import { ProviderIcon, RowIcon } from "./RowIcon";
import { useSubSessionEffectiveStatus } from "./SubSessionLive";
import { StepClock } from "./StepClock";
import { SwapText } from "./SwapText";
import { HeldTag } from "./HeldCallRowParts";
import { heldShowsOwnCall } from "./heldRow";
import { hidesOutputFromReplay } from "./replayPrivacy";
import { getCatalogLabel } from "./toolCatalog";
import { ToolResult } from "./ToolResult";
import { ToolStatusBadge } from "./ToolStatusBadge";
import { useMinimumRunning } from "./useMinimumRunning";

interface Props {
  row: ChainRow;
  isLast: boolean;
  readOnly?: boolean;
  /** The row is the newest one of a live chain and arrived already finished
   *  (see useMinimumRunning). Only its value at mount counts. */
  arrivedFinished?: boolean;
}

// memo-boundary: a chain row. Its chain rebuilds every row object on each
// streamed flush, text included; areChainRowPropsEqual finds the rows whose
// data really changed, so a settled row skips its render and its result card.
export const ChainRowView = memo(function ChainRowView({
  row,
  isLast,
  readOnly = false,
  arrivedFinished = false,
}: Props) {
  const [open, setOpen] = useState(row.requiresAction === true);
  const isReasoning = row.category === "reasoning";
  const shownState = useMinimumRunning({
    state: row.state,
    arrivedFinished,
    enabled:
      !!row.tool &&
      !isReasoning &&
      !row.held &&
      !row.requiresAction &&
      !row.supersededSubSession,
  });
  const holdingRun = shownState === "running" && row.state !== "running";
  const artifactPanelOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);

  // What the row needs from the user is only known once the output lands, so
  // a live row mounts closed and flips here — the user can still collapse it.
  useEffect(() => {
    if (row.requiresAction) setOpen(true);
  }, [row.requiresAction]);
  // Browser steps carry the page screenshots the artifact panel shows —
  // auto-expand them while the panel is open so the steps are visible
  // from the start.
  useEffect(() => {
    if (row.category === "browser" && artifactPanelOpen) setOpen(true);
  }, [row.category, artifactPanelOpen]);
  // A delegated run's card is the live view of the teammate working —
  // surface it without a click while the sub-session is still going.
  const liveSubSession = isLiveSubSessionRow(row) && !row.supersededSubSession;
  useEffect(() => {
    if (liveSubSession) setOpen(true);
  }, [liveSubSession]);
  // A delegate output frozen at "running" got its done-state label
  // ("Teammate handled:") while the teammate is in fact still working —
  // keep the running label + shimmer until the polled session goes idle.
  const output =
    row.output && typeof row.output === "object"
      ? (row.output as Record<string, unknown>)
      : null;
  const isSubTool = !!row.tool && SUB_SESSION_TOOLS.has(row.tool);
  const effectiveStatus = useSubSessionEffectiveStatus(
    isSubTool && typeof output?.sub_session_id === "string"
      ? output.sub_session_id
      : null,
    isSubTool && typeof output?.status === "string" ? output.status : null,
    isSubTool && typeof output?.background_task_id === "string",
  );
  // "unknown" means the poll died, not that the teammate finished — the row
  // only has a running label and a done label, and claiming a result landed
  // is the worse of the two guesses. It also keeps this row consistent with
  // the card below it, whose poll caps on its own mount clock.
  const stillWorking =
    isSubTool &&
    row.state === "done" &&
    ["running", "queued", "unknown"].includes(
      effectiveStatus?.toLowerCase() ?? "",
    );
  const liveReasoning =
    isReasoning && row.state === "running" && !!row.reasoningText;
  const hasContent = isReasoning
    ? !!row.reasoningText
    : !row.supersededSubSession &&
      ((row.output !== undefined && row.output !== "" && !row.emptyResult) ||
        liveSubSession ||
        !!row.held?.card ||
        heldShowsOwnCall(row));
  const showContent = liveReasoning || (open && hasContent);
  const rowText = (
    <SwapText
      text={
        holdingRun
          ? (row.runningText ?? row.text)
          : stillWorking && row.tool
            ? (getCatalogLabel(row.tool, row.input, "running")?.text ??
              row.text)
            : row.text
      }
      shimmer={shownState === "running" || stillWorking}
      className={cn(
        "max-w-full text-body transition-colors duration-300",
        row.state === "error"
          ? "text-error"
          : row.held &&
              row.held.state !== "approved" &&
              row.held.state !== "waiting"
            ? "text-foreground-subtle"
            : "text-foreground-secondary",
      )}
    />
  );
  const heldTag = row.held ? (
    <HeldTag state={row.held.state} read={row.held.read} />
  ) : null;
  const stepClock =
    row.state === "running" &&
    !isReasoning &&
    !liveSubSession &&
    row.held?.state !== "waiting" ? (
      <StepClock stepKey={row.key} />
    ) : null;

  return (
    <div className="flex items-stretch gap-2.5">
      <div className="flex w-7 flex-col items-center">
        <div
          className={cn(
            "relative flex size-7 shrink-0 items-center justify-center rounded-full transition-colors duration-300",
            row.state === "error" ? "bg-error-surface" : "bg-surface-hover",
          )}
        >
          <ToolStatusBadge state={shownState} label={row.text}>
            {row.state !== "error" && row.providerIconSrc ? (
              <ProviderIcon src={row.providerIconSrc} row={row} />
            ) : (
              <RowIcon row={row} />
            )}
          </ToolStatusBadge>
        </div>
        {!isLast && (
          <div className="w-px flex-1 origin-top animate-grow-line bg-surface-active motion-reduce:animate-none" />
        )}
      </div>
      <div className={cn("min-w-0 flex-1", isLast ? "pb-0" : "pb-3")}>
        {hasContent ? (
          <button
            type="button"
            onClick={() => setOpen(!open)}
            aria-expanded={showContent}
            className="group/row flex h-7 min-w-0 items-center gap-1.5"
          >
            {rowText}
            {heldTag}
            {stepClock}
            {!liveReasoning && (
              <Icon
                icon={ArrowDown01Icon}
                size={10}
                tone="disabled"
                motion="transform"
                className={cn(
                  "shrink-0 group-hover/row:text-foreground-tertiary",
                  open && "rotate-180",
                )}
              />
            )}
          </button>
        ) : (
          <div className="flex h-7 items-center gap-1.5">
            {rowText}
            {heldTag}
            {stepClock}
          </div>
        )}
        {row.detail && (
          <p className="animate-fade-in truncate text-caption text-error motion-reduce:animate-none">
            {row.detail}
          </p>
        )}
        <div className={ACCORDION_PANEL + " " + accordionState(showContent)}>
          <div
            aria-hidden={!showContent}
            inert={!showContent || undefined}
            className="min-h-0 overflow-hidden"
          >
            <div
              className={cn(
                "px-px pt-1.5 pb-px",
                showContent && PANEL_REVEAL,
                hidesOutputFromReplay(row) && HIDDEN_FROM_REPLAY,
              )}
            >
              {isReasoning ? (
                <ReasoningStream
                  text={row.reasoningText ?? ""}
                  live={liveReasoning}
                />
              ) : (
                <ToolResult row={row} readOnly={readOnly} />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}, areChainRowPropsEqual);
