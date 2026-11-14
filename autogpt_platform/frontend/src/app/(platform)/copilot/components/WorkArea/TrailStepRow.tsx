"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { FOCUS } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { formatDuration, type TrailStep } from "./trailSteps";
import { NeedsYouTag } from "./blocks";
import { BEND_BACK, BEND_OUT, Dot, Light, Marker } from "./TrailMarks";

/** Sub-steps shown before the older ones fold into "+N earlier". */
const VISIBLE = 5;

const LINK =
  "absolute left-[15.25px] h-2 w-[1.5px] transition-colors duration-300";
const WALKED = "bg-info";
const UNWALKED = "bg-foreground-ink/10";
const META = "text-caption whitespace-nowrap tabular-nums";

interface Props {
  step: TrailStep;
  panelID: string;
  /** The full line, who wrote it and which subtask, for the tooltip. */
  hint: string;
  now: number;
  isOpen: boolean;
  hasPrevious: boolean;
  hasNext: boolean;
  /** The line into and out of the step is one the work has walked. */
  isWalkedIn: boolean;
  isWalkedOut: boolean;
  /** The step above is open: its branch lands on this step's mark. */
  isAfterBranch: boolean;
  onToggle: () => void;
}

export function TrailStepRow({
  step,
  panelID,
  hint,
  now,
  isOpen,
  hasPrevious,
  hasNext,
  isWalkedIn,
  isWalkedOut,
  isAfterBranch,
  onToggle,
}: Props) {
  const [showsAll, setShowsAll] = useState(false);
  const isLong = step.lines.length > VISIBLE + 1;
  const folded = isLong && !showsAll ? step.lines.length - VISIBLE : 0;
  const shown = step.lines.slice(folded);
  const current =
    step.state === "running" ? shown.length - 1 + (isLong ? 1 : 0) : -1;
  const isExpandable = step.lines.length > 0;
  const wire = isWalkedOut ? WALKED : UNWALKED;
  const wireStroke = isWalkedOut ? "stroke-info" : "stroke-foreground-ink/10";

  const row = (
    <>
      {hasPrevious && !isAfterBranch ? (
        <span className={cn(LINK, "top-0", isWalkedIn ? WALKED : UNWALKED)} />
      ) : null}
      {hasNext && !isOpen ? (
        <span
          className={cn(LINK, "bottom-0", isWalkedOut ? WALKED : UNWALKED)}
        />
      ) : null}
      <Marker state={step.state} />
      <span
        className={cn(
          "min-w-0 truncate text-body-sm",
          step.state === "running" && "text-trail-shine",
          step.state === "done" && "text-foreground-text",
          (step.state === "waiting" || step.state === "failed") &&
            "text-foreground-ink",
          (step.state === "next" || step.state === "stopped") &&
            "text-foreground-quiet",
        )}
      >
        {step.title}
      </span>
      {isExpandable ? (
        <span
          className={cn(
            "ml-1 flex shrink-0 opacity-0 transition-[opacity,transform,rotate] duration-200 ease-out group-hover/step:opacity-100 group-focus-visible/step:opacity-100 pointer-coarse:opacity-100",
            !isOpen && "-rotate-90",
          )}
        >
          <Icon icon={ArrowDown01Icon} size={12} tone="faint" />
        </span>
      ) : null}
      <span className="ml-auto flex shrink-0 pl-3">
        <StepMeta step={step} now={now} />
      </span>
    </>
  );
  const rowClass =
    "group/step relative flex h-8 w-full items-center rounded-md px-2 text-left";

  return (
    <li
      className="relative"
      aria-current={step.state === "running" ? "step" : undefined}
    >
      {isExpandable ? (
        <button
          type="button"
          title={hint}
          aria-expanded={isOpen}
          aria-controls={panelID}
          onClick={onToggle}
          className={cn(
            rowClass,
            "transition-colors duration-100 ease-out hover:bg-surface-tray",
            FOCUS,
          )}
        >
          {row}
        </button>
      ) : (
        <div title={hint} className={rowClass}>
          {row}
        </div>
      )}
      {isExpandable ? (
        <>
          <span
            aria-hidden
            className={cn(
              "pointer-events-none absolute top-6 -bottom-2 left-2 w-7 opacity-0 transition-opacity duration-200",
              isOpen && "opacity-100",
            )}
          >
            <svg
              viewBox="0 0 28 14"
              className={cn(
                "absolute top-0 left-0 h-3.5 w-7 fill-none",
                wireStroke,
              )}
              strokeWidth={1.5}
            >
              <path d={BEND_OUT} />
            </svg>
            <span
              className={cn(
                "absolute top-3.5 left-[19.25px] w-[1.5px]",
                hasNext ? "bottom-3.5" : "bottom-[22px]",
                wire,
              )}
            />
            {hasNext ? (
              <svg
                viewBox="0 0 28 14"
                className={cn(
                  "absolute bottom-0 left-0 h-3.5 w-7 fill-none",
                  wireStroke,
                )}
                strokeWidth={1.5}
              >
                <path d={BEND_BACK} />
              </svg>
            ) : null}
          </span>
          {isOpen && current >= 0 ? <Light index={current} /> : null}
          <div
            id={panelID}
            inert={!isOpen || undefined}
            className={cn(
              "grid grid-rows-[0fr] transition-[grid-template-rows] duration-300 ease-out-expo",
              isOpen && "grid-rows-[1fr]",
            )}
          >
            <div className="min-h-0 overflow-hidden">
              <ul className="py-0.5">
                {isLong ? (
                  <li className="flex h-6 pl-4">
                    <button
                      type="button"
                      aria-expanded={showsAll}
                      onClick={() => setShowsAll(!showsAll)}
                      className={cn(
                        "ml-[3.5px] flex h-full flex-1 items-center rounded-md pr-2 pl-1.5 text-left text-caption text-foreground-quiet transition-colors duration-100 ease-out hover:bg-surface-tray hover:text-foreground-text",
                        FOCUS,
                      )}
                    >
                      <Dot kind="fold" />
                      <span className="ml-[7.5px]">
                        {showsAll ? "Show fewer" : `+${folded} earlier`}
                      </span>
                    </button>
                  </li>
                ) : null}
                {shown.map((line, index) => {
                  const isCurrent =
                    step.state === "running" && index === shown.length - 1;
                  return (
                    <li
                      key={line.id}
                      title={line.text}
                      className={cn(
                        "flex h-6 -translate-y-1 items-center pr-2 pl-6 opacity-0 transition-[opacity,transform,translate] duration-200 ease-out",
                        isOpen && "translate-y-0 opacity-100",
                      )}
                    >
                      <span className="ml-[1.5px] flex">
                        <Dot kind={isCurrent ? "current" : "done"} />
                      </span>
                      <span
                        className={cn(
                          "ml-[7.5px] min-w-0 truncate text-caption",
                          isCurrent
                            ? "text-foreground-text"
                            : "text-foreground-quiet",
                        )}
                      >
                        {line.text}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </>
      ) : null}
    </li>
  );
}

interface MetaProps {
  step: TrailStep;
  now: number;
}

/** What sits at a step's end: how long it took, how long it has been
 *  going, or the word for where it stopped. */
function StepMeta({ step, now }: MetaProps) {
  if (step.state === "running") {
    const elapsed =
      step.startedAt === null ? "" : formatDuration(now - step.startedAt);
    return <span className={cn(META, "text-info")}>{elapsed}</span>;
  }
  if (step.state === "waiting") return <NeedsYouTag />;
  if (step.state === "failed") {
    return <span className={cn(META, "font-medium text-error")}>Failed</span>;
  }
  if (step.state === "stopped") {
    return <span className={cn(META, "text-foreground-faint")}>Stopped</span>;
  }
  if (step.state === "next") {
    return <span className={cn(META, "text-foreground-faint")}>Up next</span>;
  }
  return (
    <span className={cn(META, "text-foreground-faint")}>
      {formatDuration(step.tookMs)}
    </span>
  );
}
