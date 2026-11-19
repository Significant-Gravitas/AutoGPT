"use client";

import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import {
  InfoGroup,
  InfoList,
} from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";

export const NEEDS = "brief-needs";
export const READY = "brief-ready";
export const FAILED = "brief-failed";
export const WORKING = "brief-working";
export const RAN = "brief-ran";
export const NEXT = "brief-next";
export const STARTING = "brief-starting";

interface BlockProps {
  id: string;
  label: string;
  count: number;
  children: ReactNode;
}

/** One block of the brief; nothing when it is empty. */
export function Block({ id, label, count, children }: BlockProps) {
  if (count === 0) return null;
  return (
    <div id={id} className="scroll-mt-3">
      <InfoGroup label={label} count={count}>
        <InfoList>{children}</InfoList>
      </InfoGroup>
    </div>
  );
}

interface JumpProps {
  to: string;
  count: number;
  isWaiting?: boolean;
  children: string;
}

/** A count that scrolls the brief to its block; nothing when it is zero. */
export function Jump({ to, count, isWaiting, children }: JumpProps) {
  if (count === 0) return null;

  function jump() {
    const isStill = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    document.getElementById(to)?.scrollIntoView({
      behavior: isStill ? "auto" : "smooth",
      block: "start",
    });
  }

  return (
    <button
      type="button"
      onClick={jump}
      className="flex h-7 items-center gap-1.5 rounded-full bg-surface px-2.5 text-body-sm shadow-(--inbox-shadow-card) transition-shadow duration-150 ease-out outline-none hover:shadow-(--inbox-shadow-card-hover) focus-visible:ring-2 focus-visible:ring-focus-inbox"
    >
      <span
        className={cn(
          "font-medium tabular-nums",
          isWaiting ? "text-status-waiting" : "text-foreground-ink",
        )}
      >
        {count}
      </span>
      <span className="text-foreground-text">{children}</span>
    </button>
  );
}

interface FigureProps {
  term: string;
  value: string;
}

export function Figure({ term, value }: FigureProps) {
  return (
    <div className="flex min-w-0 flex-col">
      <dt className="truncate text-caption text-foreground-quiet">{term}</dt>
      <dd className="text-body-lg font-medium text-foreground-ink tabular-nums">
        {value}
      </dd>
    </div>
  );
}
