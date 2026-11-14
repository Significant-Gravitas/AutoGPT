"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Sun03Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { useIsMobile } from "../../../../useIsMobile";
import { useWorkAreaStore } from "../../../../workArea";
import { briefName } from "./helpers";

const SEEN_KEY = "home-brief-seen";

function wasSeen(day: string) {
  try {
    return window.localStorage.getItem(SEEN_KEY) === day;
  } catch {
    return false;
  }
}

/** Opens the day's brief in the sidebar, with a dot until it has been
 *  opened today. A phone has no sidebar, so there it is left out. */
export function BriefButton() {
  const [now] = useState(() => Date.now());
  const day = new Date(now).toDateString();
  const label = briefName(now);
  const [isSeen, setIsSeen] = useState(() => wasSeen(day));
  const openBrief = useWorkAreaStore((s) => s.openBrief);
  const isMobile = useIsMobile();
  if (isMobile) return null;

  function open() {
    openBrief(label);
    setIsSeen(true);
    try {
      window.localStorage.setItem(SEEN_KEY, day);
    } catch {
      // Private mode: the dot simply comes back next time.
    }
  }

  return (
    <button
      type="button"
      onClick={open}
      className="relative inline-flex h-8 shrink-0 items-center gap-1.5 rounded-lg bg-surface pr-3 pl-2.5 text-body-sm font-medium whitespace-nowrap text-foreground-ink shadow-(--inbox-shadow-card) transition-[transform,box-shadow] duration-150 ease-out outline-none hover:shadow-(--inbox-shadow-card-hover) focus-visible:ring-2 focus-visible:ring-focus-inbox active:scale-[0.97] motion-reduce:transition-none"
    >
      <Icon icon={Sun03Icon} size={15} tone="inherit" />
      {label}
      {isSeen ? null : (
        <span className="absolute -top-0.5 -right-0.5 size-2 rounded-full bg-info ring-2 ring-canvas">
          <span className="sr-only">New</span>
        </span>
      )}
    </button>
  );
}
