"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { TaskAdd01Icon } from "@hugeicons/core-free-icons";
import type { RefObject } from "react";

interface Props {
  onOpen: () => void;
  buttonRef: RefObject<HTMLButtonElement | null>;
}

/** The board's one coloured button: it opens the create dialog, and never
 *  shrinks or clips, however narrow the row. */
export function CreateTask({ onOpen, buttonRef }: Props) {
  return (
    <button
      ref={buttonRef}
      type="button"
      onClick={onOpen}
      className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-lg bg-gradient-to-b from-info to-info-foreground pr-3 pl-2.5 text-body-sm font-medium whitespace-nowrap text-surface shadow-(--inbox-shadow-lit) transition-[transform,filter] duration-150 ease-out outline-none hover:brightness-110 focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:ring-offset-2 active:scale-[0.97] motion-reduce:transition-none"
    >
      <Icon icon={TaskAdd01Icon} size={15} tone="inherit" />
      Create task
    </button>
  );
}
