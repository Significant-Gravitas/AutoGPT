"use client";

import { CaretRightIcon, LightbulbIcon } from "@phosphor-icons/react";
import { useState } from "react";

interface Props {
  children: React.ReactNode;
}

export function ReasoningCollapse({ children }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="my-1">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex items-center gap-1 text-xs text-zinc-500 transition-colors hover:text-zinc-700"
      >
        <CaretRightIcon
          size={10}
          weight="bold"
          className={
            "transition-transform duration-150 " + (open ? "rotate-90" : "")
          }
        />
        <LightbulbIcon size={12} weight="bold" />
        <span>{open ? "Hide reasoning" : "Show reasoning"}</span>
      </button>
      {open && (
        <div className="mt-1 space-y-1 border-l-2 border-zinc-200 pl-3">
          {children}
        </div>
      )}
    </div>
  );
}
