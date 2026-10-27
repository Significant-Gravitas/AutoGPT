"use client";

import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { type ReactNode, useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

interface Props {
  children: ReactNode;
}

/** What a step ran on (a command, a path, an id), closed until asked for:
 *  it is for whoever wants it, not for the business owner reading along. */
export function TechnicalDetails({ children }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex items-center gap-1 text-xs text-foreground-subtle transition-colors hover:text-foreground-secondary"
      >
        {open ? "Hide technical details" : "Show technical details"}
        <Icon
          icon={ArrowDown01Icon}
          size={10}
          className={cn(
            "transition-transform duration-300 ease-out-quint",
            open && "rotate-180",
          )}
        />
      </button>
      {open && <div className="flex w-full flex-col gap-1.5">{children}</div>}
    </>
  );
}

export function TechnicalValue({ children }: Props) {
  return (
    <p className="break-all font-mono text-xs text-foreground-secondary">
      {children}
    </p>
  );
}
