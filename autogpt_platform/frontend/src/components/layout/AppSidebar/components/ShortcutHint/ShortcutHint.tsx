"use client";

import { useEffect, useState } from "react";
import { isMacPlatform } from "@/lib/platform";
import { cn } from "@/lib/utils";

interface Props {
  letter: string;
  className?: string;
}

export function ShortcutHint({ letter, className }: Props) {
  const [isMac, setIsMac] = useState(true);
  useEffect(() => setIsMac(isMacPlatform()), []);

  const keys = [isMac ? "⌘" : "Ctrl", "⇧", letter];

  return (
    <span
      aria-hidden
      className={cn(
        "ml-auto flex items-center gap-0.5 opacity-0 group-hover/menu-item:opacity-100 group-data-[collapsible=icon]:hidden",
        className,
      )}
    >
      {keys.map((key) => (
        <kbd
          key={key}
          // TODO(tokens): SECRT-3152, rare size text-[0.7rem]
          className="rounded border border-border-strong bg-surface-muted px-1.5 py-1 font-sans text-[0.7rem] leading-none text-foreground-body"
        >
          {key}
        </kbd>
      ))}
    </span>
  );
}
