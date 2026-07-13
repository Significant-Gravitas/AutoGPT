"use client";

import { useEffect, useState } from "react";
import { isMacPlatform } from "@/lib/platform";

interface Props {
  mac: string[];
  other: string[];
}

export function SidebarShortcutHint({ mac, other }: Props) {
  const [isMac, setIsMac] = useState(false);
  useEffect(() => setIsMac(isMacPlatform()), []);

  const keys = isMac ? mac : other;

  return (
    <kbd className="ml-auto hidden h-5 items-center gap-1 rounded-md bg-zinc-100 px-1.5 font-sans text-[13px] font-medium leading-none text-zinc-600 group-hover/menu-item:inline-flex group-data-[collapsible=icon]:!hidden">
      {keys.map((key, index) => (
        <span key={index} className="inline-flex items-center leading-none">
          {key}
        </span>
      ))}
    </kbd>
  );
}
