"use client";

import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

// Secondary row tucked under the composer card (Claude-style): the card
// overlaps its top edge, so the tray only peeks out below.
export function ComposerTray({ children }: Props) {
  return (
    <div className="-mt-5 flex items-center gap-2 bg-[#EFEFF0] px-3 pb-2 pt-7">
      {children}
    </div>
  );
}
