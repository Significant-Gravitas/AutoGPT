"use client";

import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

// Secondary row tucked under the composer card (Claude-style): the card
// overlaps its top edge, so the tray only peeks out below. Rounded on its
// own bottom edge so it also reads correctly outside a clipping wrapper.
export function ComposerTray({ children }: Props) {
  return (
    <div className="-mt-5 flex items-center gap-2 rounded-b-xlarge bg-[#EFEFF0] px-3 pb-1.5 pt-[1.625rem]">
      {children}
    </div>
  );
}
