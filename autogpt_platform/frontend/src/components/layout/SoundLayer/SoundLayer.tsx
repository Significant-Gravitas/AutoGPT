"use client";

import { SoundEffects } from "@/components/ui/sound";
import { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

// Kobra's sound layer: one document listener that answers every control
// press with a cue, chosen from the control's `data-slot`. Mounted once in
// src/app/providers.tsx; the mute switch is SidebarUserActions' SoundToggle.
export function SoundLayer({ children }: Props) {
  return <SoundEffects>{children}</SoundEffects>;
}
