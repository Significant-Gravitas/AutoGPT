"use client";

import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { AUTOPILOT_AURA_COLORS } from "@/components/molecules/AutopilotAvatar/helpers";
import { useAudioBars } from "../../../../BrainDumpStep/components/useAudioBars";
import { VoiceAura } from "../../../../BrainDumpStep/components/VoiceAura";

const AVATAR_SIZE = 120;

interface Props {
  // The person's voice and Otto's mixed; null while no call is live.
  levels: MediaStream | null;
  isActive: boolean;
  // Otto is working on what was said: a slow orbit and breath around him,
  // apart from the voice streaks of a live call.
  isWorking?: boolean;
}

// Otto, with the brain dump's voice streaks orbiting while the call is live,
// moved by whoever is speaking.
export function CallAura({ levels, isActive, isWorking = false }: Props) {
  const bars = useAudioBars(isActive ? levels : null);

  return (
    <div
      data-testid="call-aura"
      data-active={isActive}
      data-working={isWorking}
      className="relative"
    >
      {isWorking && (
        <>
          <span
            aria-hidden
            className="pointer-events-none absolute -inset-3 rounded-full border-2 border-transparent border-t-brand border-r-brand/30 motion-safe:animate-orbit"
          />
          <span
            aria-hidden
            className="pointer-events-none absolute -inset-1 rounded-full bg-accent-surface opacity-60 motion-safe:animate-pulse"
          />
        </>
      )}
      <VoiceAura
        colors={AUTOPILOT_AURA_COLORS}
        size={AVATAR_SIZE}
        levels={bars}
        isActive={isActive}
      >
        <AutopilotAvatar size={AVATAR_SIZE} transparent />
      </VoiceAura>
    </div>
  );
}
