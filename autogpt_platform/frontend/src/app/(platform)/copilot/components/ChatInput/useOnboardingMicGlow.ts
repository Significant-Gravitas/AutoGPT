import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import {
  takeMicGlow,
  takePendingLaterDump,
} from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useRef, useState } from "react";

// Long enough to survive AutoPilot's intro streaming in, short enough that
// it reads as a pointer rather than a permanent state.
const GLOW_DURATION_MS = 30_000;

export function useOnboardingMicGlow({
  isTranscribing,
}: {
  isTranscribing: boolean;
}) {
  const [isGlowing, setIsGlowing] = useState(false);

  useEffect(() => {
    if (!takeMicGlow()) return;
    setIsGlowing(true);
    const timer = setTimeout(() => setIsGlowing(false), GLOW_DURATION_MS);
    return () => clearTimeout(timer);
  }, []);

  // A user who skipped the onboarding dump is invited by AutoPilot's Path
  // B intro to record one here instead. Transcription finishing is the
  // closest thing to "they took the invitation" — the flag is consumed on
  // the first one, so later voice messages aren't counted as the dump.
  const wasTranscribingRef = useRef(false);
  useEffect(() => {
    const justFinished = wasTranscribingRef.current && !isTranscribing;
    wasTranscribingRef.current = isTranscribing;
    if (justFinished && takePendingLaterDump()) {
      trackBrainDump("later_dump_completed");
    }
  }, [isTranscribing]);

  return { isGlowing, dismissGlow: () => setIsGlowing(false) };
}
