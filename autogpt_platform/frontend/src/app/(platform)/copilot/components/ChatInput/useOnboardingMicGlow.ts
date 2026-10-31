import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import {
  takeMicGlow,
  takePendingLaterDump,
} from "@/services/onboarding/brain-dump-handoff";
import { useComposerVoiceRequest } from "@/services/onboarding/composer-voice-request";
import { useEffect, useRef, useState } from "react";

// Long enough to survive Otto's intro streaming in, short enough that
// it reads as a pointer rather than a permanent state.
const GLOW_DURATION_MS = 30_000;

export function useOnboardingMicGlow({
  isTranscribing,
  canRecord,
  startRecording,
}: {
  isTranscribing: boolean;
  canRecord: boolean;
  startRecording: () => void | Promise<void>;
}) {
  const [isGlowing, setIsGlowing] = useState(false);
  const voiceRequest = useComposerVoiceRequest((state) => state.count);
  const seenVoiceRequestRef = useRef(voiceRequest);

  useEffect(() => {
    if (voiceRequest === seenVoiceRequestRef.current) return;
    seenVoiceRequestRef.current = voiceRequest;
    if (!canRecord) return;
    setIsGlowing(false);
    void startRecording();
  }, [voiceRequest, canRecord, startRecording]);

  useEffect(() => {
    if (!takeMicGlow()) return;
    setIsGlowing(true);
    const timer = setTimeout(() => setIsGlowing(false), GLOW_DURATION_MS);
    return () => clearTimeout(timer);
  }, []);

  // A user who skipped the onboarding dump is invited by Otto's Path
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
