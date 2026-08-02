import { peekIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useRef, useState } from "react";

const GENERIC_CHECKLIST = [
  "Personalizing your experience",
  "Connecting automation engines",
  "Building your space",
] as const;

// Honest copy for the dump path — these steps are things that are
// actually happening on the server while the bar fills.
const BRAIN_DUMP_CHECKLIST = [
  "Reading your brain dump",
  "Briefing AutoPilot on your work",
  "Building your space",
] as const;

// Fixed duration, no pipeline gate: extraction and the greeting keep
// running server-side after this screen, and the copilot home polls the
// intro endpoint until the greeting is ready before revealing it.
const STEP_DURATION_MS = 10_000;

export function usePreparingStep({
  onComplete,
  isBrainDumpEnabled,
}: {
  onComplete: () => void;
  isBrainDumpEnabled: boolean;
}) {
  const [started, setStarted] = useState(false);
  const [completedItems, setCompletedItems] = useState(0);
  const [progress, setProgress] = useState(0);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  // Only a user who actually dumped gets the honest copy; a skip would
  // make "Reading your brain dump" a lie.
  const isDumpPath = isBrainDumpEnabled && peekIntroPath() === "A";
  const checklist = isDumpPath ? BRAIN_DUMP_CHECKLIST : GENERIC_CHECKLIST;
  const stepInterval = STEP_DURATION_MS / checklist.length;

  useEffect(() => {
    const timer = setTimeout(() => setStarted(true), 300);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!started) return;

    const startTime = Date.now();
    let finished = false;

    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      setProgress(Math.min(100, (elapsed / STEP_DURATION_MS) * 100));
      setCompletedItems(
        Math.min(checklist.length, Math.floor(elapsed / stepInterval) + 1),
      );

      if (elapsed >= STEP_DURATION_MS && !finished) {
        finished = true;
        clearInterval(progressInterval);
        onCompleteRef.current();
      }
    }, 50);

    return () => clearInterval(progressInterval);
  }, [started, checklist.length, stepInterval]);

  return { started, progress, completedItems, checklist };
}
