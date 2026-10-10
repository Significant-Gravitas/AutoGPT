import type { OnboardingWizardProgress } from "@/app/api/__generated__/models/onboardingWizardProgress";
import {
  readLocalProgress,
  sameProgress,
  writeLocalProgress,
} from "./progress";

export function createProgressSaver({
  userID,
  save,
  onError,
  onConflict,
  initialRevision,
}: {
  userID: string;
  initialRevision: number;
  onConflict: () => void;
  save: (
    progress: OnboardingWizardProgress,
    revision: number,
    signal: AbortSignal,
  ) => Promise<number>;
  onError: (message: string | null) => void;
}) {
  let revision = initialRevision;
  let conflicted = false;
  let pending: OnboardingWizardProgress | null = null;
  let inFlight: Promise<void> | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let disposed = false;
  let failures = 0;
  let controller: AbortController | null = null;

  function schedule(delay: number) {
    clearTimeout(timer);
    timer = setTimeout(() => void flush().catch(() => undefined), delay);
  }

  function enqueue(progress: OnboardingWizardProgress) {
    if (disposed || conflicted) return;
    pending = progress;
    if (!writeLocalProgress(userID, progress, true, revision)) {
      onError(
        "Your progress hasn't been saved yet. Please keep this page open while we retry.",
      );
    }
    schedule(200);
  }

  async function saveWithTimeout(progress: OnboardingWizardProgress) {
    const request = new AbortController();
    controller = request;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        save(progress, revision, request.signal),
        new Promise<never>((_resolve, reject) => {
          timeout = setTimeout(() => {
            request.abort();
            reject(new Error("Saving onboarding timed out. Please retry."));
          }, 10_000);
        }),
      ]);
    } finally {
      clearTimeout(timeout);
      if (controller === request) controller = null;
    }
  }

  async function drain() {
    while (pending && !disposed) {
      const progress = pending;
      try {
        const previousRevision = revision;
        revision = await saveWithTimeout(progress);
        if (disposed) return;
        const latest = pending ?? progress;
        const local = readLocalProgress(userID);
        const ownsLocal =
          local?.revision === previousRevision &&
          sameProgress(latest, local.progress);
        if (pending === progress) pending = null;
        if (ownsLocal)
          writeLocalProgress(userID, latest, pending !== null, revision);
        failures = 0;
        onError(null);
      } catch (error) {
        if (disposed) return;
        if (isProgressConflict(error)) {
          conflicted = true;
          onConflict();
          throw error;
        }
        failures += 1;
        onError(
          "We couldn't save your progress to your account. We'll keep retrying; you can continue here.",
        );
        schedule(Math.min(30_000, 1000 * 2 ** (failures - 1)));
        throw error;
      }
    }
  }

  async function flush() {
    clearTimeout(timer);
    if (conflicted)
      throw new Error(
        "Another session updated onboarding. Reload the latest progress.",
      );
    if (disposed)
      throw new Error("Your account changed. Please reload onboarding.");
    if (!inFlight)
      inFlight = drain().finally(() => {
        inFlight = null;
      });
    await inFlight;
    if (pending) await flush();
  }

  function dispose() {
    disposed = true;
    clearTimeout(timer);
    controller?.abort();
  }

  return { enqueue, flush, dispose };
}

export function isProgressConflict(error: unknown) {
  return (
    typeof error === "object" &&
    error !== null &&
    "status" in error &&
    error.status === 409
  );
}
