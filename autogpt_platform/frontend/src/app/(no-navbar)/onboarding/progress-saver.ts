import type { OnboardingWizardProgress } from "@/app/api/__generated__/models/onboardingWizardProgress";
import {
  readLocalProgress,
  sameProgress,
  writeLocalProgress,
} from "./progress";

interface ServerProgress {
  progress: OnboardingWizardProgress | null;
  revision: number;
}

export function createProgressSaver({
  userID,
  save,
  load,
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
  // Resolves null when the server answers for a different account.
  load: (signal: AbortSignal) => Promise<ServerProgress | null>;
  onError: (message: string | null) => void;
}) {
  let revision = initialRevision;
  let conflicted = false;
  let pending: OnboardingWizardProgress | null = null;
  const sent = new Set<OnboardingWizardProgress>();
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

  async function withTimeout<T>(run: (signal: AbortSignal) => Promise<T>) {
    const request = new AbortController();
    controller = request;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        run(request.signal),
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
      sent.add(progress);
      try {
        const saved = await saveOrAdopt(progress);
        if (disposed) return;
        confirm(saved);
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

  async function saveOrAdopt(progress: OnboardingWizardProgress) {
    try {
      const saved = await withTimeout((signal) =>
        save(progress, revision, signal),
      );
      return { progress, revision: saved };
    } catch (error) {
      if (!isProgressConflict(error) || disposed) throw error;
      // A save this tab timed out on may have landed and moved the revision on.
      const server = await withTimeout(load);
      if (!server || server.revision <= revision) throw error;
      const own = [...sent].find((sentProgress) =>
        sameProgress(sentProgress, server.progress),
      );
      if (!own) throw error;
      return { progress: own, revision: server.revision };
    }
  }

  function confirm(saved: {
    progress: OnboardingWizardProgress;
    revision: number;
  }) {
    const previousRevision = revision;
    revision = saved.revision;
    sent.clear();
    const latest = pending ?? saved.progress;
    const local = readLocalProgress(userID);
    const ownsLocal =
      local?.revision === previousRevision &&
      sameProgress(latest, local.progress);
    if (pending === saved.progress) pending = null;
    if (ownsLocal)
      writeLocalProgress(userID, latest, pending !== null, revision);
    failures = 0;
    onError(null);
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
