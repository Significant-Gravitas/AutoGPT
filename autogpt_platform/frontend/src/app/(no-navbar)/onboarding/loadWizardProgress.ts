import { getV1OnboardingState } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import type { OnboardingWizardProgress } from "@/app/api/__generated__/models/onboardingWizardProgress";
import {
  clearLocalProgress,
  readLocalProgress,
  readProgress,
  sameProgress,
  writeLocalProgress,
} from "./progress";

type LoadedWizardProgress =
  | { kind: "complete" | "accountChanged" | "conflict" | "unavailable" }
  | {
      kind: "ready";
      progress: OnboardingWizardProgress | null;
      revision: number;
      offline: boolean;
      needsSync: boolean;
    };

export async function loadWizardProgress(
  userID: string,
  reloadLatest: boolean,
  abort: AbortController,
): Promise<LoadedWizardProgress> {
  const cached = readLocalProgress(userID);
  const local = reloadLatest ? null : cached;
  try {
    const timeout = setTimeout(() => abort.abort(), 10_000);
    let result;
    try {
      result = await getV1OnboardingState({ signal: abort.signal });
    } finally {
      clearTimeout(timeout);
    }
    if (result.status !== 200) throw new Error("Could not load onboarding.");
    if (result.data.userId !== userID) return { kind: "accountChanged" };
    if (result.data.completedSteps.includes("ONBOARDING_COMPLETE"))
      return { kind: "complete" };
    const revision = result.data.wizardRevision ?? 0;
    const serverProgress = readProgress(result.data.wizardProgress);
    if (
      local?.pending &&
      local.revision !== revision &&
      !sameProgress(local.progress, serverProgress)
    ) {
      return { kind: "conflict" };
    }
    const needsSync =
      !!local?.pending && !sameProgress(local.progress, serverProgress);
    if (!needsSync && !abort.signal.aborted)
      cacheConfirmedProgress(userID, serverProgress, revision, cached);
    return {
      kind: "ready",
      progress: local?.pending ? local.progress : serverProgress,
      revision,
      offline: false,
      needsSync,
    };
  } catch {
    return local
      ? {
          kind: "ready",
          progress: local.progress,
          revision: local.revision,
          offline: true,
          needsSync: local.pending,
        }
      : { kind: "unavailable" };
  }
}

function cacheConfirmedProgress(
  userID: string,
  progress: OnboardingWizardProgress | null,
  revision: number,
  cached: ReturnType<typeof readLocalProgress>,
) {
  const current = readLocalProgress(userID);
  if (
    current?.revision !== cached?.revision ||
    current?.pending !== cached?.pending ||
    (current && !sameProgress(current.progress, cached?.progress ?? null))
  )
    return;
  if (progress) writeLocalProgress(userID, progress, false, revision);
  else clearLocalProgress(userID);
}
