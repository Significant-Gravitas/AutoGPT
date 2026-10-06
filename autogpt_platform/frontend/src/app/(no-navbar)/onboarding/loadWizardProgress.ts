import { getV1OnboardingState } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import type { OnboardingWizardProgress } from "@/app/api/__generated__/models/onboardingWizardProgress";
import { readLocalProgress, readProgress, sameProgress } from "./progress";

type LoadedWizardProgress =
  | { kind: "complete" | "accountChanged" | "conflict" | "unavailable" }
  | {
      kind: "ready";
      progress: OnboardingWizardProgress | null;
      revision: number;
      offline: boolean;
    };

export async function loadWizardProgress(
  userID: string,
  reloadLatest: boolean,
  abort: AbortController,
): Promise<LoadedWizardProgress> {
  const local = reloadLatest ? null : readLocalProgress(userID);
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
    return {
      kind: "ready",
      progress: local?.pending ? local.progress : serverProgress,
      revision,
      offline: false,
    };
  } catch {
    return local
      ? {
          kind: "ready",
          progress: local.progress,
          revision: local.revision,
          offline: true,
        }
      : { kind: "unavailable" };
  }
}
