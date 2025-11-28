import BackendAPI from "@/lib/autogpt-server-api";

/**
 * Narrow an orval response to its success payload if and only if it is a `200` status with OK shape.
 *
 * Usage with React Query select:
 * ```ts
 *   const { data: agent } = useGetV2GetLibraryAgent(agentId, {
 *     query: { select: okData<LibraryAgent> },
 *   });
 *
 *   data // is now properly typed as LibraryAgent | undefined
 * ```
 */
export function okData<T>(res: unknown): T | undefined {
  if (!res || typeof res !== "object") return undefined;

  // status must exist and be exactly 200
  const maybeStatus = (res as { status?: unknown }).status;
  if (maybeStatus !== 200) return undefined;

  // data must exist and be an object/array/string/number/etc. We only need to
  // check presence to safely return it as T; the generic T is enforced at call sites.
  if (!("data" in (res as Record<string, unknown>))) return undefined;

  return (res as { data: T }).data;
}

export async function shouldShowOnboarding() {
  const api = new BackendAPI();
  const isEnabled = await api.isOnboardingEnabled();
  const onboarding = await api.getUserOnboarding();
  const isCompleted = onboarding.completedSteps.includes("CONGRATS");
  return isEnabled && !isCompleted;
}
