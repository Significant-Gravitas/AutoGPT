// Shared resolution of the wizard's "Other" / "Something else" sentinels into
// real values. Both `useSubscriptionStep` (pre-Stripe-redirect) and
// `useOnboardingPage` (Preparing-step submit) post the profile, and label or
// mapping changes must stay in sync across the two.

interface ProfileSource {
  name: string;
  role: string;
  otherRole: string;
  painPoints: string[];
  otherPainPoint: string;
}

interface NormalizedProfile {
  name: string;
  role: string;
  painPoints: string[];
}

export function normalizeOnboardingProfile(
  state: ProfileSource,
): NormalizedProfile {
  const resolvedRole = state.role === "Other" ? state.otherRole : state.role;
  const resolvedPainPoints = state.painPoints
    .filter((p) => p !== "Something else")
    .concat(
      state.painPoints.includes("Something else") && state.otherPainPoint.trim()
        ? [state.otherPainPoint.trim()]
        : [],
    );
  return {
    name: state.name,
    role: resolvedRole,
    painPoints: resolvedPainPoints,
  };
}
