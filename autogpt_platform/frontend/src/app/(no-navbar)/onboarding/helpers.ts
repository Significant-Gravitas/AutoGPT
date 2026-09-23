import type { User } from "@/lib/auth/types";

// Resolution of the wizard's "Other" / "Something else" sentinels into real
// values for the Preparing-step profile submit.

interface ProfileSource {
  role: string;
  otherRole: string;
  painPoints: string[];
  otherPainPoint: string;
}

interface NormalizedProfile {
  role: string;
  painPoints: string[];
}

// The wizard no longer asks for a name; the profile carries whatever the
// account already knows — the same precedence the copilot greeting uses.
export function accountDisplayName(user: User | null | undefined): string {
  const preferred = user?.user_metadata.preferred_name?.trim();
  if (preferred) return preferred;
  const name = user?.user_metadata.name?.trim();
  if (name) return name.split(" ")[0];
  return user?.email.split("@")[0] ?? "";
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
    role: resolvedRole,
    painPoints: resolvedPainPoints,
  };
}
