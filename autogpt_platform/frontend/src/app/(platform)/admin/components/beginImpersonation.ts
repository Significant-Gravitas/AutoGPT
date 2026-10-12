import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { ImpersonationState } from "@/lib/impersonation";

export type BeginImpersonationOutcome =
  | "started"
  | "no-admin"
  | "audit-failed"
  | "admin-changed";

/**
 * Sends the audit alert, then records the target against the admin who asked.
 * The alert gates the swap, and it can take a while: if that admin signed
 * out, lost the admin role or was replaced in the meantime, nothing is
 * recorded.
 */
export async function beginImpersonation(
  targetId: string,
  sendAuditAlert: () => Promise<unknown>,
): Promise<BeginImpersonationOutcome> {
  const { user: admin, sessionEpoch } = useAuthStore.getState();
  if (!admin || admin.role !== "admin") return "no-admin";

  try {
    await sendAuditAlert();
  } catch {
    return "audit-failed";
  }

  const now = useAuthStore.getState();
  if (
    now.sessionEpoch !== sessionEpoch ||
    now.user?.id !== admin.id ||
    now.user?.role !== "admin"
  ) {
    return "admin-changed";
  }

  ImpersonationState.set(targetId, admin.id);
  return "started";
}
