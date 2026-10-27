import type { MicrosoftAccessCheck } from "@/app/api/__generated__/models/microsoftAccessCheck";

/** Every permission the person can grant themselves right now, so one
 *  consent screen covers all of them. */
export function pendingScopes(check?: MicrosoftAccessCheck): string[] {
  if (!check) return [];
  return Array.from(
    new Set(
      check.capabilities
        .filter((item) => item.status === "permission_needed")
        .flatMap((item) => item.scopes ?? []),
    ),
  );
}

export function needsAdmin(check?: MicrosoftAccessCheck): boolean {
  return Boolean(
    check?.capabilities.some((item) => item.status === "admin_required"),
  );
}

export function needsWorkIQSetup(check?: MicrosoftAccessCheck): boolean {
  return Boolean(
    check?.capabilities.some(
      (item) => item.service === "work_iq" && item.status === "setup_required",
    ),
  );
}
