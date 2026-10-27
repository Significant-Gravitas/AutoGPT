import type { MicrosoftAccessCheck } from "@/app/api/__generated__/models/microsoftAccessCheck";
import type { MicrosoftCapability } from "@/app/api/__generated__/models/microsoftCapability";

/** Every permission the person can grant themselves right now, so one
 *  consent screen covers all of them. Work IQ keeps its own Allow: until
 *  Microsoft has set it up for the organization, asking for it fails
 *  everything asked with it, and nothing shows that before someone asks. */
export function pendingScopes(check?: MicrosoftAccessCheck): string[] {
  if (!check) return [];
  return Array.from(
    new Set(
      check.capabilities
        .filter(
          (item) =>
            item.status === "permission_needed" && item.service !== "work_iq",
        )
        .flatMap((item) => item.scopes ?? []),
    ),
  );
}

/** Whether a service gets its own Allow. Work IQ's also shows when an admin
 *  must approve it: an admin approving it there can consent for everyone. */
export function canAllow(item: MicrosoftCapability): boolean {
  if (!item.scopes?.length) return false;
  if (item.status === "permission_needed") return true;
  return item.service === "work_iq" && item.status === "admin_required";
}

export function workIQToAllow(
  check?: MicrosoftAccessCheck,
): MicrosoftCapability | undefined {
  return check?.capabilities.find(
    (item) => item.service === "work_iq" && canAllow(item),
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
