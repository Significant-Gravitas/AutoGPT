// The "General" tab keeps the internal `profile` id so existing ?tab=profile
// deep links stay valid; only the visible label changed.
export type OrgSettingsTab = "profile" | "members" | "invitations" | "billing";

interface TabAccess {
  isAdmin: boolean;
  canManageBilling: boolean;
}

export function getAllowedTabs({
  isAdmin,
  canManageBilling,
}: TabAccess): OrgSettingsTab[] {
  const tabs: OrgSettingsTab[] = ["profile", "members"];
  if (isAdmin) tabs.push("invitations");
  if (canManageBilling) tabs.push("billing");
  return tabs;
}

// A ?tab= deep link (or a tab picked before a role change) that the viewer may
// not open falls back to General rather than an empty panel.
export function resolveTab(
  requested: string | null,
  allowed: OrgSettingsTab[],
): OrgSettingsTab {
  return allowed.find((tab) => tab === requested) ?? "profile";
}
