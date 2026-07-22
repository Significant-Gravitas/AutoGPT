import type { SelectOption } from "@/components/atoms/Select/Select";

// The grant's principal is always a team in this UI (org-home / user grants
// aren't offered here).
export const PRINCIPAL_TYPE_TEAM = "TEAM";

// What a shared team can do with the agent. EXECUTE implies VIEW on the
// backend, so the two are offered as a single either/or choice.
export const GrantCapability = {
  View: "VIEW",
  Execute: "EXECUTE",
} as const;

// Whose connected accounts a shared run uses. CONSUMER = the team runs with
// their own credentials; OWNER = runs use the sharer's credentials.
export const CredentialMode = {
  Consumer: "CONSUMER",
  Owner: "OWNER",
} as const;

export const capabilityOptions: SelectOption[] = [
  { value: GrantCapability.View, label: "Can view" },
  { value: GrantCapability.Execute, label: "Can run" },
];

export const credentialModeOptions: SelectOption[] = [
  { value: CredentialMode.Consumer, label: "Run with their credentials" },
  { value: CredentialMode.Owner, label: "Run with my credentials" },
];

export function capabilityLabel(capability: string): string {
  return capability === GrantCapability.Execute ? "Can run" : "Can view";
}
