// Org role ladder. Each selectable role maps to the two backend boolean flags
// (is_admin, is_billing_manager). Admin and billing are DISJOINT capabilities in
// the enforced permission matrix (autogpt_libs/auth/permissions.py: MANAGE_BILLING
// is owner + billing_manager only — admins do NOT get it), so all four
// flag combinations are meaningful roles.
//
// Capabilities below are derived directly from _ORG_PERMISSIONS in
// autogpt_libs/auth/permissions.py; keep them in sync if that map changes.

export const ORG_ROLE_VALUES = [
  "member",
  "billing_manager",
  "admin",
  "admin_billing",
] as const;

export type OrgRole = (typeof ORG_ROLE_VALUES)[number];

export interface OrgRoleFlags {
  is_admin: boolean;
  is_billing_manager: boolean;
}

export interface OrgRoleDefinition {
  value: OrgRole;
  label: string;
  flags: OrgRoleFlags;
  // Short, human-readable summary of what the role can do.
  capabilities: string[];
}

export const ORG_ROLES: OrgRoleDefinition[] = [
  {
    value: "member",
    label: "Member",
    flags: { is_admin: false, is_billing_manager: false },
    capabilities: [
      "View the organization",
      "Create and share resources",
      "Publish to the store",
    ],
  },
  {
    value: "billing_manager",
    label: "Billing manager",
    flags: { is_admin: false, is_billing_manager: true },
    capabilities: ["View the organization", "Manage billing", "Create teams"],
  },
  {
    value: "admin",
    label: "Admin",
    flags: { is_admin: true, is_billing_manager: false },
    capabilities: [
      "Rename org, manage members & teams",
      "Transfer resources",
      "Create and share resources, publish to the store",
      "No billing access",
    ],
  },
  {
    value: "admin_billing",
    label: "Admin & billing",
    flags: { is_admin: true, is_billing_manager: true },
    capabilities: ["Everything an admin can do", "Manage billing"],
  },
];

// The org owner is never selectable here — ownership is transferred separately in
// the danger zone — so the owner role has no entry in the ladder above; owner
// rows render a static "Owner" badge instead of this dropdown.

const ORG_ROLE_BY_VALUE = new Map<OrgRole, OrgRoleDefinition>(
  ORG_ROLES.map((role) => [role.value, role]),
);

export const ORG_ROLE_OPTIONS = ORG_ROLES.map((role) => ({
  value: role.value,
  label: role.label,
}));

export function flagsToRole(flags: OrgRoleFlags): OrgRole {
  if (flags.is_admin && flags.is_billing_manager) return "admin_billing";
  if (flags.is_admin) return "admin";
  if (flags.is_billing_manager) return "billing_manager";
  return "member";
}

export function roleToFlags(role: OrgRole): OrgRoleFlags {
  return getRole(role).flags;
}

export function roleLabel(role: OrgRole): string {
  return getRole(role).label;
}

// Markdown bullet list of the role's capabilities, for the info tooltip.
export function roleCapabilitiesMarkdown(role: OrgRole): string {
  const def = getRole(role);
  const bullets = def.capabilities.map((cap) => `- ${cap}`).join("\n");
  return `**${def.label}**\n\n${bullets}`;
}

function getRole(role: OrgRole): OrgRoleDefinition {
  const def = ORG_ROLE_BY_VALUE.get(role);
  if (!def) throw new Error(`Unknown org role: ${role}`);
  return def;
}
