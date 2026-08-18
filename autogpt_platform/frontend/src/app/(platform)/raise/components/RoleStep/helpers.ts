export interface RoleOption {
  id: string;
  label: string;
  // Seeds the name step, so suggestions fit the job the expert was hired for.
  nameSuggestions: string[];
}

const FALLBACK_NAMES = ["Otto", "Nova", "Juno"];
const ABOUT_PLACEHOLDER =
  "How they should work, what you care about, anything that helps them sound like yours…";

export const CUSTOM_ROLE_MAX_LENGTH = 100;

export const ROLE_OPTIONS: RoleOption[] = [
  {
    id: "marketer",
    label: "Marketer",
    nameSuggestions: ["Echo", "Reach", "Nova"],
  },
  {
    id: "sales",
    label: "Sales",
    nameSuggestions: ["Pitch", "Ace", "Rain"],
  },
  {
    id: "developer",
    label: "Developer",
    nameSuggestions: ["Ada", "Turing", "Bit"],
  },
  {
    id: "researcher",
    label: "Researcher",
    nameSuggestions: ["Kepler", "Curie", "Juno"],
  },
  {
    id: "writer",
    label: "Writer",
    nameSuggestions: ["Quill", "Hemingway", "Ink"],
  },
  {
    id: "analyst",
    label: "Analyst",
    nameSuggestions: ["Tally", "Vector", "Sigma"],
  },
  {
    id: "recruiter",
    label: "Recruiter",
    nameSuggestions: ["Scout", "Hire", "Vera"],
  },
  {
    id: "support",
    label: "Support",
    nameSuggestions: ["Remy", "Aide", "Piper"],
  },
  {
    id: "operations",
    label: "Operations",
    nameSuggestions: ["Cadence", "Clockwork", "Sol"],
  },
];

export function normalizeCustomRole(value: string) {
  return value.trim();
}

export function isValidCustomRole(value: string) {
  const trimmed = normalizeCustomRole(value);
  return trimmed.length > 0 && trimmed.length <= CUSTOM_ROLE_MAX_LENGTH;
}

export function findRoleOption(id: string | null) {
  return ROLE_OPTIONS.find((option) => option.id === id) ?? null;
}

export function roleLabelFor(role: string | null) {
  if (!role) return null;
  return findRoleOption(role)?.label ?? role;
}

export function roleOptionsForSelection(selectedRole: string | null) {
  if (!selectedRole) return ROLE_OPTIONS;
  const preset = findRoleOption(selectedRole);
  if (preset) return [preset];
  return [{ id: selectedRole, label: selectedRole }];
}

export function nameSuggestionsFor(roleId: string | null) {
  return findRoleOption(roleId)?.nameSuggestions ?? FALLBACK_NAMES;
}

export function aboutPlaceholderFor(name: string | null) {
  const trimmed = name?.trim();
  if (!trimmed) return ABOUT_PLACEHOLDER;
  return `How ${trimmed} should work, what you care about, anything that helps them sound like yours…`;
}
