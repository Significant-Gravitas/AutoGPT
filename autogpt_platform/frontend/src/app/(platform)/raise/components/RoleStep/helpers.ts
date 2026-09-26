import { getExpertCategory } from "@/components/molecules/ExpertAvatar/colors";
import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";

export interface RoleOption {
  id: string;
  label: string;
  // The avatar category — and so the color — a role suggests.
  category: ExpertAvatarRequestCategory;
  // Seeds the name step, so suggestions fit the job the expert was hired for.
  nameSuggestions: string[];
  jobTitleSuggestions: string[];
}

const FALLBACK_NAMES = ["Otto", "Nova", "Juno"];
const ABOUT_PLACEHOLDER =
  "How they should work, what you care about, anything that helps them sound like yours…";

export const CUSTOM_ROLE_MAX_LENGTH = 100;

export const ROLE_OPTIONS: RoleOption[] = [
  {
    id: "marketer",
    category: "marketing",
    label: "Marketer",
    nameSuggestions: ["Echo", "Reach", "Nova"],
    jobTitleSuggestions: [
      "Marketing Manager",
      "Growth Marketer",
      "Content Marketer",
    ],
  },
  {
    id: "sales",
    category: "sales",
    label: "Sales",
    nameSuggestions: ["Pitch", "Ace", "Rain"],
    jobTitleSuggestions: [
      "Sales Development Rep",
      "Account Executive",
      "Sales Manager",
    ],
  },
  {
    id: "developer",
    category: "development",
    label: "Developer",
    nameSuggestions: ["Ada", "Turing", "Bit"],
    jobTitleSuggestions: [
      "Software Engineer",
      "Full-Stack Developer",
      "DevOps Engineer",
    ],
  },
  {
    id: "researcher",
    category: "research",
    label: "Researcher",
    nameSuggestions: ["Kepler", "Curie", "Juno"],
    jobTitleSuggestions: [
      "Research Analyst",
      "Market Researcher",
      "UX Researcher",
    ],
  },
  {
    id: "writer",
    category: "content",
    label: "Writer",
    nameSuggestions: ["Quill", "Hemingway", "Ink"],
    jobTitleSuggestions: ["Content Writer", "Copywriter", "Technical Writer"],
  },
  {
    id: "analyst",
    category: "finance",
    label: "Analyst",
    nameSuggestions: ["Tally", "Vector", "Sigma"],
    jobTitleSuggestions: [
      "Data Analyst",
      "Business Analyst",
      "Financial Analyst",
    ],
  },
  {
    id: "recruiter",
    category: "operations",
    label: "Recruiter",
    nameSuggestions: ["Scout", "Hire", "Vera"],
    jobTitleSuggestions: ["Recruiter", "Talent Sourcer", "Hiring Coordinator"],
  },
  {
    id: "support",
    category: "support",
    label: "Support",
    nameSuggestions: ["Remy", "Aide", "Piper"],
    jobTitleSuggestions: [
      "Support Specialist",
      "Customer Success Manager",
      "Support Engineer",
    ],
  },
  {
    id: "operations",
    category: "operations",
    label: "Operations",
    nameSuggestions: ["Cadence", "Clockwork", "Sol"],
    jobTitleSuggestions: [
      "Operations Manager",
      "Executive Assistant",
      "Project Coordinator",
    ],
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

/** The category the category beat lands on before the user touches it: the one
 *  a preset role stands for, or the topic a typed role reads like. */
export function suggestedCategoryFor(role: string | null) {
  return findRoleOption(role)?.category ?? getExpertCategory(role);
}

export function nameSuggestionsFor(roleId: string | null) {
  return findRoleOption(roleId)?.nameSuggestions ?? FALLBACK_NAMES;
}

export function jobTitleSuggestionsFor(roleId: string | null) {
  return findRoleOption(roleId)?.jobTitleSuggestions ?? [];
}

export function aboutPlaceholderFor(name: string | null) {
  const trimmed = name?.trim();
  if (!trimmed) return ABOUT_PLACEHOLDER;
  return `How ${trimmed} should work, what you care about, anything that helps them sound like yours…`;
}
