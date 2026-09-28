import { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { EXPERT_AVATARS } from "@/components/molecules/ExpertAvatar/helpers";

interface CategoryDetails {
  label: string;
  // Saved as the expert's role, so the intro reads "I'm Curie, your Researcher".
  role: string;
  // Seeds the name step, so suggestions fit the area the expert works in.
  nameSuggestions: string[];
  jobTitleSuggestions: string[];
}

const CATEGORY_DETAILS: Record<ExpertAvatarRequestCategory, CategoryDetails> = {
  marketing: {
    label: "Marketing",
    role: "Marketer",
    nameSuggestions: ["Echo", "Reach", "Nova"],
    jobTitleSuggestions: [
      "Marketing Manager",
      "Growth Marketer",
      "Content Marketer",
    ],
  },
  sales: {
    label: "Sales",
    role: "Sales",
    nameSuggestions: ["Pitch", "Ace", "Rain"],
    jobTitleSuggestions: [
      "Sales Development Rep",
      "Account Executive",
      "Sales Manager",
    ],
  },
  finance: {
    label: "Finance",
    role: "Analyst",
    nameSuggestions: ["Tally", "Vector", "Sigma"],
    jobTitleSuggestions: [
      "Financial Analyst",
      "Data Analyst",
      "Business Analyst",
    ],
  },
  support: {
    label: "Support",
    role: "Support",
    nameSuggestions: ["Remy", "Aide", "Piper"],
    jobTitleSuggestions: [
      "Support Specialist",
      "Customer Success Manager",
      "Support Engineer",
    ],
  },
  operations: {
    label: "Operations",
    role: "Operations",
    nameSuggestions: ["Cadence", "Clockwork", "Scout"],
    jobTitleSuggestions: [
      "Operations Manager",
      "Executive Assistant",
      "Recruiter",
    ],
  },
  research: {
    label: "Research",
    role: "Researcher",
    nameSuggestions: ["Kepler", "Curie", "Juno"],
    jobTitleSuggestions: [
      "Research Analyst",
      "Market Researcher",
      "UX Researcher",
    ],
  },
  content: {
    label: "Content",
    role: "Writer",
    nameSuggestions: ["Quill", "Hemingway", "Ink"],
    jobTitleSuggestions: ["Content Writer", "Copywriter", "Technical Writer"],
  },
  development: {
    label: "Development",
    role: "Developer",
    nameSuggestions: ["Ada", "Turing", "Bit"],
    jobTitleSuggestions: [
      "Software Engineer",
      "Full-Stack Developer",
      "DevOps Engineer",
    ],
  },
};

// The role ids a `/raise?role=` link can carry; keep in step with the
// backend's RAISE_ROLES in onboarding_dump/recommend_experts.py.
const ROLE_CATEGORIES = new Map<string, ExpertAvatarRequestCategory>([
  ["marketer", "marketing"],
  ["sales", "sales"],
  ["developer", "development"],
  ["researcher", "research"],
  ["writer", "content"],
  ["analyst", "finance"],
  ["recruiter", "operations"],
  ["support", "support"],
  ["operations", "operations"],
]);

const FALLBACK_NAMES = ["Otto", "Nova", "Juno"];

export interface CategoryOption {
  id: ExpertAvatarRequestCategory;
  label: string;
  hex: string;
}

export const CATEGORY_OPTIONS: CategoryOption[] = Object.values(
  ExpertAvatarRequestCategory,
).map((id) => ({
  id,
  label: CATEGORY_DETAILS[id].label,
  hex: EXPERT_AVATARS.find((avatar) => avatar.id === id)?.hex ?? "#B5ADA0",
}));

export function categoryOptionsForSelection(
  selected: ExpertAvatarRequestCategory | null,
) {
  if (!selected) return CATEGORY_OPTIONS;
  return CATEGORY_OPTIONS.filter((option) => option.id === selected);
}

/** The wizard color a category answers for, used from the first beat onwards. */
export function colorForCategory(category: ExpertAvatarRequestCategory) {
  return (
    EXPERT_AVATARS.find((avatar) => avatar.id === category)?.color ??
    "amber-300"
  );
}

export function categoryForRole(role: string) {
  return ROLE_CATEGORIES.get(role) ?? null;
}

export function roleFor(category: ExpertAvatarRequestCategory | null) {
  return category ? CATEGORY_DETAILS[category].role : null;
}

export function nameSuggestionsFor(
  category: ExpertAvatarRequestCategory | null,
) {
  return category ? CATEGORY_DETAILS[category].nameSuggestions : FALLBACK_NAMES;
}

export function jobTitleSuggestionsFor(
  category: ExpertAvatarRequestCategory | null,
) {
  return category ? CATEGORY_DETAILS[category].jobTitleSuggestions : [];
}
