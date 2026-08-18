export interface RoleOption {
  id: string;
  label: string;
  // Seeds the name step, so suggestions fit the job the expert was hired for.
  nameSuggestions: string[];
  aboutPlaceholder: string;
}

const FALLBACK_NAMES = ["Otto", "Nova", "Juno"];
const FALLBACK_PLACEHOLDER =
  "What they should know, how they should work, anything at all…";

export const ROLE_OPTIONS: RoleOption[] = [
  {
    id: "marketer",
    label: "Marketer",
    nameSuggestions: ["Echo", "Reach", "Nova"],
    aboutPlaceholder:
      "Who you're targeting, the channels you care about, the voice your brand uses…",
  },
  {
    id: "sales",
    label: "Sales",
    nameSuggestions: ["Pitch", "Ace", "Rain"],
    aboutPlaceholder:
      "Your ideal customer, how you qualify leads, what a good follow-up looks like…",
  },
  {
    id: "developer",
    label: "Developer",
    nameSuggestions: ["Ada", "Turing", "Bit"],
    aboutPlaceholder:
      "Your stack, the repos they'll touch, review standards they should hold…",
  },
  {
    id: "researcher",
    label: "Researcher",
    nameSuggestions: ["Kepler", "Curie", "Juno"],
    aboutPlaceholder:
      "Which sources to trust, how deep to go, what a good answer looks like…",
  },
  {
    id: "writer",
    label: "Writer",
    nameSuggestions: ["Quill", "Hemingway", "Ink"],
    aboutPlaceholder:
      "Who you're writing for, the tone you like, words to avoid…",
  },
  {
    id: "analyst",
    label: "Analyst",
    nameSuggestions: ["Tally", "Vector", "Sigma"],
    aboutPlaceholder:
      "Where the numbers live, what you measure, how you like them cut…",
  },
  {
    id: "recruiter",
    label: "Recruiter",
    nameSuggestions: ["Scout", "Hire", "Vera"],
    aboutPlaceholder:
      "Roles you're filling, what a strong candidate looks like, your screening bar…",
  },
  {
    id: "support",
    label: "Support",
    nameSuggestions: ["Remy", "Aide", "Piper"],
    aboutPlaceholder:
      "Your product, common questions, when to escalate to a human…",
  },
  {
    id: "operations",
    label: "Operations",
    nameSuggestions: ["Cadence", "Clockwork", "Sol"],
    aboutPlaceholder:
      "The processes they'll run, your working hours, what needs your sign-off…",
  },
  {
    id: "other",
    label: "Something else",
    nameSuggestions: FALLBACK_NAMES,
    aboutPlaceholder: FALLBACK_PLACEHOLDER,
  },
];

export function findRoleOption(id: string | null) {
  return ROLE_OPTIONS.find((option) => option.id === id) ?? null;
}

export function nameSuggestionsFor(roleId: string | null) {
  return findRoleOption(roleId)?.nameSuggestions ?? FALLBACK_NAMES;
}

export function aboutPlaceholderFor(roleId: string | null) {
  return findRoleOption(roleId)?.aboutPlaceholder ?? FALLBACK_PLACEHOLDER;
}
