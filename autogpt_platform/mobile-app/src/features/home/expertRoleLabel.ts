// Ported from the web's services/experts/expert-role-label.ts and
// EmptySession/helpers.ts so the home sentence reads the same on both.
const ROLE_LABELS: Record<string, string> = {
  "SEO & Content": "SEO Content Manager",
  "Social & Content Repurposing": "Social Media Manager",
  "Market & Competitor Intelligence": "Market Research Analyst",
  "Email & Lifecycle": "Email Marketing Manager",
  "Finance, Invoicing & Bookkeeping": "Bookkeeper",
  "Finance, Fundraising & Investor Relations": "Investor Relations Manager",
  "Research, Data & KPI Analysis": "Data Analyst",
  Sales: "Sales Development Rep",
  "Dependency & Security Hygiene": "Application Security Engineer",
  "Customer Success & Retention": "Customer Success Manager",
  "Deal Desk & Proposal Support": "Deal Desk Manager",
  Ops: "Executive Assistant",
  "Recruiting & Hiring": "Recruiter",
  "Vendor & Procurement": "Procurement Specialist",
  "Contracts (Non-Advisory)": "Contract Manager",
  "Social Media": "Social Media Manager",
};

// A role either names a person ("Marketer") or a domain ("Support"); a domain
// gets "expert" appended so the sentence does not read "your Support".
const PERSON_NOUN_SUFFIXES = ["er", "or", "ist", "yst", "ant", "ian"];
const PERSON_NOUNS = new Set([
  "agent",
  "assistant",
  "chef",
  "chief",
  "expert",
  "head",
  "lead",
  "pro",
  "rep",
  "specialist",
]);

function namesAPerson(role: string) {
  // Only the last word decides: "Customer Success" is a domain.
  const head = role.split(/\s+/).pop()?.toLowerCase() ?? "";
  if (PERSON_NOUNS.has(head)) return true;
  return PERSON_NOUN_SUFFIXES.some((suffix) => head.endsWith(suffix));
}

export function expertRoleLabel(role: string) {
  const displayRole = ROLE_LABELS[role] ?? role;
  return namesAPerson(displayRole) ? displayRole : `${displayRole} expert`;
}
