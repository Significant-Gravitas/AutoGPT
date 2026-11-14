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

export function getExpertRoleLabel(role: string): string {
  return ROLE_LABELS[role] ?? role;
}
