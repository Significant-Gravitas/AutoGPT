const ROLE_LABELS: Record<string, string> = {
  "Social & Content Repurposing": "Social Media",
  "Market & Competitor Intelligence": "Market Intelligence",
};

export function getExpertRoleLabel(role: string): string {
  return ROLE_LABELS[role] ?? role;
}
