const ROLE_LABELS: Record<string, string> = {
  "Social & Content Repurposing": "Social media",
  "Social Media": "Social media",
  "Market & Competitor Intelligence": "Market intelligence",
};

export function getExpertRoleLabel(role: string): string {
  return ROLE_LABELS[role] ?? role;
}
