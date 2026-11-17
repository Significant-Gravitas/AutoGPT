// Relative imports keep this module free of the auth client, so its test loads
// without native modules.
import type { Expert, ExpertTemplate } from "../../lib/api/experts";
import { teamFullMessage } from "../createExpert/helpers";

export type Area =
  | "marketing"
  | "sales"
  | "finance"
  | "support"
  | "operations"
  | "research"
  | "content"
  | "development";

// The backend's marketplace categories (`StoreCategory`), in its order.
export const AREAS: { key: Area; label: string }[] = [
  { key: "marketing", label: "Marketing" },
  { key: "sales", label: "Sales" },
  { key: "finance", label: "Finance" },
  { key: "support", label: "Support" },
  { key: "operations", label: "Operations" },
  { key: "research", label: "Research" },
  { key: "content", label: "Content" },
  { key: "development", label: "Development" },
];

/** Experts in `area` (any area when null) whose name, title or tagline holds the query. */
export function filterTemplates(
  templates: ExpertTemplate[],
  query: string,
  area: Area | null,
) {
  const needle = query.trim().toLowerCase();
  return templates.filter((template) => {
    if (area && !template.categories.includes(area)) return false;
    if (!needle) return true;
    return [template.name, template.job_title, template.role, template.tagline].some(
      (field) => field?.toLowerCase().includes(needle),
    );
  });
}

export function hiredTemplateIDs(experts: Pick<Expert, "source_template_id">[]) {
  return new Set(
    experts.flatMap((expert) => (expert.source_template_id ? [expert.source_template_id] : [])),
  );
}

/** The team list with a just-hired expert in it, once. */
export function withHiredExpert(experts: Expert[] | undefined, hired: Expert) {
  const list = experts ?? [];
  if (!list.some((expert) => expert.id === hired.id)) return [...list, hired];
  return list.map((expert) => (expert.id === hired.id ? hired : expert));
}

interface Refusal {
  status?: unknown;
  code?: unknown;
  limit?: unknown;
}

/** The sentence for a hire that failed for any reason but a dead session. */
export function hireErrorMessage(error: unknown, name: string) {
  const { status, code, limit } =
    typeof error === "object" && error !== null ? (error as Refusal) : {};
  if (status === 409 && code === "active_expert_limit") return teamFullMessage(limit);
  return `Couldn't hire ${name}. Try again.`;
}

export function emptyMessage(query: string, area: Area | null) {
  const trimmed = query.trim();
  if (trimmed) {
    return { title: `No experts match “${trimmed}”`, detail: "Try another word or area." };
  }
  const label = AREAS.find((option) => option.key === area)?.label;
  if (!label) return { title: "No experts yet", detail: "Pull down to check again." };
  return { title: `No ${label} experts yet`, detail: "Try another area." };
}
