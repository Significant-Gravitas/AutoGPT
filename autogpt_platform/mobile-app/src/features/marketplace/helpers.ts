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

function isArea(value: string): value is Area {
  return AREAS.some((area) => area.key === value);
}

/** The area a template is listed under: its first category the marketplace knows. */
export function primaryArea(template: Pick<ExpertTemplate, "categories">) {
  return (
    template.categories
      .map((category) => category.toLowerCase())
      .find((category): category is Area => isArea(category)) ?? null
  );
}

export function areaLabel(area: Area) {
  return AREAS.find((option) => option.key === area)?.label ?? area;
}

/** Experts whose job title, role, name or tagline holds the query. */
export function filterTemplates(templates: ExpertTemplate[], query: string) {
  const needle = query.trim().toLowerCase();
  if (!needle) return templates;
  return templates.filter((template) =>
    [template.job_title, template.role, template.name, template.tagline].some((field) =>
      field?.toLowerCase().includes(needle),
    ),
  );
}

export interface AreaSection {
  /** Null for the experts in no area the marketplace knows. */
  area: Area | null;
  label: string;
  templates: ExpertTemplate[];
  /** Someone on the team was hired from this area. */
  isCovered: boolean;
}

/** An area nobody on the team covers, shown as an empty seat on the team card. */
export interface OpenSeat {
  area: Area;
  label: string;
}

const OTHER_LABEL = "More experts";

/**
 * One section per area that has templates, each template under its first
 * area only: areas nobody on the team covers first, then the rest, each part
 * in `AREAS` order. Templates in no known area come last, so none is hidden.
 */
export function areaSections(templates: ExpertTemplate[], hiredIDs: ReadonlySet<string>) {
  const sections: AreaSection[] = AREAS.map(({ key, label }) => {
    const inArea = templates.filter((template) => primaryArea(template) === key);
    return {
      area: key,
      label,
      templates: inArea,
      isCovered: inArea.some((template) => hiredIDs.has(template.id)),
    };
  }).filter((section) => section.templates.length > 0);
  const others = templates.filter((template) => primaryArea(template) === null);
  return [
    ...sections.filter((section) => !section.isCovered),
    ...sections.filter((section) => section.isCovered),
    // Not an area a team can be missing, so it never shows as an open seat.
    ...(others.length > 0
      ? [{ area: null, label: OTHER_LABEL, templates: others, isCovered: true }]
      : []),
  ];
}

export function openSeats(sections: AreaSection[]): OpenSeat[] {
  return sections.flatMap(({ area, label, isCovered }) =>
    area && !isCovered ? [{ area, label }] : [],
  );
}

/** "Jordan, finds leads that fit…": the name, then the tagline as the rest of the sentence. */
export function whoLine(template: Pick<ExpertTemplate, "name" | "tagline">) {
  const tagline = template.tagline?.trim();
  if (!tagline) return template.name;
  // Keep words that open in capitals, such as "SEO" or "I".
  const keepsCase = /^[A-Z]([A-Z]|\s|['’]|$)/.test(tagline);
  const rest = keepsCase ? tagline : tagline.charAt(0).toLowerCase() + tagline.slice(1);
  return `${template.name}, ${rest}`;
}

export function firstName(name: string) {
  return name.trim().split(/\s+/)[0] ?? name;
}

export function hiredTemplateIDs(experts: Pick<Expert, "source_template_id">[]) {
  return new Set(
    experts.flatMap((expert) => (expert.source_template_id ? [expert.source_template_id] : [])),
  );
}

/**
 * The experts hired from the marketplace: the team list's, then any hired in
 * this sign-in that the list doesn't have yet.
 */
export function hiredTeam(experts: Expert[], justHired: ReadonlyMap<string, Expert>) {
  const fromList = experts.filter((expert) => expert.source_template_id !== null);
  const pending = [...justHired]
    .filter(([, hired]) => !experts.some((expert) => expert.id === hired.id))
    .map(([templateID, hired]) => ({
      ...hired,
      source_template_id: hired.source_template_id ?? templateID,
    }));
  return [...fromList, ...pending];
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
