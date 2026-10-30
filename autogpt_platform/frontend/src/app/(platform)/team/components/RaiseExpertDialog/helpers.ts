import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";

interface AreaDetails {
  // Saved as the expert's role id, which the roster and avatar colours read.
  role: string;
  color: string;
  keywords: RegExp;
}

// Checked in order, so a narrow area wins over a broad one ("sales analyst"
// is sales, not finance). Matching runs over the name and the role together:
// a "Deps merger" says development without a role at all.
const AREAS: [ExpertAvatarRequestCategory, AreaDetails][] = [
  [
    "development",
    {
      role: "Developer",
      color: "blue-300",
      keywords:
        /\b(dev\w*|deps?|dependenc\w*|merge\w*|engineer\w*|code\w*|coder|program\w*|software|github|git|pull requests?|prs?|bugs?|qa|tests?|testing|release\w*|deploy\w*|infra\w*|api|backend|frontend|full[- ]?stack|ci|build\w*|security)\b/i,
    },
  ],
  [
    "sales",
    {
      role: "Sales",
      color: "amber-300",
      keywords:
        /\b(sales|sdr|bdr|leads|lead gen\w*|prospect\w*|outreach|deals?|pipeline|closer|account exec\w*|crm)\b/i,
    },
  ],
  [
    "marketing",
    {
      role: "Marketer",
      color: "orange-300",
      keywords:
        /\b(market\w*|seo|growth|social|campaigns?|brand\w*|ads|advertis\w*|launch\w*|community)\b/i,
    },
  ],
  [
    "support",
    {
      role: "Support",
      color: "rose-300",
      keywords:
        /\b(support|customers?|tickets?|help ?desk|success|inbox|onboard\w*)\b/i,
    },
  ],
  [
    "finance",
    {
      role: "Analyst",
      color: "green-300",
      keywords:
        /\b(financ\w*|invoic\w*|account\w*|bookkeep\w*|budget\w*|tax\w*|expenses?|payroll|analyst|metrics|revenue)\b/i,
    },
  ],
  [
    "content",
    {
      role: "Writer",
      color: "amber-300",
      keywords:
        /\b(writ\w*|content|copy\w*|blog\w*|editor|newsletters?|docs|documentation|scribe|posts?)\b/i,
    },
  ],
  [
    "research",
    {
      role: "Researcher",
      color: "lime-300",
      keywords:
        /\b(research\w*|competitor\w*|insights?|scout|analysis|trends?|intel\w*)\b/i,
    },
  ],
  [
    "operations",
    {
      role: "Operations",
      color: "sky-300",
      keywords:
        /\b(ops|operations?|recruit\w*|hiring|hr|schedul\w*|calendar|admin\w*|assistant|logistics|planner|coordinator)\b/i,
    },
  ],
];

const GENERAL: AreaDetails = {
  role: "Assistant",
  color: "amber-300",
  keywords: /$^/,
};

export interface RaiseArea {
  category: ExpertAvatarRequestCategory;
  role: string;
  color: string;
}

/** The work area a name and role point at. It picks the expert's colour and
 *  the family its generated avatar belongs to, so it is never asked for. */
export function inferArea(name: string, jobTitle: string): RaiseArea {
  const text = `${name} ${jobTitle}`;
  const match = AREAS.find(([, area]) => area.keywords.test(text));
  if (!match) {
    return { category: "general", role: GENERAL.role, color: GENERAL.color };
  }
  const [category, area] = match;
  return { category, role: area.role, color: area.color };
}

// A `/raise?role=` link from Otto's greeting names an area by its role id.
const ROLE_LINK_TITLES: Record<string, string> = {
  marketer: "Marketer",
  sales: "Sales rep",
  developer: "Software engineer",
  researcher: "Researcher",
  writer: "Writer",
  analyst: "Analyst",
  recruiter: "Recruiter",
  support: "Support specialist",
  operations: "Operations manager",
};

export function jobTitleForRoleLink(role: string | null): string {
  return role ? (ROLE_LINK_TITLES[role] ?? "") : "";
}

export const MAX_NAME_LENGTH = 100;
export const MAX_JOB_TITLE_LENGTH = 100;
export const MAX_IMAGE_MB = 4;
export const ACCEPTED_IMAGE_TYPES = "image/png,image/jpeg,image/webp";
