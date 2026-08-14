import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Briefcase01Icon,
  ChartIncreaseIcon,
  Megaphone01Icon,
  Settings01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

export interface ExpertAccent {
  wash: string;
  /** Full-bleed variant for wide surfaces (e.g. the expert page header) —
   *  the radial `wash` is sized for the ~640px sheet and fades out halfway
   *  across a full-width card. */
  washWide: string;
  pill: string;
  icon: string;
  roleIcon: IconSvgElement;
}

const ACCENTS: Record<string, ExpertAccent> = {
  violet: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(139,92,246,0.10),transparent_70%)]",
    washWide:
      "bg-[linear-gradient(180deg,rgba(139,92,246,0.10),rgba(139,92,246,0.03)_60%,transparent)]",
    pill: "bg-violet-50 text-violet-700 ring-1 ring-inset ring-violet-600/10",
    icon: "text-violet-500",
    roleIcon: Megaphone01Icon,
  },
  amber: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(245,158,11,0.10),transparent_70%)]",
    washWide:
      "bg-[linear-gradient(180deg,rgba(245,158,11,0.10),rgba(245,158,11,0.03)_60%,transparent)]",
    pill: "bg-amber-50 text-amber-700 ring-1 ring-inset ring-amber-600/10",
    icon: "text-amber-500",
    roleIcon: ChartIncreaseIcon,
  },
  sky: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(14,165,233,0.10),transparent_70%)]",
    washWide:
      "bg-[linear-gradient(180deg,rgba(14,165,233,0.10),rgba(14,165,233,0.03)_60%,transparent)]",
    pill: "bg-sky-50 text-sky-700 ring-1 ring-inset ring-sky-600/10",
    icon: "text-sky-500",
    roleIcon: Settings01Icon,
  },
  zinc: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(113,113,122,0.08),transparent_70%)]",
    washWide:
      "bg-[linear-gradient(180deg,rgba(113,113,122,0.08),rgba(113,113,122,0.02)_60%,transparent)]",
    pill: "bg-zinc-100 text-zinc-600 ring-1 ring-inset ring-zinc-500/10",
    icon: "text-zinc-500",
    roleIcon: Briefcase01Icon,
  },
};

interface RoleTheme {
  accent: string;
  avatar: string;
}

/** Role taxonomy for the roster. Each role maps to a colour accent and the
 *  committed persona avatar under `/public/experts`, so an expert with no
 *  `avatar_url` still shows a fitting face instead of a marble gradient. */
const ROLE_THEMES: Array<[RegExp, RoleTheme]> = [
  [
    /marketing|growth|brand/i,
    { accent: "violet", avatar: "/experts/maria.svg" },
  ],
  [/sales|revenue/i, { accent: "amber", avatar: "/experts/max.svg" }],
  [
    /ops|operations|support/i,
    { accent: "sky", avatar: "/experts/frankie.svg" },
  ],
];

function matchRoleTheme(role: string): RoleTheme | null {
  for (const [pattern, theme] of ROLE_THEMES) {
    if (pattern.test(role)) return theme;
  }
  return null;
}

export function getExpertAccent(role: string): ExpertAccent {
  const theme = matchRoleTheme(role);
  return theme ? ACCENTS[theme.accent] : ACCENTS.zinc;
}

/** Prefer the expert's own avatar, then a role-based persona avatar, so the
 *  marble gradient only appears for experts whose role we can't place. */
export function getExpertAvatarUrl(
  expert: Pick<Expert, "avatar_url" | "role">,
): string | null {
  if (expert.avatar_url) return expert.avatar_url;
  return matchRoleTheme(expert.role)?.avatar ?? null;
}

export function getExpertFirstName(name: string): string {
  return name.trim().split(/\s+/)[0] || name;
}
