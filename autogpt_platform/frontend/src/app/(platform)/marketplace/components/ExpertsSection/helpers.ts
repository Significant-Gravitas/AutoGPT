import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
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
}

/** Role taxonomy for the roster's colour and icon treatment. */
const ROLE_THEMES: Array<[RegExp, RoleTheme]> = [
  [/marketing|growth|brand/i, { accent: "violet" }],
  [/sales|revenue/i, { accent: "amber" }],
  [/ops|operations|support/i, { accent: "sky" }],
];

const PERSONA_AVATARS = new Map([
  ["maria", "/experts/maria.svg"],
  ["max", "/experts/max.svg"],
  ["frankie", "/experts/frankie.svg"],
]);

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

/** Prefer the expert's own avatar, then a known seed persona's committed
 *  avatar. Role alone cannot select a face: unrelated experts can share a
 *  role without sharing an identity. */
export function getExpertAvatarUrl(
  expert: Pick<Expert, "avatar_url" | "name">,
): string | null {
  const customUrl = expert.avatar_url?.trim();
  if (customUrl) return customUrl;
  return PERSONA_AVATARS.get(expert.name.trim().toLowerCase()) ?? null;
}

export function getHiredExpertsLookup<
  T extends Pick<Expert, "id" | "source_template_id">,
>(experts: T[] | undefined, query: { isError: boolean; isFetching: boolean }) {
  const byTemplateId = new Map<string, T>();
  for (const expert of experts ?? []) {
    if (expert.source_template_id) {
      byTemplateId.set(expert.source_template_id, expert);
    }
  }
  const state =
    experts !== undefined
      ? ("loaded" as const)
      : query.isError && !query.isFetching
        ? ("error" as const)
        : ("loading" as const);
  return { byTemplateId, state };
}

export function getExpertFirstName(name: string): string {
  return name.trim().split(/\s+/)[0] || name;
}

/** The workflow promised in the day-one highlight. API ordering is not
 *  contractual and dangling refs have null names, so pick the first workflow
 *  with displayable copy instead of trusting index 0. */
export function getDayOneWorkflow(
  workflows: ExpertWorkflowRef[],
): ExpertWorkflowRef | null {
  return workflows.find((workflow) => workflow.name?.trim()) ?? null;
}
