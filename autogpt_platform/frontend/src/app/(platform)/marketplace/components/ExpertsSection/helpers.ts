import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import {
  Briefcase01Icon,
  ChartIncreaseIcon,
  Coins01Icon,
  HeadsetIcon,
  Megaphone01Icon,
  QuillWrite01Icon,
  Search01Icon,
  Settings01Icon,
  SourceCodeIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { findColorOption } from "@/app/(platform)/raise/components/ColorStep/helpers";
import { cn } from "@/lib/utils";

export interface ExpertAccent {
  wash: string;
  /** Full-bleed variant for wide surfaces (e.g. the expert page header) —
   *  the radial `wash` is sized for the ~640px sheet and fades out halfway
   *  across a full-width card. */
  washWide: string;
  pill: string;
  /** Glossy tag for the expert page's skills: a white-to-tint gradient with
   *  a top highlight and a soft glow in the accent hue. */
  chip: string;
  icon: string;
  roleIcon: IconSvgElement;
}

const ACCENTS: Record<string, ExpertAccent> = {
  violet: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(139,92,246,0.10),transparent_70%)]",
    washWide:
      "bg-[linear-gradient(180deg,rgba(139,92,246,0.10),rgba(139,92,246,0.03)_60%,transparent)]",
    pill: "bg-violet-50 text-violet-700 ring-1 ring-inset ring-violet-600/10",
    chip: "bg-gradient-to-b from-white to-violet-50 text-violet-800 ring-1 ring-inset ring-violet-500/20 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_1px_2px_rgba(139,92,246,0.12),0_0_18px_-4px_rgba(139,92,246,0.45)]",
    icon: "text-violet-500",
    roleIcon: Megaphone01Icon,
  },
  amber: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(245,158,11,0.10),transparent_70%)]",
    washWide:
      "bg-[linear-gradient(180deg,rgba(245,158,11,0.10),rgba(245,158,11,0.03)_60%,transparent)]",
    pill: "bg-amber-50 text-amber-700 ring-1 ring-inset ring-amber-600/10",
    chip: "bg-gradient-to-b from-white to-amber-50 text-amber-800 ring-1 ring-inset ring-amber-500/20 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_1px_2px_rgba(245,158,11,0.12),0_0_18px_-4px_rgba(245,158,11,0.45)]",
    icon: "text-amber-500",
    roleIcon: ChartIncreaseIcon,
  },
  sky: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(14,165,233,0.10),transparent_70%)]",
    washWide:
      "bg-[linear-gradient(180deg,rgba(14,165,233,0.10),rgba(14,165,233,0.03)_60%,transparent)]",
    pill: "bg-sky-50 text-sky-700 ring-1 ring-inset ring-sky-600/10",
    chip: "bg-gradient-to-b from-white to-sky-50 text-sky-800 ring-1 ring-inset ring-sky-500/20 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_1px_2px_rgba(14,165,233,0.12),0_0_18px_-4px_rgba(14,165,233,0.45)]",
    icon: "text-sky-500",
    roleIcon: Settings01Icon,
  },
  zinc: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(113,113,122,0.08),transparent_70%)]",
    washWide:
      "bg-[linear-gradient(180deg,rgba(113,113,122,0.08),rgba(113,113,122,0.02)_60%,transparent)]",
    pill: "bg-zinc-100 text-zinc-600 ring-1 ring-inset ring-zinc-500/10",
    chip: "bg-gradient-to-b from-white to-zinc-50 text-zinc-700 ring-1 ring-inset ring-zinc-500/20 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_1px_2px_rgba(113,113,122,0.12),0_0_18px_-4px_rgba(113,113,122,0.45)]",
    icon: "text-zinc-500",
    roleIcon: Briefcase01Icon,
  },
};

const ROLE_ACCENTS: Array<[RegExp, string]> = [
  [/marketing|growth|brand/i, "violet"],
  [/sales|revenue/i, "amber"],
  [/ops|operations|support/i, "sky"],
];

/** A skill's category is its identity mark, the way an expert's role is:
 *  the eight canonical categories fold onto the same four accents, so
 *  "Outreach playbook · Sales" carries the amber a Sales expert already has.
 *  An unknown or missing category keeps the neutral wash and shows no pill. */
export function getCategoryAccent(category: string | undefined): {
  accent: ExpertAccent;
  icon: IconSvgElement | null;
} {
  const entry = category ? CATEGORY_ACCENTS[category.toLowerCase()] : undefined;
  if (!entry) return { accent: ACCENTS.zinc, icon: null };
  return { accent: ACCENTS[entry[0]], icon: entry[1] };
}

const CATEGORY_ACCENTS: Record<string, [string, IconSvgElement]> = {
  marketing: ["violet", Megaphone01Icon],
  content: ["violet", QuillWrite01Icon],
  sales: ["amber", ChartIncreaseIcon],
  finance: ["amber", Coins01Icon],
  operations: ["sky", Settings01Icon],
  support: ["sky", HeadsetIcon],
  research: ["zinc", Search01Icon],
  development: ["zinc", SourceCodeIcon],
};

export function getExpertAccent(role: string): ExpertAccent {
  for (const [pattern, key] of ROLE_ACCENTS) {
    if (pattern.test(role)) return ACCENTS[key];
  }
  return ACCENTS.zinc;
}

/** An expert raised through /raise carries the color its owner picked, which
 *  outranks the role guess. Marketplace templates have no color, so they keep
 *  falling back to the role accent. The role icon is never colour-derived. */
export function getRaisedExpertAccent(
  role: string,
  color: string | null | undefined,
): ExpertAccent {
  const roleAccent = getExpertAccent(role);
  const option = findColorOption(color ?? null);
  if (!option) return roleAccent;

  const wash = cn("bg-gradient-to-b to-transparent", option.washFromClassName);
  return {
    ...roleAccent,
    wash,
    washWide: wash,
    pill: cn("border", option.bubbleClassName, option.textClassName),
    icon: option.textClassName,
  };
}

export function getExpertFirstName(name: string): string {
  return name.trim().split(/\s+/)[0] || "Expert";
}

/** The workflow named as what the expert sets up on day one. The backend
 *  orders workflows deterministically, but a dangling ref has a null name,
 *  so take the first one that has copy to show rather than index 0. */
export function getDayOneWorkflow(
  workflows: ExpertWorkflowRef[],
): ExpertWorkflowRef | null {
  return workflows.find((workflow) => workflow.name?.trim()) ?? null;
}

/** The integrations an expert's workflows will ask this viewer to connect.
 *
 *  Reads `integration_providers`, never `chain`: the chain is a three-item
 *  display summary, so a workflow with a fourth integration silently loses
 *  one, picked by node counts rather than by whether anyone connects it.
 *  Providers the platform already pays for are then subtracted — the roster
 *  leans on Anthropic, OpenAI, Jina and a Webshare proxy, none of which a user
 *  ever connects. Without the system list this returns nothing, because naming
 *  a provider the platform supplies is worse than naming none. */
export function getExpertAccessProviders(
  workflows: ExpertWorkflowRef[],
  systemProviders: string[] | undefined,
): string[] {
  if (!systemProviders) return [];
  const supplied = new Set(systemProviders);
  const providers = workflows
    .flatMap((workflow) => workflow.integration_providers ?? [])
    .filter((provider) => !supplied.has(provider));
  return [...new Set(providers)];
}
