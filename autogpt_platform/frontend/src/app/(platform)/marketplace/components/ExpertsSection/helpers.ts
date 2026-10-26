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
  /** A flat cover band for the top of a card — the wash without the fade. */
  band: string;
  /** Glossy tag for the expert page's skills: a white-to-tint gradient with
   *  a top highlight and a soft glow in the accent hue. */
  chip: string;
  icon: string;
  roleIcon: IconSvgElement;
}

const ACCENTS: Record<string, ExpertAccent> = {
  violet: {
    // TODO(tokens): no exact role: bg-[…] #8B5CF6
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(139,92,246,0.10),transparent_70%)]",
    washWide:
      // TODO(tokens): no exact role: bg-[…] #8B5CF6
      "bg-[linear-gradient(180deg,rgba(139,92,246,0.10),rgba(139,92,246,0.03)_60%,transparent)]",
    // TODO(tokens): no exact role: ring-violet-600/10 #7C3AED
    pill: "bg-accent-surface text-accent-strong ring-1 ring-inset ring-violet-600/10",
    band: "bg-accent-surface/60",
    // TODO(tokens): no exact role: text-violet-800 #5B21B6, ring-violet-500/20 #8B5CF6, shadow-[…] #8B5CF6; no fitting role for shadow-[…] #FFFFFF; #FFFFFF is card (a surface fill), popover (a surface fill), accent-foreground (a text colour)
    chip: "bg-gradient-to-b from-surface to-accent-surface text-violet-800 ring-1 ring-inset ring-violet-500/20 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_1px_2px_rgba(139,92,246,0.12),0_0_18px_-4px_rgba(139,92,246,0.45)]",
    // TODO(tokens): no exact role: text-violet-500 #8B5CF6
    icon: "text-violet-500",
    roleIcon: Megaphone01Icon,
  },
  amber: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgb(var(--color-warning)/0.10),transparent_70%)]",
    washWide:
      "bg-[linear-gradient(180deg,rgb(var(--color-warning)/0.10),rgb(var(--color-warning)/0.03)_60%,transparent)]",
    // TODO(tokens): no exact role: ring-amber-600/10 #D97706
    pill: "bg-warning-surface text-warning-foreground ring-1 ring-inset ring-amber-600/10",
    band: "bg-warning-surface/60",
    // TODO(tokens): no exact role: text-amber-800 #92400E; no fitting role for shadow-[…] #FFFFFF; #FFFFFF is card (a surface fill), popover (a surface fill), accent-foreground (a text colour); no fitting role for shadow-[…] #F59E0B; #F59E0B is warning (a status colour)
    chip: "bg-gradient-to-b from-surface to-warning-surface text-amber-800 ring-1 ring-inset ring-warning/20 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_1px_2px_rgba(245,158,11,0.12),0_0_18px_-4px_rgba(245,158,11,0.45)]",
    icon: "text-warning",
    roleIcon: ChartIncreaseIcon,
  },
  sky: {
    // TODO(tokens): no exact role: bg-[…] #0EA5E9
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(14,165,233,0.10),transparent_70%)]",
    washWide:
      // TODO(tokens): no exact role: bg-[…] #0EA5E9
      "bg-[linear-gradient(180deg,rgba(14,165,233,0.10),rgba(14,165,233,0.03)_60%,transparent)]",
    // TODO(tokens): no exact role: bg-sky-50 #F0F9FF, text-sky-700 #0369A1, ring-sky-600/10 #0284C7
    pill: "bg-sky-50 text-sky-700 ring-1 ring-inset ring-sky-600/10",
    // TODO(tokens): no exact role: bg-sky-50/60 #F0F9FF
    band: "bg-sky-50/60",
    // TODO(tokens): no exact role: to-sky-50 #F0F9FF, text-sky-800 #075985, ring-sky-500/20 #0EA5E9, shadow-[…] #0EA5E9; no fitting role for shadow-[…] #FFFFFF; #FFFFFF is card (a surface fill), popover (a surface fill), accent-foreground (a text colour)
    chip: "bg-gradient-to-b from-surface to-sky-50 text-sky-800 ring-1 ring-inset ring-sky-500/20 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_1px_2px_rgba(14,165,233,0.12),0_0_18px_-4px_rgba(14,165,233,0.45)]",
    // TODO(tokens): no exact role: text-sky-500 #0EA5E9
    icon: "text-sky-500",
    roleIcon: Settings01Icon,
  },
  zinc: {
    // TODO(tokens): no fitting role for bg-[…] #71717A; #71717A is muted-foreground (a text colour)
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(113,113,122,0.08),transparent_70%)]",
    washWide:
      // TODO(tokens): no fitting role for bg-[…] #71717A; #71717A is muted-foreground (a text colour)
      "bg-[linear-gradient(180deg,rgba(113,113,122,0.08),rgba(113,113,122,0.02)_60%,transparent)]",
    // TODO(tokens): no fitting role for ring-zinc-500/10 #83838C; #83838C is foreground-tertiary (a text colour), placeholder (a text colour)
    pill: "bg-surface-hover text-foreground-secondary ring-1 ring-inset ring-zinc-500/10",
    band: "bg-surface-hover/60",
    // TODO(tokens): no fitting role for ring-zinc-500/20 #83838C; #83838C is foreground-tertiary (a text colour), placeholder (a text colour); no fitting role for shadow-[…] #FFFFFF; #FFFFFF is card (a surface fill), popover (a surface fill), accent-foreground (a text colour); no fitting role for shadow-[…] #71717A; #71717A is muted-foreground (a text colour)
    chip: "bg-gradient-to-b from-surface to-surface-muted text-foreground-body ring-1 ring-inset ring-zinc-500/20 shadow-[inset_0_1px_0_rgba(255,255,255,0.95),0_1px_2px_rgba(113,113,122,0.12),0_0_18px_-4px_rgba(113,113,122,0.45)]",
    icon: "text-foreground-tertiary",
    roleIcon: Briefcase01Icon,
  },
};

const ROLE_ACCENTS: Array<[RegExp, string]> = [
  [
    /marketing|growth|brand|seo|content|social|market|competitor|email|lifecycle/i,
    "violet",
  ],
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
