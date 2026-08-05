import {
  Briefcase01Icon,
  ChartIncreaseIcon,
  Megaphone01Icon,
  Settings01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

export interface ExpertAccent {
  wash: string;
  pill: string;
  icon: string;
  roleIcon: IconSvgElement;
}

const ACCENTS: Record<string, ExpertAccent> = {
  violet: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(139,92,246,0.10),transparent_70%)]",
    pill: "bg-violet-50 text-violet-700 ring-1 ring-inset ring-violet-600/10",
    icon: "text-violet-500",
    roleIcon: Megaphone01Icon,
  },
  amber: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(245,158,11,0.10),transparent_70%)]",
    pill: "bg-amber-50 text-amber-700 ring-1 ring-inset ring-amber-600/10",
    icon: "text-amber-500",
    roleIcon: ChartIncreaseIcon,
  },
  sky: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(14,165,233,0.10),transparent_70%)]",
    pill: "bg-sky-50 text-sky-700 ring-1 ring-inset ring-sky-600/10",
    icon: "text-sky-500",
    roleIcon: Settings01Icon,
  },
  zinc: {
    wash: "bg-[radial-gradient(120%_100%_at_50%_0%,rgba(113,113,122,0.08),transparent_70%)]",
    pill: "bg-zinc-100 text-zinc-600 ring-1 ring-inset ring-zinc-500/10",
    icon: "text-zinc-500",
    roleIcon: Briefcase01Icon,
  },
};

const ROLE_ACCENTS: Array<[RegExp, string]> = [
  [/marketing|growth|brand/i, "violet"],
  [/sales|revenue/i, "amber"],
  [/ops|operations|support/i, "sky"],
];

export function getExpertAccent(role: string): ExpertAccent {
  for (const [pattern, key] of ROLE_ACCENTS) {
    if (pattern.test(role)) return ACCENTS[key];
  }
  return ACCENTS.zinc;
}
