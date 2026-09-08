import {
  File01Icon,
  PlugIcon,
  RepeatIcon,
  Robot01Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { HomeBriefingOutcome } from "@/app/api/__generated__/models/homeBriefingOutcome";
import type { HomeRecentWorkGroup } from "@/app/api/__generated__/models/homeRecentWorkGroup";
import type { HomeRecentWorkItemCategory } from "@/app/api/__generated__/models/homeRecentWorkItemCategory";
import type { HomeWorkActorKind } from "@/app/api/__generated__/models/homeWorkActorKind";

export function getWorkItemIcon(
  category: HomeRecentWorkItemCategory,
): IconSvgElement {
  if (category === "integration") return PlugIcon;
  if (category === "schedule") return RepeatIcon;
  return File01Icon;
}

export function getActorIcon(kind: HomeWorkActorKind): IconSvgElement {
  if (kind === "autopilot") return SparklesIcon;
  return Robot01Icon;
}

// The team did the work on someone's behalf; a workflow ran on its own.
export function splitGroupsBySection(groups: HomeRecentWorkGroup[]) {
  return {
    team: groups.filter((group) => group.actor.kind !== "workflow"),
    workflows: groups.filter((group) => group.actor.kind === "workflow"),
  };
}

export function getRunTriggerLabel(
  trigger: HomeBriefingOutcome["trigger"],
): string {
  if (trigger === "schedule") return "Scheduled run";
  if (trigger === "webhook") return "Triggered run";
  return "Manual run";
}

type ActorChip = { label: string; className: string };

// The chip is the fastest way to tell the two halves of the card apart, so
// each kind gets its own colour and a matching glow.
const ACTOR_CHIPS: Record<HomeWorkActorKind, ActorChip> = {
  expert: {
    label: "Expert",
    className:
      "border-blue-200 bg-blue-50 text-blue-700 shadow-[0_0_8px_-1px_rgba(96,165,250,0.7)]",
  },
  workflow: {
    label: "Workflow",
    className:
      "border-yellow-200 bg-yellow-50 text-yellow-700 shadow-[0_0_8px_-1px_rgba(247,205,51,0.9)]",
  },
  autopilot: {
    label: "Autopilot",
    className: "border-zinc-200 bg-white text-zinc-500",
  },
};

export function getActorChip(kind: HomeWorkActorKind): ActorChip {
  return ACTOR_CHIPS[kind] ?? ACTOR_CHIPS.autopilot;
}

// The feed spans a week, so the weekday is load-bearing: "Mon 10:45" vs
// three files all labelled "10:45".
export function formatWorkTime(
  value: Date | null | undefined,
  timeZone: string,
  locale?: string,
): string {
  if (!value) return "Recently";
  return new Intl.DateTimeFormat(locale, {
    timeZone,
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

export function formatGroupCounts(group: HomeRecentWorkGroup): string {
  return [
    countLabel(group.run_count ?? 0, "run"),
    countLabel(group.file_count ?? 0, "file"),
    countLabel(group.integration_count ?? 0, "action"),
    countLabel(group.schedule_count ?? 0, "schedule"),
  ]
    .filter(Boolean)
    .join(" · ");
}

function countLabel(count: number, noun: string): string | null {
  if (count === 0) return null;
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}
