import {
  File01Icon,
  FlowIcon,
  PlugIcon,
  RepeatIcon,
  Robot01Icon,
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
  if (kind === "workflow") return FlowIcon;
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

export function getActorKindLabel(kind: HomeWorkActorKind): string {
  if (kind === "workflow") return "Workflow";
  if (kind === "autopilot") return "Autopilot";
  return "Expert";
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
