import {
  File01Icon,
  FlowIcon,
  PlugIcon,
  RepeatIcon,
  Robot01Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
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
  if (kind === "autopilot") return SparklesIcon;
  return Robot01Icon;
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
