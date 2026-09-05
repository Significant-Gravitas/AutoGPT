import {
  File01Icon,
  FlowIcon,
  PlugIcon,
  RepeatIcon,
  Robot01Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
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
