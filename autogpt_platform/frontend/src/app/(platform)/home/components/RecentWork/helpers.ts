import { File01Icon, PlugIcon, RepeatIcon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { HomeRecentWorkItemCategory } from "@/app/api/__generated__/models/homeRecentWorkItemCategory";

export function getWorkItemIcon(
  category: HomeRecentWorkItemCategory,
): IconSvgElement {
  if (category === "integration") return PlugIcon;
  if (category === "schedule") return RepeatIcon;
  return File01Icon;
}

// The feed spans up to a week, so unlike the briefing's time-only stamps the
// weekday is load-bearing: "Mon 10:45" vs three files all labelled "10:45".
export function formatWorkTime(
  value: Date,
  timeZone: string,
  locale?: string,
): string {
  return new Intl.DateTimeFormat(locale, {
    timeZone,
    weekday: "short",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}
