import {
  Calendar03Icon,
  ChartLineData01Icon,
  File01Icon,
  Mail01Icon,
  PencilEdit02Icon,
  Search01Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react-native";

import { palette } from "@/lib/expertPalette";

export interface PromptTopic {
  icon: IconSvgElement;
  tint: string;
}

// First match wins, so the narrower topics come before the broad ones.
const TOPICS: { pattern: RegExp; topic: PromptTopic }[] = [
  {
    pattern: /\b(e-?mails?|inbox|repl(y|ies)|follow[- ]?ups?|outreach|messages?)\b/i,
    topic: { icon: Mail01Icon, tint: palette.support.hex },
  },
  {
    pattern: /\b(reports?|reporting|data|analy\w*|metrics?|dashboards?|numbers|budgets?|revenue|kpis?)\b/i,
    topic: { icon: ChartLineData01Icon, tint: palette.finance.hex },
  },
  {
    pattern: /\b(plan\w*|schedules?|week|calendar|meetings?|agenda|trips?|itinerar\w*)\b/i,
    topic: { icon: Calendar03Icon, tint: palette.operations.hex },
  },
  {
    pattern: /\b(research\w*|find|look up|compare|competitors?|markets?|topics?)\b/i,
    topic: { icon: Search01Icon, tint: palette.research.hex },
  },
  {
    pattern: /\b(sum(marize|marise| up)|documents?|docs?|notes|pdfs?|read)\b/i,
    topic: { icon: File01Icon, tint: palette.content.hex },
  },
  {
    pattern: /\b(write|draft|posts?|blogs?|copy|captions?|content)\b/i,
    topic: { icon: PencilEdit02Icon, tint: palette.marketing.hex },
  },
];

const DEFAULT_TOPIC: PromptTopic = { icon: SparklesIcon, tint: palette.otto.hex };

/** An icon and palette colour that say what a suggested prompt is about. */
export function promptTopic(prompt: string): PromptTopic {
  return TOPICS.find(({ pattern }) => pattern.test(prompt))?.topic ?? DEFAULT_TOPIC;
}
