import {
  AiMagicIcon,
  Alert02Icon,
  Clock01Icon,
  Coins01Icon,
  CreditCardIcon,
  HelpCircleIcon,
  KeyboardIcon,
  MicOff01Icon,
  PencilEdit02Icon,
  PackageIcon,
  PuzzleIcon,
  SquareLock02Icon,
  Target02Icon,
  TestTube01Icon,
  UserQuestion01Icon,
  HourglassIcon,
  MessageQuestionIcon,
} from "@hugeicons/core-free-icons";
import type { Reason } from "./components/ReasonTiles";

export const OTHER_LABEL = "Something else";

const OTHER: Reason = {
  label: OTHER_LABEL,
  icon: PencilEdit02Icon,
  tone: "neutral",
};

export const TRIAL_CANCEL_REASONS: Reason[] = [
  { label: "It costs too much", icon: Coins01Icon, tone: "warning" },
  { label: "I haven't seen results yet", icon: HourglassIcon, tone: "info" },
  { label: "It's missing something", icon: PuzzleIcon, tone: "accent" },
  { label: "I was only trying it out", icon: TestTube01Icon, tone: "success" },
  OTHER,
];

export const CHECKOUT_REASONS: Reason[] = [
  { label: "The price", icon: Coins01Icon, tone: "warning" },
  { label: "Not sure what I get", icon: HelpCircleIcon, tone: "info" },
  { label: "I want to try it first", icon: TestTube01Icon, tone: "success" },
  { label: "My payment failed", icon: CreditCardIcon, tone: "error" },
  OTHER,
];

export const WORK_REASONS: Reason[] = [
  { label: "Got facts wrong", icon: Alert02Icon, tone: "error" },
  { label: "Missed what I asked", icon: Target02Icon, tone: "warning" },
  { label: "Too slow", icon: Clock01Icon, tone: "info" },
  { label: "Needed too much help", icon: UserQuestion01Icon, tone: "accent" },
  OTHER,
];

export const MISTAKE_REASONS: Reason[] = [
  { label: "Got facts wrong", icon: Alert02Icon, tone: "error" },
  { label: "Ignored my instructions", icon: Target02Icon, tone: "warning" },
  { label: "Made things up", icon: AiMagicIcon, tone: "accent" },
  { label: "Too slow", icon: Clock01Icon, tone: "info" },
  OTHER,
];

export const TOP_UP_REASONS: Reason[] = [
  { label: "A lower price", icon: Coins01Icon, tone: "warning" },
  { label: "Smaller amounts", icon: PackageIcon, tone: "info" },
  {
    label: "Knowing what credits buy",
    icon: MessageQuestionIcon,
    tone: "accent",
  },
  { label: "Just trying things out", icon: TestTube01Icon, tone: "success" },
  OTHER,
];

export const BRAIN_DUMP_SKIP_REASONS: Reason[] = [
  { label: "No microphone", icon: MicOff01Icon, tone: "info" },
  { label: "Privacy", icon: SquareLock02Icon, tone: "accent" },
  { label: "Not sure what to say", icon: HelpCircleIcon, tone: "warning" },
  { label: "I'd rather type", icon: KeyboardIcon, tone: "success" },
];

export const HIRE_SKIP_REASONS: Reason[] = [
  { label: "None fit my work", icon: PuzzleIcon, tone: "accent" },
  { label: "Not clear what they do", icon: HelpCircleIcon, tone: "info" },
  { label: "I'll pick later", icon: Clock01Icon, tone: "success" },
];

// What GET /api/onboarding/brain-dump/suggested-tasks returns for two different
// onboarding answers, for the stories.
export const SAMPLE_STARTERS = {
  developer: [
    "Triage this week's open GitHub issues",
    "Write release notes from the merged PRs",
    "Find the flaky tests in our last CI runs",
  ],
  sales: [
    "Find 10 leads that match my best customer",
    "Draft follow-ups for last week's demos",
    "Update the CRM from my call notes",
  ],
};
