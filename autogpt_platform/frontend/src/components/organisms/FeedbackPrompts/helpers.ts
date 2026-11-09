import {
  AiMagicIcon,
  Alert02Icon,
  Clock01Icon,
  Coins01Icon,
  CreditCardIcon,
  CreditCardNotAcceptIcon,
  HelpCircleIcon,
  KeyboardIcon,
  MicOff01Icon,
  PencilEdit02Icon,
  PackageIcon,
  PuzzleIcon,
  Search01Icon,
  SquareLock02Icon,
  Target02Icon,
  TestTube01Icon,
  UserQuestion01Icon,
  HourglassIcon,
  MessageQuestionIcon,
} from "@hugeicons/core-free-icons";
import type { Reason } from "./components/ReasonTiles";

export const OTHER_VALUE = "other";
export const MAX_FEEDBACK_NOTE_CHARS = 2000;

const OTHER: Reason = {
  value: OTHER_VALUE,
  label: "Something else",
  icon: PencilEdit02Icon,
  tone: "neutral",
};

export const TRIAL_CANCEL_REASONS: Reason[] = [
  {
    value: "too_expensive",
    label: "It costs too much",
    icon: Coins01Icon,
    tone: "warning",
  },
  {
    value: "no_results_yet",
    label: "I haven't seen results yet",
    icon: HourglassIcon,
    tone: "info",
  },
  {
    value: "missing_something",
    label: "It's missing something",
    icon: PuzzleIcon,
    tone: "accent",
  },
  {
    value: "only_trying",
    label: "I was only trying it out",
    icon: TestTube01Icon,
    tone: "success",
  },
  OTHER,
];

export const TRIAL_CHECKOUT_REASONS: Reason[] = [
  {
    value: "unclear_value",
    label: "Not sure what I get",
    icon: HelpCircleIcon,
    tone: "info",
  },
  {
    value: "no_card_yet",
    label: "Didn't want to add a card yet",
    icon: CreditCardIcon,
    tone: "warning",
  },
  {
    value: "looking_around",
    label: "Want to look around first",
    icon: Search01Icon,
    tone: "success",
  },
  {
    value: "card_failed",
    label: "My card didn't go through",
    icon: CreditCardNotAcceptIcon,
    tone: "error",
  },
  OTHER,
];

export const UPGRADE_CHECKOUT_REASONS: Reason[] = [
  { value: "price", label: "The price", icon: Coins01Icon, tone: "warning" },
  {
    value: "unclear_value",
    label: "Not sure what I get",
    icon: HelpCircleIcon,
    tone: "info",
  },
  {
    value: "card_failed",
    label: "My card didn't go through",
    icon: CreditCardNotAcceptIcon,
    tone: "error",
  },
  OTHER,
];

export const WORK_REASONS: Reason[] = [
  {
    value: "wrong_facts",
    label: "Got facts wrong",
    icon: Alert02Icon,
    tone: "error",
  },
  {
    value: "missed_ask",
    label: "Missed what I asked",
    icon: Target02Icon,
    tone: "warning",
  },
  { value: "too_slow", label: "Too slow", icon: Clock01Icon, tone: "info" },
  {
    value: "too_much_help",
    label: "Needed too much help",
    icon: UserQuestion01Icon,
    tone: "accent",
  },
  OTHER,
];

export const MISTAKE_REASONS: Reason[] = [
  {
    value: "wrong_facts",
    label: "Got facts wrong",
    icon: Alert02Icon,
    tone: "error",
  },
  {
    value: "ignored_instructions",
    label: "Ignored my instructions",
    icon: Target02Icon,
    tone: "warning",
  },
  {
    value: "made_up",
    label: "Made things up",
    icon: AiMagicIcon,
    tone: "accent",
  },
  { value: "too_slow", label: "Too slow", icon: Clock01Icon, tone: "info" },
  OTHER,
];

export const TOP_UP_REASONS: Reason[] = [
  {
    value: "lower_price",
    label: "A lower price",
    icon: Coins01Icon,
    tone: "warning",
  },
  {
    value: "smaller_amounts",
    label: "Smaller amounts",
    icon: PackageIcon,
    tone: "info",
  },
  {
    value: "credits_unclear",
    label: "Knowing what credits buy",
    icon: MessageQuestionIcon,
    tone: "accent",
  },
  {
    value: "just_trying",
    label: "Just trying things out",
    icon: TestTube01Icon,
    tone: "success",
  },
  OTHER,
];

export const BRAIN_DUMP_SKIP_REASONS: Reason[] = [
  { value: "no_mic", label: "No microphone", icon: MicOff01Icon, tone: "info" },
  {
    value: "privacy",
    label: "Privacy",
    icon: SquareLock02Icon,
    tone: "accent",
  },
  {
    value: "unsure_what_to_say",
    label: "Not sure what to say",
    icon: HelpCircleIcon,
    tone: "warning",
  },
  {
    value: "prefer_typing",
    label: "I'd rather type",
    icon: KeyboardIcon,
    tone: "success",
  },
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
