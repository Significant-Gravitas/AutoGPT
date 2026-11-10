import {
  AiMagicIcon,
  Alert02Icon,
  Clock01Icon,
  Coins01Icon,
  CreditCardNotAcceptIcon,
  HelpCircleIcon,
  HourglassIcon,
  MessageQuestionIcon,
  PackageIcon,
  PencilEdit02Icon,
  PuzzleIcon,
  Search01Icon,
  Target02Icon,
  TestTube01Icon,
  ThumbsDownIcon,
  UserQuestion01Icon,
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

export const CARD_FAILED_VALUE = "card_failed";

export const UPGRADE_CHECKOUT_REASONS: Reason[] = [
  { value: "price", label: "The price", icon: Coins01Icon, tone: "warning" },
  {
    value: "unclear_value",
    label: "Not sure what I get",
    icon: HelpCircleIcon,
    tone: "info",
  },
  {
    value: "just_looking",
    label: "Just looking for now",
    icon: Search01Icon,
    tone: "success",
  },
  {
    value: CARD_FAILED_VALUE,
    label: "My card didn't go through",
    icon: CreditCardNotAcceptIcon,
    tone: "error",
  },
  OTHER,
];

// A work check and a mistake flag offer the same list, so their answers can
// be read together.
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
  {
    value: "made_up",
    label: "Made things up",
    icon: AiMagicIcon,
    tone: "accent",
  },
  { value: "too_slow", label: "Too slow", icon: Clock01Icon, tone: "info" },
  {
    value: "too_much_help",
    label: "I had to step in too much",
    icon: UserQuestion01Icon,
    tone: "accent",
  },
  {
    value: "not_usable",
    label: "Not good enough to use",
    icon: ThumbsDownIcon,
    tone: "warning",
  },
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
