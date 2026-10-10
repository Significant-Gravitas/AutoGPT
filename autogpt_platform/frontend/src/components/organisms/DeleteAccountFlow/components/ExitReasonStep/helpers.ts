import type { ExitReason } from "../../helpers";

interface ExitReasonOption {
  key: ExitReason;
  label: string;
}

export const EXIT_REASON_OPTIONS: readonly ExitReasonOption[] = [
  { key: "privacy", label: "I'm worried about my data or privacy" },
  { key: "cost", label: "It costs too much" },
  { key: "not_using", label: "I'm not using it enough" },
  { key: "missing_capability", label: "It couldn't do what I needed" },
  { key: "limits", label: "I kept hitting usage limits" },
  { key: "other", label: "Something else" },
];

const DEFAULT_PLACEHOLDER = "Anything else we should know? (optional)";

export function getTextPlaceholder(reason: ExitReason | null): string {
  if (reason === "missing_capability") {
    return "What were you trying to get done?";
  }
  if (reason === "other") return "Tell us more (optional)";
  return DEFAULT_PLACEHOLDER;
}
