import type { DeletionFeedbackRequest } from "@/app/api/__generated__/models/deletionFeedbackRequest";
import type { DeleteIntegrationTarget } from "@/components/contextual/IntegrationsPanel/components/hooks/useDeleteIntegration";
import { uniqueNames } from "../../context-helpers";
import type {
  DisconnectState,
  ExitReason,
  PanelActionKind,
} from "../../helpers";

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

export const EXIT_REASON_LABELS = EXIT_REASON_OPTIONS.map(
  (option) => option.label,
);

export const MAX_FEEDBACK_TEXT_LENGTH = 1000;

export const SKIPPED_FEEDBACK: DeletionFeedbackRequest = {
  kind: "answer",
  reason: "skipped",
};

export const PANEL_ROUTES: Partial<Record<PanelActionKind, string>> = {
  manage_apps: "/settings/integrations",
  billing: "/settings/billing",
};

interface AnswerArgs {
  reason: ExitReason | null;
  text: string;
}

interface DisconnectOutcome {
  succeeded: DeleteIntegrationTarget[];
  failed: DeleteIntegrationTarget[];
  needsConfirmation: { target: DeleteIntegrationTarget }[];
}

export function getReasonLabel(reason: ExitReason | null): string {
  return (
    EXIT_REASON_OPTIONS.find((option) => option.key === reason)?.label ?? ""
  );
}

export function getReasonFromLabel(label: string): ExitReason | null {
  return (
    EXIT_REASON_OPTIONS.find((option) => option.label === label)?.key ?? null
  );
}

export function buildAnswerFeedback({
  reason,
  text,
}: AnswerArgs): DeletionFeedbackRequest {
  const trimmed = text.trim();
  if (!reason && !trimmed) return SKIPPED_FEEDBACK;
  return {
    kind: "answer",
    reason: reason ?? undefined,
    text: trimmed || undefined,
  };
}

export function getTargetNames(targets: DeleteIntegrationTarget[]): string[] {
  return uniqueNames(targets.map((target) => target.name ?? target.provider));
}

export function getDisconnectResult(
  outcome: DisconnectOutcome,
): DisconnectState {
  const failed = [
    ...outcome.failed,
    ...outcome.needsConfirmation.map((item) => item.target),
  ];
  if (failed.length > 0) {
    return { status: "failed", names: getTargetNames(failed) };
  }
  return { status: "done", names: getTargetNames(outcome.succeeded) };
}
