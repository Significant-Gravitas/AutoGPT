import type { OptionsBlock } from "@/app/api/__generated__/models/optionsBlock";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { isPaywallError } from "@/components/contextual/PaywallToast/helpers";
import { cn } from "@/lib/utils";
import { capturePostHogEvent } from "@/services/analytics/posthog-capture";
import { VisualAnswerEvent } from "@/services/analytics/posthog-events";
import { AlertCircleIcon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { useCopilotChatActions } from "../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { CopilotAuthError } from "../../../helpers";
import { QueueFollowUpFullError } from "../../../helpers/queueFollowUpMessage";
import { toastSendFailure } from "../../../helpers/sendFailureToast";
import { SendInFlightError } from "../../../stream/turnRuntime";
import { type BlockTone, TEXT_TONE } from "../helpers";

const ICON_BADGE: Record<BlockTone, string> = {
  neutral: "bg-accent-surface",
  positive: "bg-success-surface",
  negative: "bg-error-surface",
  warning: "bg-warning-surface",
};

interface Props {
  block: OptionsBlock;
  canAct: boolean;
}

export function OptionsBlockView({ block, canAct }: Props) {
  const { onSend } = useCopilotChatActions();
  const [hasSelected, setHasSelected] = useState(false);
  const isDisabled = !canAct || hasSelected;

  async function handleSelect(title: string) {
    if (isDisabled) return;
    setHasSelected(true);
    capturePostHogEvent(VisualAnswerEvent.VISUAL_OPTION_CLICKED, {});
    try {
      await onSend(title);
    } catch (error) {
      const canRetry =
        error instanceof CopilotAuthError ||
        error instanceof SendInFlightError ||
        error instanceof QueueFollowUpFullError ||
        isPaywallError(error);
      // A request that failed in transit may already have started a turn.
      if (canRetry) setHasSelected(false);
      toastSendFailure({
        title: canRetry
          ? "Couldn't send the message"
          : "Couldn't confirm the message was sent",
        error,
        description: canRetry
          ? undefined
          : "Check the chat to see whether the message arrived.",
        recovery: "you can select the option again",
        paywallRecovery: "you can select the same option once you have a plan",
      });
    }
  }

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      {block.items.map((item, index) => {
        const tone = item.tone ?? "neutral";
        const isWarning = tone === "negative" || tone === "warning";
        return (
          <button
            key={index}
            type="button"
            disabled={isDisabled}
            onClick={() => void handleSelect(item.title)}
            className="group flex min-w-0 items-start gap-3 rounded-2xl bg-gradient-to-br from-surface to-accent-surface/40 p-4 text-left smooth-shadow-ring-sm transition-transform duration-150 hover:-translate-y-0.5 focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none disabled:pointer-events-none disabled:opacity-60 motion-reduce:transition-none motion-reduce:hover:translate-y-0"
          >
            <span
              aria-hidden
              className={cn(
                "flex size-7 shrink-0 items-center justify-center rounded-full",
                ICON_BADGE[tone],
              )}
            >
              <Icon
                icon={isWarning ? AlertCircleIcon : ArrowRight01Icon}
                size={14}
                tone={tone === "neutral" ? "accent" : TEXT_TONE[tone]}
              />
            </span>
            <span className="flex min-w-0 flex-col gap-0.5">
              <Text variant="body-medium" as="span" className="break-words">
                {item.title}
              </Text>
              {item.description ? (
                <Text
                  variant="body-tight"
                  tone="secondary"
                  as="span"
                  className="break-words"
                >
                  {item.description}
                </Text>
              ) : null}
            </span>
          </button>
        );
      })}
    </div>
  );
}
