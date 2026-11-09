import type { OptionsBlock } from "@/app/api/__generated__/models/optionsBlock";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { capturePostHogEvent } from "@/services/analytics/posthog-capture";
import { VisualAnswerEvent } from "@/services/analytics/posthog-events";
import { AlertCircleIcon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { useCopilotChatActions } from "../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { TEXT_TONE } from "../helpers";

interface Props {
  block: OptionsBlock;
  canAct: boolean;
}

export function OptionsBlockView({ block, canAct }: Props) {
  const { onSend } = useCopilotChatActions();
  const [isSent, setIsSent] = useState(false);
  const isDisabled = !canAct || isSent;

  function handleSelect(title: string) {
    if (isDisabled) return;
    setIsSent(true);
    capturePostHogEvent(VisualAnswerEvent.VISUAL_OPTION_CLICKED, {});
    void onSend(title);
  }

  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
      {block.items.map((item, index) => {
        const tone = item.tone ?? "neutral";
        return (
          <button
            key={index}
            type="button"
            disabled={isDisabled}
            onClick={() => handleSelect(item.title)}
            className="flex min-w-0 items-start gap-2 rounded-2xl border border-border-base bg-surface p-3 text-left transition-colors hover:border-brand disabled:pointer-events-none disabled:opacity-60"
          >
            <Icon
              icon={
                tone === "negative" || tone === "warning"
                  ? AlertCircleIcon
                  : ArrowRight01Icon
              }
              size={16}
              tone={TEXT_TONE[tone]}
              className="mt-0.5 shrink-0"
            />
            <span className="min-w-0">
              <Text
                variant="body-medium"
                as="span"
                className="block break-words"
              >
                {item.title}
              </Text>
              {item.description ? (
                <Text
                  variant="small"
                  tone="secondary"
                  as="span"
                  className="block break-words"
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
