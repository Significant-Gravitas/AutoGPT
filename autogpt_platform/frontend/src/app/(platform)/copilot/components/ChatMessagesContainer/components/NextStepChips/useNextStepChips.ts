"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useState } from "react";
import { useCopilotChatActions } from "../../../CopilotChatActionsProvider/useCopilotChatActions";
import type { MessagePart } from "../../helpers";
import { getNextStepSuggestions } from "./helpers";

export function useNextStepChips(parts: MessagePart[]) {
  const isEnabled = useGetFlag(Flag.COPILOT_NEXT_STEP_CHIPS);
  const { onSend } = useCopilotChatActions();
  const [sentLabel, setSentLabel] = useState<string | null>(null);

  const suggestions = isEnabled ? getNextStepSuggestions(parts) : [];

  async function handleSelect(label: string) {
    if (sentLabel) return;
    setSentLabel(label);
    try {
      await onSend(label);
    } catch {
      setSentLabel(null);
    }
  }

  return {
    suggestions,
    sentLabel,
    handleSelect,
  };
}
