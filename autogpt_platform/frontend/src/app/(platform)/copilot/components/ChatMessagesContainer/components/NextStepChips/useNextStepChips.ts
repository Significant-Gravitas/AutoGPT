"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useContext, useState } from "react";
import { CopilotChatActionsContext } from "../../../CopilotChatActionsProvider/useCopilotChatActions";
import type { MessagePart } from "../../helpers";
import { getNextStepSuggestions } from "./helpers";

export function useNextStepChips(parts: MessagePart[]) {
  const isEnabled = useGetFlag(Flag.COPILOT_NEXT_STEP_CHIPS);
  // Read the context rather than the throwing hook: the message list also
  // renders on surfaces that mount no provider, and a chip that cannot send
  // should simply not appear there.
  const actions = useContext(CopilotChatActionsContext);
  const [sentLabel, setSentLabel] = useState<string | null>(null);

  const suggestions = isEnabled && actions ? getNextStepSuggestions(parts) : [];

  async function handleSelect(label: string) {
    if (sentLabel || !actions) return;
    setSentLabel(label);
    try {
      await actions.onSend(label);
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
