"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Key, storage } from "@/services/storage/local-storage";
import { ShieldCheckIcon } from "@phosphor-icons/react";
import { useCallback, useEffect, useState } from "react";

interface Props {
  /**
   * Called when the user acknowledges the popup
   */
  onAcknowledge: () => void;
  /**
   * Whether the popup should be shown (controlled externally)
   */
  isOpen: boolean;
}

/**
 * One-time safety popup shown the first time a user runs an AI-generated agent
 * with sensitive actions or human-in-the-loop blocks.
 */
export function AIAgentSafetyPopup({ onAcknowledge, isOpen }: Props) {
  function handleAcknowledge() {
    // Mark popup as shown so it won't appear again
    storage.set(Key.AI_AGENT_SAFETY_POPUP_SHOWN, "true");
    onAcknowledge();
  }

  if (!isOpen) return null;

  return (
    <Dialog
      controlled={{ isOpen, set: () => {} }}
      styling={{ maxWidth: "480px" }}
    >
      <Dialog.Content>
        <div className="flex flex-col items-center p-6 text-center">
          <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-blue-50">
            <ShieldCheckIcon
              weight="fill"
              size={32}
              className="text-blue-600"
            />
          </div>

          <Text variant="h3" className="mb-4">
            Safety Checks Enabled
          </Text>

          <Text variant="body" className="mb-2 text-zinc-700">
            AI-generated agents may take actions that affect your data or
            external systems.
          </Text>

          <Text variant="body" className="mb-8 text-zinc-700">
            AutoGPT includes safety checks so you&apos;ll always have the
            opportunity to review and approve sensitive actions before they
            happen.
          </Text>

          <Button
            variant="primary"
            size="large"
            className="w-full"
            onClick={handleAcknowledge}
          >
            Got it
          </Button>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

/**
 * Hook to manage the one-time AI agent safety popup state.
 *
 * @param hasSensitiveAction - Whether the agent has sensitive actions
 * @param hasHumanInTheLoop - Whether the agent has human-in-the-loop blocks
 * @returns Object with shouldShowPopup and dismissPopup function
 */
export function useAIAgentSafetyPopup(
  hasSensitiveAction: boolean,
  hasHumanInTheLoop: boolean,
) {
  const [shouldShowPopup, setShouldShowPopup] = useState(false);
  const [hasChecked, setHasChecked] = useState(false);

  useEffect(() => {
    // Only check once after mount (to avoid SSR issues)
    if (hasChecked) return;

    const hasSeenPopup =
      storage.get(Key.AI_AGENT_SAFETY_POPUP_SHOWN) === "true";
    const isRelevantAgent = hasSensitiveAction || hasHumanInTheLoop;

    setShouldShowPopup(!hasSeenPopup && isRelevantAgent);
    setHasChecked(true);
  }, [hasSensitiveAction, hasHumanInTheLoop, hasChecked]);

  const dismissPopup = useCallback(() => {
    setShouldShowPopup(false);
  }, []);

  return {
    shouldShowPopup,
    dismissPopup,
  };
}
