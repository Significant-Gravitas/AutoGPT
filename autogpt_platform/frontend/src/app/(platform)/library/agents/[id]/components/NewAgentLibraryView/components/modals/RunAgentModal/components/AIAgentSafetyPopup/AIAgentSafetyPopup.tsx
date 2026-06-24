"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Key, storage } from "@/services/storage/local-storage";
import { ShieldCheckIcon } from "@phosphor-icons/react";
import { useCallback, useEffect, useState } from "react";

interface Props {
  agentId: string;
  onAcknowledge: () => void;
  isOpen: boolean;
}

export function AIAgentSafetyPopup({ agentId, onAcknowledge, isOpen }: Props) {
  function handleAcknowledge() {
    // Add this agent to the list of agents for which popup has been shown
    const seenAgentsJson = storage.get(Key.AI_AGENT_SAFETY_POPUP_SHOWN);
    const seenAgents: string[] = seenAgentsJson
      ? JSON.parse(seenAgentsJson)
      : [];

    if (!seenAgents.includes(agentId)) {
      seenAgents.push(agentId);
      storage.set(Key.AI_AGENT_SAFETY_POPUP_SHOWN, JSON.stringify(seenAgents));
    }

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

export function useAIAgentSafetyPopup(
  agentId: string,
  hasSensitiveAction: boolean,
  hasHumanInTheLoop: boolean,
) {
  const [shouldShowPopup, setShouldShowPopup] = useState(false);
  const [hasChecked, setHasChecked] = useState(false);

  useEffect(() => {
    if (hasChecked) return;

    const seenAgentsJson = storage.get(Key.AI_AGENT_SAFETY_POPUP_SHOWN);
    const seenAgents: string[] = seenAgentsJson
      ? JSON.parse(seenAgentsJson)
      : [];
    const hasSeenPopupForThisAgent = seenAgents.includes(agentId);
    const isRelevantAgent = hasSensitiveAction || hasHumanInTheLoop;

    setShouldShowPopup(!hasSeenPopupForThisAgent && isRelevantAgent);
    setHasChecked(true);
  }, [agentId, hasSensitiveAction, hasHumanInTheLoop, hasChecked]);

  const dismissPopup = useCallback(() => {
    setShouldShowPopup(false);
  }, []);

  return {
    shouldShowPopup,
    dismissPopup,
  };
}
