"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { SecurityCheckIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { getSafetyChecks, markSafetyPopupSeen } from "./helpers";

interface Props {
  agent: LibraryAgent;
  onAcknowledge: () => void;
  isOpen: boolean;
}

export function AIAgentSafetyPopup({ agent, onAcknowledge, isOpen }: Props) {
  function handleAcknowledge() {
    markSafetyPopupSeen(agent.id);
    onAcknowledge();
  }

  if (!isOpen) return null;

  const checks = getSafetyChecks(agent);

  return (
    <Dialog
      title="Safety Checks"
      controlled={{ isOpen, set: () => {} }}
      styling={{ maxWidth: "480px" }}
    >
      <Dialog.Content>
        <div className="flex flex-col items-center p-6 text-center">
          <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-info-surface text-info-foreground">
            <Icon icon={SecurityCheckIcon} size={32} />
          </div>

          <Text variant="h3" className="mb-4">
            Safety Checks Enabled
          </Text>

          <Text variant="body" tone="body" className="mb-2">
            AI-generated workflows may take actions that affect your data or
            external systems.
          </Text>

          {checks.sensitiveActions ? (
            <Text variant="body" tone="body" className="mb-8">
              AutoGPT includes safety checks so you&apos;ll always have the
              opportunity to review and approve sensitive actions before they
              happen.
            </Text>
          ) : (
            <Text variant="body" tone="body" className="mb-8">
              This workflow will pause at its human-in-the-loop blocks and wait
              for your review before continuing.
            </Text>
          )}

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
