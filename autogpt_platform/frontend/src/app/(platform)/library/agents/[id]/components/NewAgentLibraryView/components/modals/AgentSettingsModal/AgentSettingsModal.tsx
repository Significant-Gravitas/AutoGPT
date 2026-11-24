"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useAgentSafeMode } from "@/hooks/useAgentSafeMode";
import { useState } from "react";
import { Settings01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SafeModeSetting } from "./components/SafeModeSetting/SafeModeSetting";

interface Props {
  agent: LibraryAgent;
  controlledOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function AgentSettingsModal({
  agent,
  controlledOpen,
  onOpenChange,
}: Props) {
  const [internalIsOpen, setInternalIsOpen] = useState(false);
  const isOpen = controlledOpen !== undefined ? controlledOpen : internalIsOpen;

  function setIsOpen(open: boolean) {
    if (onOpenChange) {
      onOpenChange(open);
    } else {
      setInternalIsOpen(open);
    }
  }

  const {
    currentHITLSafeMode,
    showHITLToggle,
    handleHITLToggle,
    currentSensitiveActionSafeMode,
    showSensitiveActionToggle,
    handleSensitiveActionToggle,
    isPending,
    shouldShowToggle,
  } = useAgentSafeMode(agent);

  if (!shouldShowToggle) return null;

  return (
    <Dialog
      controlled={{ isOpen, set: setIsOpen }}
      styling={{ maxWidth: "600px", maxHeight: "90vh" }}
      title="Workflow Settings"
    >
      {controlledOpen === undefined && (
        <Dialog.Trigger>
          <Button
            variant="ghost"
            size="inline"
            // oxlint-disable-next-line design-system/no-restyle -- a text-link-like trigger; no Button size has 4px side padding at 36px tall
            className="m-0 h-9 px-1"
            aria-label="Workflow Settings"
          >
            <span className="contents text-foreground-secondary">
              <Icon icon={Settings01Icon} size={18} />
            </span>
            <Text variant="small">Workflow Settings</Text>
          </Button>
        </Dialog.Trigger>
      )}
      <Dialog.Content>
        <div className="space-y-6">
          {showHITLToggle && (
            <SafeModeSetting
              label="Human-in-the-loop approval"
              description="The workflow will pause at human-in-the-loop blocks and wait for your review before continuing"
              checked={currentHITLSafeMode || false}
              onCheckedChange={handleHITLToggle}
              disabled={isPending}
            />
          )}
          {showSensitiveActionToggle && (
            <SafeModeSetting
              label="Sensitive action approval"
              description="The workflow will pause at sensitive action blocks and wait for your review before continuing"
              checked={currentSensitiveActionSafeMode}
              onCheckedChange={handleSensitiveActionToggle}
              disabled={isPending}
            />
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
