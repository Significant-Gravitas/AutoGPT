"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useAgentSafeMode } from "@/hooks/useAgentSafeMode";
import { GearIcon } from "@phosphor-icons/react";
import { useState } from "react";

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
      title="Agent Settings"
    >
      {controlledOpen === undefined && (
        <Dialog.Trigger>
          <Button
            variant="ghost"
            size="small"
            className="m-0 min-w-0 rounded-full p-0 px-1"
            aria-label="Agent Settings"
          >
            <GearIcon size={18} className="text-zinc-600" />
            <Text variant="small">Agent Settings</Text>
          </Button>
        </Dialog.Trigger>
      )}
      <Dialog.Content>
        <div className="space-y-6">
          {showHITLToggle && (
            <div className="flex w-full flex-col items-start gap-4 rounded-xl border border-zinc-100 bg-white p-6">
              <div className="flex w-full items-start justify-between gap-4">
                <div className="flex-1">
                  <Text variant="large-semibold">
                    Human-in-the-loop approval
                  </Text>
                  <Text variant="large" className="mt-1 text-zinc-900">
                    The agent will pause at human-in-the-loop blocks and wait
                    for your review before continuing
                  </Text>
                </div>
                <Switch
                  checked={currentHITLSafeMode || false}
                  onCheckedChange={handleHITLToggle}
                  disabled={isPending}
                  className="mt-1"
                />
              </div>
            </div>
          )}
          {showSensitiveActionToggle && (
            <div className="flex w-full flex-col items-start gap-4 rounded-xl border border-zinc-100 bg-white p-6">
              <div className="flex w-full items-start justify-between gap-4">
                <div className="flex-1">
                  <Text variant="large-semibold">
                    Sensitive action approval
                  </Text>
                  <Text variant="large" className="mt-1 text-zinc-900">
                    The agent will pause at sensitive action blocks and wait for
                    your review before continuing
                  </Text>
                </div>
                <Switch
                  checked={currentSensitiveActionSafeMode}
                  onCheckedChange={handleSensitiveActionToggle}
                  disabled={isPending}
                  className="mt-1"
                />
              </div>
            </div>
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
