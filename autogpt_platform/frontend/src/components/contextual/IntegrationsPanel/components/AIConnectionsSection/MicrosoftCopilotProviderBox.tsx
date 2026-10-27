"use client";

import { useState } from "react";
import { Text } from "@/components/atoms/Text/Text";
import { MicrosoftConnectionPanel } from "@/components/contextual/MicrosoftConnection/MicrosoftConnectionPanel";
import { useMicrosoftBrowserSignIn } from "@/components/contextual/MicrosoftConnection/useMicrosoftBrowserSignIn";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  MICROSOFT_BASE_SCOPES,
  MICROSOFT_COPILOT_SCOPES,
} from "@/lib/microsoft-setup";
import { ProviderBox } from "./ProviderBox";

interface Props {
  isLinked: boolean;
  onSuccess: () => void;
}

export function MicrosoftCopilotProviderBox({ isLinked, onSuccess }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const { browserAvailable } = useMicrosoftBrowserSignIn();

  function handleDone() {
    setIsOpen(false);
    onSuccess();
  }

  const card = (
    <ProviderBox
      name="Microsoft 365 Copilot"
      logoSrc="/integrations/microsoft.webp"
      state={isLinked ? "connected" : "available"}
    />
  );

  if (isLinked) return card;

  return (
    <Dialog
      title="Connect Microsoft 365 Copilot"
      variant="compact"
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Trigger>{card}</Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="small" as="p" tone="muted">
            Requires a paid Microsoft Copilot or Copilot Business add-on from
            your work or school organization. The included Microsoft 365 Copilot
            Chat does not qualify. This connection answers chats but does not
            run AutoGPT tools.
          </Text>
          <MicrosoftConnectionPanel
            browserAvailable={browserAvailable}
            scopes={[...MICROSOFT_BASE_SCOPES, ...MICROSOFT_COPILOT_SCOPES]}
            onDone={handleDone}
          />
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
