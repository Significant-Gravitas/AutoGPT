"use client";

import { ConnectServiceDialog } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/ConnectServiceDialog";
import { IntegrationsPanel } from "@/components/contextual/IntegrationsPanel/IntegrationsPanel";
import { SchedulesPanel } from "@/components/contextual/SchedulesPanel/SchedulesPanel";
import { SkillsPanel } from "@/components/contextual/SkillsPanel/SkillsPanel";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { grantToExpert } from "@/services/experts/grant-to-expert";
import { useQueryClient } from "@tanstack/react-query";
import { useCopilotUIStore } from "../../store";
import { useCopilotModal } from "../../useCopilotModal";

export function CopilotModals() {
  const { modal, closeModal } = useCopilotModal();
  const setInitialPrompt = useCopilotUIStore((s) => s.setInitialPrompt);
  const expert = useCopilotUIStore((s) => s.contextPanelExpert);
  const queryClient = useQueryClient();
  const { toast } = useToast();

  function handleGuidedPrompt(prompt: string) {
    closeModal();
    setInitialPrompt(prompt);
  }

  function handleOpenChange(open: boolean) {
    if (!open) closeModal();
  }

  async function handleConnected(credential: { id: string }) {
    if (!expert) return;
    const ok = await grantToExpert(queryClient, expert.id, credential.id);
    if (!ok) {
      toast({
        title: `Connected, but could not give ${expert.name} access`,
        description: "Grant it from the Integrations tab.",
        variant: "destructive",
      });
    }
  }

  return (
    <>
      <Dialog
        controlled={{ isOpen: modal === "skills", set: handleOpenChange }}
        styling={{ maxWidth: "44rem" }}
        title="Skills"
      >
        <Dialog.Content>
          <SkillsPanel
            withHeading={false}
            onGuidedPrompt={handleGuidedPrompt}
          />
        </Dialog.Content>
      </Dialog>

      <Dialog
        controlled={{ isOpen: modal === "scheduled", set: handleOpenChange }}
        styling={{ maxWidth: "44rem" }}
        title="Scheduled"
      >
        <Dialog.Content>
          <SchedulesPanel
            withHeading={false}
            onGuidedPrompt={handleGuidedPrompt}
          />
        </Dialog.Content>
      </Dialog>

      <Dialog
        controlled={{ isOpen: modal === "integrations", set: handleOpenChange }}
        styling={{ maxWidth: "56rem" }}
        title="Integrations"
      >
        <Dialog.Content>
          <IntegrationsPanel
            withHeading={false}
            preferMcp
            onConnected={expert ? handleConnected : undefined}
          />
        </Dialog.Content>
      </Dialog>

      <ConnectServiceDialog
        preferMcp
        open={modal === "connect"}
        onOpenChange={handleOpenChange}
        onConnected={expert ? handleConnected : undefined}
      />
    </>
  );
}
