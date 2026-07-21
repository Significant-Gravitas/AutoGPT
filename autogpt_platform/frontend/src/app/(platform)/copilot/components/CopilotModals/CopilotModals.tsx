"use client";

import { IntegrationsPanel } from "@/components/contextual/IntegrationsPanel/IntegrationsPanel";
import { SchedulesPanel } from "@/components/contextual/SchedulesPanel/SchedulesPanel";
import { SkillsPanel } from "@/components/contextual/SkillsPanel/SkillsPanel";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useCopilotUIStore } from "../../store";
import { useCopilotModal } from "../../useCopilotModal";

export function CopilotModals() {
  const { modal, closeModal } = useCopilotModal();
  const setInitialPrompt = useCopilotUIStore((s) => s.setInitialPrompt);

  function handleGuidedPrompt(prompt: string) {
    closeModal();
    setInitialPrompt(prompt);
  }

  function handleOpenChange(open: boolean) {
    if (!open) closeModal();
  }

  return (
    <>
      <Dialog
        controlled={{ isOpen: modal === "skills", set: handleOpenChange }}
        styling={{ maxWidth: "44rem" }}
        title="AutoPilot skills"
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
        title="Third Party Integrations"
      >
        <Dialog.Content>
          <IntegrationsPanel withHeading={false} />
        </Dialog.Content>
      </Dialog>
    </>
  );
}
