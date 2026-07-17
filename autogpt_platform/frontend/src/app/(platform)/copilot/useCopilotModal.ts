import { parseAsStringLiteral, useQueryState } from "nuqs";

export const COPILOT_MODALS = ["integrations", "skills", "scheduled"] as const;
export type CopilotModalType = (typeof COPILOT_MODALS)[number];

export function useCopilotModal() {
  const [modal, setModal] = useQueryState(
    "modal",
    parseAsStringLiteral(COPILOT_MODALS),
  );

  function openModal(next: CopilotModalType) {
    void setModal(next);
  }

  function closeModal() {
    void setModal(null);
  }

  return { modal, openModal, closeModal };
}
