import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { isKey } from "@/lib/keyboard";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";
import { useState } from "react";
import type { Card } from "../../../../board-helpers";
import type { AssigneeIdentity } from "./components/AssigneeButton/AssigneeButton";

const OTTO: AssigneeIdentity = {
  name: "Otto",
  roleLabel: "Head of AI",
  avatarUrl: null,
  isAutopilot: true,
};

interface Args {
  title: string;
  card: Card | null;
  onRename: (title: string) => void;
}

export function useTaskPanelHeader({ title, card, onRename }: Args) {
  const [draft, setDraft] = useState<string | null>(null);
  // Without experts the only assignee is Otto, so there is no one to hand to.
  const { isExpertsEnabled, expertsById } = useExpertMap();
  const expertId = card?.expert_id ?? null;
  const expert = expertId ? expertsById.get(expertId) : undefined;
  const assignee: AssigneeIdentity = expertId
    ? {
        name: expert?.name ?? "This expert",
        roleLabel:
          expert?.jobTitle ||
          (expert?.role ? getExpertRoleLabel(expert.role) : "Expert"),
        avatarUrl: expert?.avatarUrl ?? null,
        color: expert?.color,
        isAutopilot: false,
      }
    : OTTO;

  function startEditing() {
    setDraft(title);
  }

  function stopEditing() {
    setDraft(null);
  }

  function handleChange(
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    setDraft(event.target.value);
  }

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    if (isKey(event, "Escape")) {
      event.preventDefault();
      stopEditing();
      return;
    }
    if (!isKey(event, "Enter")) return;
    event.preventDefault();
    const next = draft?.trim() ?? "";
    if (next && next !== title) onRename(next);
    stopEditing();
  }

  return {
    isEditing: draft !== null,
    assignee,
    canChangeAssignee: !!card && !card.started && isExpertsEnabled,
    canReassign:
      !!card?.started && !card.reassigned_to_session_id && isExpertsEnabled,
    draft: draft ?? "",
    startEditing,
    stopEditing,
    handleChange,
    handleKeyDown,
  };
}
