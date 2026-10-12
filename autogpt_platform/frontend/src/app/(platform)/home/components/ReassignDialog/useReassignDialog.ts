import type { RecipientOption } from "@/app/(platform)/copilot/components/ChatInput/components/RecipientChip";
import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { usePostV2ReassignSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import {
  applyReassign,
  BOARD_MUTATION_KEY,
  type Card,
} from "../../board-helpers";
import { useBoardCache } from "../../useBoardCache";
import { assigneeOptions } from "../AssigneePicker/helpers";
import { reassignErrorToast } from "./helpers";

interface Args {
  card: Card | null;
  onClose: () => void;
  onReassigned: (newSessionId: string) => void;
}

export function useReassignDialog({ card, onClose, onReassigned }: Args) {
  const { activeExperts } = useExpertMap();
  const { applyOptimistic, settleBoard } = useBoardCache();
  const [selected, setSelected] = useState<RecipientOption | null>(null);

  const { mutate, isPending } = usePostV2ReassignSession({
    mutation: {
      mutationKey: [...BOARD_MUTATION_KEY, "reassign"],
      onSuccess: async (response) => {
        if (response.status !== 200) return;
        const { old_session, new_session } = response.data;
        await applyOptimistic((board) =>
          applyReassign(board, old_session, new_session),
        );
        onClose();
        onReassigned(new_session.id);
      },
      onError: (error) => toast(reassignErrorToast(error)),
      onSettled: () => settleBoard(),
    },
  });

  const currentId = card?.expert_id ?? null;
  const options = card
    ? assigneeOptions(activeExperts).filter((option) => option.id !== currentId)
    : [];

  function handleOpenChange(open: boolean) {
    if (!open) onClose();
  }

  function handleConfirm() {
    if (!card || !selected || isPending) return;
    mutate({ sessionId: card.id, data: { expert_id: selected.id } });
  }

  return {
    options,
    selected,
    isPending,
    select: setSelected,
    handleOpenChange,
    handleConfirm,
  };
}
