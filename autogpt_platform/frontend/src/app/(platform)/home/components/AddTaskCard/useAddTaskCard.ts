import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { usePostV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { isKey } from "@/lib/keyboard";
import { useRef, useState } from "react";
import {
  BOARD_CREATE_MUTATION_KEY,
  parseMention,
  refusalReason,
} from "../../board-helpers";
import { useBoardRef } from "../../BoardContext";
import { useBoardCache } from "../../useBoardCache";

interface Args {
  columnKey: string;
  autoExpand: boolean;
}

export function useAddTaskCard({ columnKey, autoExpand }: Args) {
  const [isEditing, setIsEditing] = useState(autoExpand);
  const [title, setTitle] = useState("");
  const [expertId, setExpertId] = useState<string | null>(null);
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);
  const { isExpertsEnabled } = useExpertMap();
  const { boardRef, isDefault } = useBoardRef();
  const { settleBoard } = useBoardCache(boardRef);

  const { mutate: createTask, isPending } = usePostV2CreateSession({
    mutation: {
      mutationKey: BOARD_CREATE_MUTATION_KEY,
      onSuccess: () => {
        setTitle("");
        setExpertId(null);
      },
      onError: (error) => {
        toast({
          title: refusalReason(error) ?? "Couldn't add the task",
          variant: "destructive",
        });
      },
      onSettled: () => settleBoard(),
    },
  });

  function reset() {
    setIsEditing(false);
    setTitle("");
    setExpertId(null);
    setIsPickerOpen(false);
  }

  function handleChange(
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    const value = event.target.value;
    if (!isExpertsEnabled) {
      setTitle(value);
      return;
    }
    const mention = parseMention(value);
    setTitle(mention.text);
    if (mention.mentionOpen) setIsPickerOpen(true);
  }

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    if (isKey(event, "Escape")) {
      event.preventDefault();
      reset();
      return;
    }
    if (!isKey(event, "Enter")) return;
    event.preventDefault();
    const trimmed = title.trim();
    if (!trimmed || isPending) return;
    createTask({
      data: {
        title: trimmed,
        board_column: columnKey,
        expert_id: expertId,
        board_id: isDefault ? undefined : boardRef,
      },
    });
  }

  function handleSelectExpert(id: string | null) {
    setExpertId(id);
  }

  function handlePickerCloseAutoFocus(event: Event) {
    event.preventDefault();
    inputRef.current?.focus();
  }

  return {
    isEditing,
    title,
    expertId,
    isPickerOpen,
    isExpertsEnabled,
    isPending,
    inputRef,
    startEditing: () => setIsEditing(true),
    setIsPickerOpen,
    handleChange,
    handleKeyDown,
    handleSelectExpert,
    handlePickerCloseAutoFocus,
  };
}
