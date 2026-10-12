import { usePostV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { isKey } from "@/lib/keyboard";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { BOARD_QUERY_KEY, parseMention } from "../../board-helpers";

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
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const { mutate: createTask, isPending } = usePostV2CreateSession({
    mutation: {
      onSuccess: () => {
        setTitle("");
        setExpertId(null);
        void queryClient.invalidateQueries({ queryKey: BOARD_QUERY_KEY });
      },
      onError: () => {
        toast({ title: "Couldn't add the task", variant: "destructive" });
      },
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
      data: { title: trimmed, board_column: columnKey, expert_id: expertId },
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
