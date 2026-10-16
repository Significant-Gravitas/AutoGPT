import { useState } from "react";
import type { DragEvent } from "react";
import { TASK_CARD_MIME } from "../../helpers";

interface Args {
  columnId: string;
  onMoveCard: (sessionId: string, columnId: string) => void;
}

export function useBoardColumn({ columnId, onMoveCard }: Args) {
  const [isOver, setIsOver] = useState(false);

  function handleDragOver(e: DragEvent<HTMLElement>) {
    if (!e.dataTransfer.types.includes(TASK_CARD_MIME)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    setIsOver(true);
  }

  function handleDragLeave(e: DragEvent<HTMLElement>) {
    if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
    setIsOver(false);
  }

  function handleDrop(e: DragEvent<HTMLElement>) {
    e.preventDefault();
    setIsOver(false);
    const sessionId = e.dataTransfer.getData(TASK_CARD_MIME);
    if (sessionId) onMoveCard(sessionId, columnId);
  }

  return { isOver, handleDragOver, handleDragLeave, handleDrop };
}
