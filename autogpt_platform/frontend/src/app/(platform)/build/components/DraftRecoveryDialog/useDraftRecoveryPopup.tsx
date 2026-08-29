import { useEffect, useRef } from "react";
import { useDraftManager } from "../FlowEditor/Flow/useDraftManager";

export const useDraftRecoveryPopup = (isInitialLoadComplete: boolean) => {
  const popupRef = useRef<HTMLDivElement>(null);

  const {
    isRecoveryOpen: isOpen,
    savedAt,
    nodeCount,
    edgeCount,
    diff,
    loadDraft: onLoad,
    discardDraft: onDiscard,
  } = useDraftManager(isInitialLoadComplete);

  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (
        popupRef.current &&
        !popupRef.current.contains(event.target as Node)
      ) {
        onDiscard();
      }
    };

    const timeoutId = setTimeout(() => {
      document.addEventListener("mousedown", handleClickOutside);
    }, 100);

    return () => {
      clearTimeout(timeoutId);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen, onDiscard]);

  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onDiscard();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, onDiscard]);
  return {
    popupRef,
    isOpen,
    nodeCount,
    edgeCount,
    diff,
    savedAt,
    onLoad,
    onDiscard,
  };
};
