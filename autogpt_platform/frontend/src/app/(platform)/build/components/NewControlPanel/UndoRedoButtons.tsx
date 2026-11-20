import { Separator } from "@/components/atoms/Separator/Separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useHistoryStore } from "../../stores/historyStore";
import { ControlPanelButton } from "./ControlPanelButton";

import { useEffect } from "react";
import {
  ArrowTurnBackwardIcon,
  ArrowTurnForwardIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

const TEXT_INPUT_TYPES = new Set([
  "text",
  "search",
  "email",
  "url",
  "tel",
  "password",
  "number",
]);

function hasOwnTextUndo(target: EventTarget | null): boolean {
  if (target instanceof HTMLTextAreaElement) return !target.readOnly;
  if (target instanceof HTMLInputElement)
    return TEXT_INPUT_TYPES.has(target.type) && !target.readOnly;
  return target instanceof HTMLElement && target.isContentEditable;
}

export const UndoRedoButtons = () => {
  const { undo, redo, canUndo, canRedo } = useHistoryStore();

  // Keyboard shortcuts for undo and redo
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // A text field keeps its own undo/redo, so the workflow history must not take the shortcut there.
      if (hasOwnTextUndo(event.target)) return;

      const isMac = /Mac/i.test(navigator.userAgent);
      const isCtrlOrCmd = isMac ? event.metaKey : event.ctrlKey;

      if (isCtrlOrCmd && event.key === "z" && !event.shiftKey) {
        event.preventDefault();
        if (canUndo()) {
          undo();
        }
      } else if (isCtrlOrCmd && event.key === "y") {
        event.preventDefault();
        if (canRedo()) {
          redo();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [undo, redo, canUndo, canRedo]);

  return (
    <>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <ControlPanelButton
            aria-label="Undo"
            data-id="undo-button"
            disabled={!canUndo()}
            onClick={undo}
          >
            <Icon icon={ArrowTurnBackwardIcon} className="size-5" aria-hidden />
          </ControlPanelButton>
        </TooltipTrigger>
        <TooltipContent side="right">Undo</TooltipContent>
      </Tooltip>
      <Separator />
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <ControlPanelButton
            aria-label="Redo"
            data-id="redo-button"
            disabled={!canRedo()}
            onClick={redo}
          >
            <Icon icon={ArrowTurnForwardIcon} className="size-5" aria-hidden />
          </ControlPanelButton>
        </TooltipTrigger>
        <TooltipContent side="right">Redo</TooltipContent>
      </Tooltip>
    </>
  );
};
