import { Separator } from "@/components/__legacy__/ui/separator";
import { ControlPanelButton } from "./ControlPanelButton";
import { ArrowUUpLeftIcon, ArrowUUpRightIcon } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useHistoryStore } from "../../stores/historyStore";

import { useEffect } from "react";

export const UndoRedoButtons = () => {
  const { undo, redo, canUndo, canRedo } = useHistoryStore();

  // Keyboard shortcuts for undo and redo
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
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
          <ControlPanelButton as="button" disabled={!canUndo()} onClick={undo}>
            <ArrowUUpLeftIcon className="h-6 w-6" />
          </ControlPanelButton>
        </TooltipTrigger>
        <TooltipContent side="right">Undo</TooltipContent>
      </Tooltip>
      <Separator className="text-[#E1E1E1]" />
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <ControlPanelButton as="button" disabled={!canRedo()} onClick={redo}>
            <ArrowUUpRightIcon className="h-6 w-6" />
          </ControlPanelButton>
        </TooltipTrigger>
        <TooltipContent side="right">Redo</TooltipContent>
      </Tooltip>
    </>
  );
};
