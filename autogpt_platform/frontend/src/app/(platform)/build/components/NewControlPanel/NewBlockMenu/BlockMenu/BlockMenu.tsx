import React from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { BlockMenuContent } from "../BlockMenuContent/BlockMenuContent";
import { ControlPanelButton } from "../../ControlPanelButton";
import { LegoIcon } from "@phosphor-icons/react";
import { useControlPanelStore } from "@/app/(platform)/build/stores/controlPanelStore";

export const BlockMenu = () => {
  const { blockMenuOpen, setBlockMenuOpen } = useControlPanelStore();
  return (
    // pinBlocksPopover ? true : open
    <Popover onOpenChange={setBlockMenuOpen}>
      <PopoverTrigger className="hover:cursor-pointer">
        <ControlPanelButton
          data-id="blocks-control-popover-trigger"
          data-testid="blocks-control-blocks-button"
          selected={blockMenuOpen}
          className="rounded-none"
        >
          {/* Need to find phosphor icon alternative for this lucide icon */}
          <LegoIcon className="h-6 w-6" />
        </ControlPanelButton>
      </PopoverTrigger>

      <PopoverContent
        side="right"
        align="start"
        sideOffset={16}
        className="absolute h-[80vh] w-[46.625rem] overflow-hidden rounded-[1rem] border-none p-0 shadow-[0_2px_6px_0_rgba(0,0,0,0.05)]"
        data-id="blocks-control-popover-content"
      >
        <BlockMenuContent />
      </PopoverContent>
    </Popover>
  );
};
