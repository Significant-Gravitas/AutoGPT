import React from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ToyBrick } from "lucide-react";
import { BlockMenuContent } from "../BlockMenuContent/BlockMenuContent";
import { ControlPanelButton } from "../ControlPanelButton";
import { useBlockMenu } from "./useBlockMenu";

interface BlockMenuProps {
  pinBlocksPopover: boolean;
  blockMenuSelected: "save" | "block" | "";
  setBlockMenuSelected: React.Dispatch<
    React.SetStateAction<"" | "save" | "block">
  >;
}

export const BlockMenu: React.FC<BlockMenuProps> = ({
  pinBlocksPopover,
  blockMenuSelected,
  setBlockMenuSelected,
}) => {
  const {open, onOpen} = useBlockMenu({pinBlocksPopover, setBlockMenuSelected});
  return (
    <Popover open={pinBlocksPopover ? true : open} onOpenChange={onOpen}>
      <PopoverTrigger className="hover:cursor-pointer">
        <ControlPanelButton
          data-id="blocks-control-popover-trigger"
          data-testid="blocks-control-blocks-button"
          selected={blockMenuSelected === "block"}
          className="rounded-none"
        >
           {/* Need to find phosphor icon alternative for this lucide icon */}
          <ToyBrick className="h-5 w-6" strokeWidth={2} /> 
        </ControlPanelButton>
      </PopoverTrigger>

      <PopoverContent
        side="right"
        align="start"
        sideOffset={16}
        className="absolute h-[75vh] w-[46.625rem] overflow-hidden rounded-[1rem] border-none p-0 shadow-[0_2px_6px_0_rgba(0,0,0,0.05)]"
        data-id="blocks-control-popover-content"
      >
          <BlockMenuContent />
      </PopoverContent>
    </Popover>
  );
};