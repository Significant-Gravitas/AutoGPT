import React, { useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ControlPanelButton } from "@/components/builder/block-menu/ControlPanelButton";
import { ToyBrick } from "lucide-react";
import { BlockMenuContent } from "./BlockMenuContent";
import { BlockMenuStateProvider } from "./block-menu-provider";
import { Block } from "@/lib/autogpt-server-api";

interface BlockMenuProps {
  addNode: (block: Block) => void;
  pinBlocksPopover: boolean;
  blockMenuSelected: "save" | "block" | "";
  setBlockMenuSelected: React.Dispatch<
    React.SetStateAction<"" | "save" | "block">
  >;
}

export const BlockMenu: React.FC<BlockMenuProps> = ({
  addNode,
  pinBlocksPopover,
  blockMenuSelected,
  setBlockMenuSelected,
}) => {
  const [open, setOpen] = useState(false);
  const onOpen = (newOpen: boolean) => {
    if (!pinBlocksPopover) {
      setOpen(newOpen);
      setBlockMenuSelected(newOpen ? "block" : "");
    }
  };

  return (
    <Popover open={pinBlocksPopover ? true : open} onOpenChange={onOpen}>
      <PopoverTrigger className="hover:cursor-pointer">
        <ControlPanelButton
          data-id="blocks-control-popover-trigger"
          data-testid="blocks-control-blocks-button"
          selected={blockMenuSelected === "block"}
          className="rounded-none"
        >
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
        <BlockMenuStateProvider addNode={addNode}>
          <BlockMenuContent />
        </BlockMenuStateProvider>
      </PopoverContent>
    </Popover>
  );
};
