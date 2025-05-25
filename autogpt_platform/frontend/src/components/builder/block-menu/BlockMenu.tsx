// BLOCK MENU TODO: Currently when i click on the control panel button, if it is already open, then it needs to close, currently its not happening

import React, { useCallback, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import ControlPanelButton from "@/components/builder/block-menu/ControlPanelButton";
import { ToyBrick } from "lucide-react";
import BlockMenuContent from "./BlockMenuContent";
import { BlockMenuStateProvider } from "./block-menu-provider";
import { Block } from "@/lib/autogpt-server-api";

interface BlockMenuProps {
  addBlock: (
    blockId: string,
    nodeType: string,
    hardcodedValues: any | undefined,
    nodeSchema: Block | undefined,
  ) => void;
  pinBlocksPopover: boolean;
  blockMenuSelected: "save" | "block" | "";
  setBlockMenuSelected: React.Dispatch<
    React.SetStateAction<"" | "save" | "block">
  >;
}

export const BlockMenu: React.FC<BlockMenuProps> = ({
  addBlock,
  pinBlocksPopover,
  blockMenuSelected,
  setBlockMenuSelected,
}) => {
  const [open, setOpen] = useState(false);
  const onOpen = useCallback(
    (newOpen: boolean) => {
      if (!pinBlocksPopover) {
        setOpen(newOpen);
        setBlockMenuSelected(newOpen ? "block" : "");
      }
    },
    [pinBlocksPopover, setOpen, setBlockMenuSelected],
  );

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
        <BlockMenuStateProvider addNode={addBlock}>
          <BlockMenuContent />
        </BlockMenuStateProvider>
      </PopoverContent>
    </Popover>
  );
};
