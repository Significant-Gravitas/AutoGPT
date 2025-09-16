import { useState } from "react";

interface useBlockMenuProps {
  // pinBlocksPopover: boolean;
  setBlockMenuSelected: React.Dispatch<
    React.SetStateAction<"" | "save" | "block" | "search">
  >;
}

export const useBlockMenu = ({
  // pinBlocksPopover,
  setBlockMenuSelected,
}: useBlockMenuProps) => {
  const [open, setOpen] = useState(false);
  const onOpen = (newOpen: boolean) => {
    // if (!pinBlocksPopover) {
    setOpen(newOpen);
    setBlockMenuSelected(newOpen ? "block" : "");
    // }
  };

  return {
    open,
    onOpen,
  };
};
