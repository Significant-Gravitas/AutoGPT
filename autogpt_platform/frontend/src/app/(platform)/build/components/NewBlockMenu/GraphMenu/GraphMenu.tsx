import React from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { MagnifyingGlassIcon } from "@phosphor-icons/react";
import { GraphSearchContent } from "../GraphMenuContent/GraphContent";
import { ControlPanelButton } from "../ControlPanelButton";
import { CustomNode } from "@/components/CustomNode";
import { useGraphMenu } from "./useGraphMenu";

interface GraphSearchMenuProps {
  nodes: CustomNode[];
  blockMenuSelected: "save" | "block" | "search" | "";
  setBlockMenuSelected: React.Dispatch<
    React.SetStateAction<"" | "save" | "block" | "search">
  >;
  onNodeSelect: (nodeId: string) => void;
  onNodeHover?: (nodeId: string | null) => void;
}

export const GraphSearchMenu: React.FC<GraphSearchMenuProps> = ({
  nodes,
  blockMenuSelected,
  setBlockMenuSelected,
  onNodeSelect,
  onNodeHover,
}) => {
  const {
    open,
    searchQuery,
    setSearchQuery,
    filteredNodes,
    handleNodeSelect,
    handleOpenChange,
  } = useGraphMenu({
    nodes,
    blockMenuSelected,
    setBlockMenuSelected,
    onNodeSelect,
  });

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger className="hover:cursor-pointer">
        <ControlPanelButton
          data-id="graph-search-control-popover-trigger"
          data-testid="graph-search-control-button"
          selected={blockMenuSelected === "search"}
          className="rounded-none"
        >
          <MagnifyingGlassIcon className="h-5 w-6" strokeWidth={2} />
        </ControlPanelButton>
      </PopoverTrigger>

      <PopoverContent
        side="right"
        align="start"
        sideOffset={16}
        className="absolute h-[75vh] w-[46.625rem] overflow-hidden rounded-[1rem] border-none p-0 shadow-[0_2px_6px_0_rgba(0,0,0,0.05)]"
        data-id="graph-search-popover-content"
      >
        <GraphSearchContent
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          filteredNodes={filteredNodes}
          onNodeSelect={handleNodeSelect}
          onNodeHover={onNodeHover}
        />
      </PopoverContent>
    </Popover>
  );
};