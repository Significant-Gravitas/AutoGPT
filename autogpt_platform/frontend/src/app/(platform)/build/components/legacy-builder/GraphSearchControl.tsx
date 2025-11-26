import React from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { Button } from "@/components/atoms/Button/Button";
import { MagnifyingGlassIcon } from "@radix-ui/react-icons";
import { CustomNode } from "@/app/(platform)/build/components/legacy-builder/CustomNode/CustomNode";
import { GraphSearchContent } from "../NewControlPanel/NewSearchGraph/GraphMenuContent/GraphContent";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useGraphMenu } from "../NewControlPanel/NewSearchGraph/GraphMenu/useGraphMenu";

interface GraphSearchControlProps {
  nodes: CustomNode[];
  onNodeSelect: (nodeId: string) => void;
  onNodeHover?: (nodeId: string | null) => void;
}

export function GraphSearchControl({
  nodes,
  onNodeSelect,
  onNodeHover,
}: GraphSearchControlProps) {
  // Use the same hook as GraphSearchMenu for consistency
  const {
    open,
    searchQuery,
    setSearchQuery,
    filteredNodes,
    handleNodeSelect,
    handleOpenChange,
  } = useGraphMenu({
    nodes,
    blockMenuSelected: "", // We don't need to track this in the old control panel
    setBlockMenuSelected: () => {}, // Not needed in this context
    onNodeSelect,
  });

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <Tooltip delayDuration={500}>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              data-id="graph-search-control-trigger"
              data-testid="graph-search-control-button"
              name="Search"
              className="dark:hover:bg-slate-800"
            >
              <MagnifyingGlassIcon className="h-5 w-5" />
            </Button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="right">Search Graph</TooltipContent>
      </Tooltip>

      <PopoverContent
        side="right"
        sideOffset={22}
        align="start"
        alignOffset={-50} // Offset upward to align with control panel top
        className="absolute -top-3 w-[17rem] rounded-xl border-none p-0 shadow-none md:w-[30rem]"
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
}
