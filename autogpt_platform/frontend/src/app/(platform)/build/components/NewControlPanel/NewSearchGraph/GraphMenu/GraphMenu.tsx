"use client";

import { useControlPanelStore } from "@/app/(platform)/build/stores/controlPanelStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { MagnifyingGlass } from "@phosphor-icons/react";
import { useReactFlow } from "@xyflow/react";
import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { ControlPanelButton } from "../../ControlPanelButton";
import { GraphSearchContent } from "../GraphMenuContent/GraphContent";
import { useGraphMenu } from "./useGraphMenu";

export function GraphSearchMenu() {
  const nodes = useNodeStore(useShallow((state) => state.nodes));
  const { graphSearchOpen, setGraphSearchOpen } = useControlPanelStore();
  const reactFlow = useReactFlow();

  const isMac = useMemo(
    () =>
      typeof navigator !== "undefined" &&
      /Mac|iPhone|iPad|iPod/.test(navigator.platform),
    [],
  );

  const { searchQuery, setSearchQuery, filteredNodes, handleNodeSelect } =
    useGraphMenu({
      nodes,
      onNodeSelect(nodeID) {
        reactFlow.fitView({
          nodes: [{ id: nodeID }],
          duration: 600,
          maxZoom: 1.5,
          padding: 0.5,
        });
      },
    });

  return (
    <Popover
      open={graphSearchOpen}
      onOpenChange={(open: boolean) => {
        setGraphSearchOpen(open);
        if (!open) {
          setSearchQuery("");
        }
      }}
    >
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild className="hover:cursor-pointer">
            <ControlPanelButton
              data-id="graph-search-control-popover-trigger"
              data-testid="graph-search-control-button"
              selected={graphSearchOpen}
              className="rounded-none"
            >
              <MagnifyingGlass className="size-5" />
            </ControlPanelButton>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="right">
          Search Graph ({isMac ? "Cmd" : "Ctrl"}+F)
        </TooltipContent>
      </Tooltip>

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
        />
      </PopoverContent>
    </Popover>
  );
}
