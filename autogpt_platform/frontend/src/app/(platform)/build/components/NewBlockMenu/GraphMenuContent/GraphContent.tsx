import React from "react";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { beautifyString, getPrimaryCategoryColor } from "@/lib/utils";
import { SearchableNode } from "../GraphMenuSearchBar/useGraphMenuSearchBar";
import { TextRenderer } from "@/components/ui/render";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { GraphMenuSearchBar } from "../GraphMenuSearchBar/GraphMenuSearchBar";
import { useGraphContent } from "./useGraphContent";

interface GraphSearchContentProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  filteredNodes: SearchableNode[];
  onNodeSelect: (nodeId: string) => void;
  onNodeHover?: (nodeId: string | null) => void;
}

export const GraphSearchContent: React.FC<GraphSearchContentProps> = ({
  searchQuery,
  onSearchChange,
  filteredNodes,
  onNodeSelect,
  onNodeHover,
}) => {
  const {
    selectedIndex,
    setSelectedIndex,
    handleKeyDown,
    getNodeInputOutputSummary,
  } = useGraphContent({
    searchQuery,
    filteredNodes,
    onNodeSelect,
  });

  return (
    <div className="flex h-full w-full flex-col">
      {/* Search Bar */}
      <GraphMenuSearchBar
        searchQuery={searchQuery}
        onSearchChange={onSearchChange}
        onKeyDown={handleKeyDown}
      />

      <Separator className="h-[1px] w-full text-zinc-300" />

      {/* Search Results */}
      <div className="flex-1 overflow-hidden">
        {searchQuery && (
          <div className="px-4 py-2 text-xs text-gray-500">
            Found {filteredNodes.length} node
            {filteredNodes.length !== 1 ? "s" : ""}
          </div>
        )}
        <ScrollArea className="h-full w-full">
          {filteredNodes.length === 0 ? (
            <div className="flex h-32 items-center justify-center text-sm text-gray-500 dark:text-gray-400">
              {searchQuery
                ? "No nodes found matching your search"
                : "Start typing to search nodes"}
            </div>
          ) : (
            filteredNodes.map((node, index) => {
              // Safety check for node data
              if (!node || !node.data) {
                return null;
              }

              const nodeTitle =
                node.data?.metadata?.customized_name ||
                beautifyString(node.data?.blockType || "").replace(
                  / Block$/,
                  "",
                );
              const nodeType = beautifyString(
                node.data?.blockType || "",
              ).replace(/ Block$/, "");

              return (
                <TooltipProvider key={node.id}>
                  <Tooltip delayDuration={300}>
                    <TooltipTrigger asChild>
                      <div
                        className={`mx-4 my-2 flex h-20 cursor-pointer rounded-lg border border-zinc-200 bg-white ${
                          index === selectedIndex
                            ? "border-zinc-400 shadow-md"
                            : "hover:border-zinc-300 hover:shadow-sm"
                        }`}
                        onClick={() => onNodeSelect(node.id)}
                        onMouseEnter={() => {
                          setSelectedIndex(index);
                          onNodeHover?.(node.id);
                        }}
                        onMouseLeave={() => onNodeHover?.(null)}
                      >
                        <div
                          className={`h-full w-3 rounded-l-[7px] ${getPrimaryCategoryColor(node.data?.categories)}`}
                        />
                        <div className="mx-3 flex flex-1 items-center justify-between">
                          <div className="mr-2 min-w-0">
                            <span className="block truncate pb-1 text-sm font-semibold text-zinc-800">
                              <TextRenderer
                                value={nodeTitle}
                                truncateLengthLimit={45}
                              />
                            </span>
                            <span className="block break-all text-xs font-normal text-zinc-500">
                              <TextRenderer
                                value={
                                  getNodeInputOutputSummary(node) ||
                                  node.data?.description ||
                                  ""
                                }
                                truncateLengthLimit={165}
                              />
                            </span>
                          </div>
                        </div>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <div className="space-y-1">
                        <div className="font-semibold">
                          Node Type: {nodeType}
                        </div>
                        {node.data?.metadata?.customized_name && (
                          <div className="text-xs text-gray-500">
                            Custom Name: {node.data.metadata.customized_name}
                          </div>
                        )}
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              );
            })
          )}
        </ScrollArea>
      </div>
    </div>
  );
};
