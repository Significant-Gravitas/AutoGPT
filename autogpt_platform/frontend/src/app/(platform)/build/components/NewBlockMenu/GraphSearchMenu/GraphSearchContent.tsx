import React, { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MagnifyingGlass } from "@phosphor-icons/react";
import { beautifyString, getPrimaryCategoryColor } from "@/lib/utils";
import { SearchableNode } from "./useGraphSearch";
import { TextRenderer } from "@/components/ui/render";

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
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    searchInputRef.current?.focus();
  }, []);

  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, filteredNodes.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === "Enter" && filteredNodes.length > 0) {
      e.preventDefault();
      onNodeSelect(filteredNodes[selectedIndex].id);
    }
  };

  const getNodeInputOutputSummary = (node: SearchableNode) => {
    const inputs = Object.keys(node.data.inputSchema?.properties || {});
    const outputs = Object.keys(node.data.outputSchema?.properties || {});
    const parts = [];
    
    if (inputs.length > 0) {
      parts.push(`Inputs: ${inputs.slice(0, 3).join(", ")}${inputs.length > 3 ? "..." : ""}`);
    }
    if (outputs.length > 0) {
      parts.push(`Outputs: ${outputs.slice(0, 3).join(", ")}${outputs.length > 3 ? "..." : ""}`);
    }
    
    return parts.join(" | ");
  };

  return (
    <Card className="p-3 pb-0 dark:bg-slate-900">
      <CardHeader className="flex flex-col gap-x-8 gap-y-1 p-3 px-2">
        <div className="items-center justify-between">
          <Label
            htmlFor="search-nodes"
            className="whitespace-nowrap text-base font-bold text-black dark:text-white 2xl:text-xl"
          >
            Search Graph
          </Label>
        </div>
        <div className="relative flex items-center">
          <MagnifyingGlass className="absolute m-2 h-5 w-5 text-gray-500 dark:text-gray-400" weight="regular" />
          <Input
            ref={searchInputRef}
            id="search-nodes"
            type="text"
            placeholder="Search nodes, inputs, outputs..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            onKeyDown={handleKeyDown}
            className="rounded-lg px-8 py-5 dark:bg-slate-800 dark:text-white"
            autoComplete="off"
          />
        </div>
        {searchQuery && (
          <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Found {filteredNodes.length} node{filteredNodes.length !== 1 ? "s" : ""}
          </div>
        )}
      </CardHeader>
      
      <CardContent className="overflow-scroll border-t border-t-gray-200 p-0 dark:border-t-slate-700">
        <ScrollArea className="h-[60vh] w-full">
          {filteredNodes.length === 0 ? (
            <div className="flex h-32 items-center justify-center text-sm text-gray-500 dark:text-gray-400">
              {searchQuery ? "No nodes found matching your search" : "Start typing to search nodes"}
            </div>
          ) : (
            filteredNodes.map((node, index) => (
              <Card
                key={node.id}
                className={`m-2 my-4 flex h-20 cursor-pointer shadow-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 ${
                  index === selectedIndex 
                    ? "shadow-lg dark:bg-slate-700" 
                    : "hover:shadow-lg dark:hover:bg-slate-700"
                }`}
                onClick={() => onNodeSelect(node.id)}
                onMouseEnter={() => {
                  setSelectedIndex(index);
                  onNodeHover?.(node.id);
                }}
                onMouseLeave={() => onNodeHover?.(null)}
              >
                <div
                  className={`-ml-px h-full w-3 rounded-l-xl ${getPrimaryCategoryColor(node.data.categories)}`}
                />
                <div className="mx-3 flex flex-1 items-center justify-between">
                  <div className="mr-2 min-w-0">
                    <span className="block truncate pb-1 text-sm font-semibold dark:text-white">
                      <TextRenderer
                        value={beautifyString(node.data.blockType).replace(/ Block$/, "")}
                        truncateLengthLimit={45}
                      />
                    </span>
                    <span className="block break-all text-xs font-normal text-gray-500 dark:text-gray-400">
                      <TextRenderer
                        value={getNodeInputOutputSummary(node) || node.data.description}
                        truncateLengthLimit={165}
                      />
                    </span>
                  </div>
                </div>
              </Card>
            ))
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
};