import { formatNodeDisplayTitle } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/helpers";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { beautifyString, cn } from "@/lib/utils";
import { SearchableNode } from "../GraphMenuSearchBar/useGraphMenuSearchBar";
import { ArrowBendUpRight } from "@phosphor-icons/react";
import { GraphMenuSearchBar } from "../GraphMenuSearchBar/GraphMenuSearchBar";
import { getNodeInputOutputSummary } from "./helpers";
import { useGraphContent } from "./useGraphContent";

interface Props {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  filteredNodes: SearchableNode[];
  onNodeSelect: (nodeID: string) => void;
}

export function GraphSearchContent({
  searchQuery,
  onSearchChange,
  filteredNodes,
  onNodeSelect,
}: Props) {
  const { selectedIndex, setSelectedIndex, setItemRef, handleKeyDown } =
    useGraphContent({
      searchQuery,
      filteredNodes,
      onNodeSelect,
    });

  const trimmedQuery = searchQuery?.trim();

  return (
    <div className="flex h-full w-full flex-col">
      <GraphMenuSearchBar
        searchQuery={searchQuery}
        onSearchChange={onSearchChange}
        onKeyDown={handleKeyDown}
      />

      <Separator className="h-[1px] w-full text-zinc-300" />

      <div className="flex-1 overflow-hidden">
        {trimmedQuery && (
          <div className="px-4 pt-3 text-xs text-zinc-500">
            Found {filteredNodes.length} node
            {filteredNodes.length !== 1 ? "s" : ""}
          </div>
        )}
        <ScrollArea className="h-full w-full">
          <div role="listbox" className="space-y-3 px-4 py-4">
            {filteredNodes.length === 0 ? (
              <div className="flex h-32 items-center justify-center text-sm text-zinc-500">
                {trimmedQuery
                  ? "No nodes found matching your search"
                  : "Start typing to search nodes"}
              </div>
            ) : (
              filteredNodes.map((node, index) => {
                if (!node?.data) return null;

                const nodeTitle = formatNodeDisplayTitle(node.data);
                const nodeType = beautifyString(node.data.title || "").replace(
                  / Block$/,
                  "",
                );
                const description =
                  getNodeInputOutputSummary(node) ||
                  node.data.description ||
                  "";

                const hasCustomName = !!(
                  node.data.metadata?.customized_name ||
                  node.data.hardcodedValues?.agent_name
                );

                return (
                  <div
                    key={node.id}
                    ref={(el) => setItemRef(index, el)}
                    role="option"
                    aria-selected={index === selectedIndex}
                    tabIndex={index === selectedIndex ? 0 : -1}
                    className={cn(
                      "flex h-16 w-full cursor-pointer items-center gap-3 rounded-[0.75rem] bg-zinc-50 px-[0.875rem] py-[0.625rem]",
                      index === selectedIndex
                        ? "bg-zinc-100 ring-1 ring-zinc-300"
                        : "hover:bg-zinc-100",
                    )}
                    onClick={() => onNodeSelect(node.id)}
                    onMouseEnter={() => setSelectedIndex(index)}
                  >
                    <div className="flex flex-1 flex-col items-start gap-0.5 overflow-hidden">
                      <div className="flex items-center gap-2">
                        <span className="line-clamp-1 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
                          {nodeTitle}
                        </span>
                        {hasCustomName && (
                          <span className="shrink-0 rounded-[0.75rem] bg-zinc-200 px-2 font-sans text-xs leading-5 text-zinc-500">
                            {nodeType}
                          </span>
                        )}
                      </div>
                      {description && (
                        <span className="line-clamp-1 font-sans text-xs font-normal leading-5 text-zinc-500">
                          {description}
                        </span>
                      )}
                    </div>
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[0.5rem] bg-zinc-700">
                      <ArrowBendUpRight className="h-4 w-4 text-zinc-50" />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
