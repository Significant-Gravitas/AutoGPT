import { Text } from "@/components/atoms/Text/Text";
import { formatNodeDisplayTitle } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/helpers";
import { Separator } from "@/components/atoms/Separator/Separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { beautifyString, cn } from "@/lib/utils";
import { SearchableNode } from "../GraphMenuSearchBar/useGraphMenuSearchBar";
import { GraphMenuSearchBar } from "../GraphMenuSearchBar/GraphMenuSearchBar";
import { getNodeInputOutputSummary } from "./helpers";
import { useGraphContent } from "./useGraphContent";
import { CornerUpRightIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
          <Text
            variant="small"
            as="div"
            tone="muted"
            className="px-4 pt-3 leading-4"
          >
            Found {filteredNodes.length} node
            {filteredNodes.length !== 1 ? "s" : ""}
          </Text>
        )}
        <ScrollArea className="h-full w-full">
          <div role="listbox" className="space-y-3 px-4 py-4">
            {filteredNodes.length === 0 ? (
              <Text
                variant="body"
                as="div"
                tone="muted"
                className="flex h-32 items-center justify-center leading-5"
              >
                {trimmedQuery
                  ? "No nodes found matching your search"
                  : "Start typing to search nodes"}
              </Text>
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
                        <Text
                          variant="body-medium"
                          as="span"
                          tone="emphasis"
                          className="line-clamp-1"
                        >
                          {nodeTitle}
                        </Text>
                        {hasCustomName && (
                          <Text
                            variant="small"
                            as="span"
                            tone="muted"
                            className="shrink-0 rounded-[0.75rem] bg-zinc-200 px-2 leading-5"
                          >
                            {nodeType}
                          </Text>
                        )}
                      </div>
                      {description && (
                        <Text
                          variant="small"
                          as="span"
                          tone="muted"
                          className="line-clamp-1 leading-5"
                        >
                          {description}
                        </Text>
                      )}
                    </div>
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[0.5rem] bg-zinc-700">
                      <Icon
                        icon={CornerUpRightIcon}
                        className="h-4 w-4 text-zinc-50"
                      />
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
