import { formatNodeDisplayTitle } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/helpers";
import { Separator } from "@/components/atoms/Separator/Separator";
import { ScrollArea } from "@/components/molecules/ScrollArea/ScrollArea";
import { beautifyString, cn } from "@/lib/utils";
import { SearchableNode } from "../GraphMenuSearchBar/useGraphMenuSearchBar";
import { GraphMenuSearchBar } from "../GraphMenuSearchBar/GraphMenuSearchBar";
import { getNodeInputOutputSummary } from "./helpers";
import { useGraphContent } from "./useGraphContent";
import { CornerUpRightIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

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

      <Separator
        // TODO(tokens): no fitting role for text-zinc-300 #C5C5C9; #C5C5C9 is foreground-disabled (disabled text) and border-strong (a border colour), and this is a static divider (QUALITY 41(c))
        className="h-[1px] w-full text-zinc-300"
      />

      <div className="flex-1 overflow-hidden">
        {trimmedQuery && (
          <Text
            variant="small-tight"
            as="div"
            tone="muted"
            className="px-4 pt-3"
          >
            Found {filteredNodes.length} node
            {filteredNodes.length !== 1 ? "s" : ""}
          </Text>
        )}
        <ScrollArea className="h-full w-full">
          <div role="listbox" className="space-y-3 px-4 py-4">
            {filteredNodes.length === 0 ? (
              <Text
                variant="body-tight"
                as="div"
                tone="muted"
                className="flex h-32 items-center justify-center"
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
                      "flex h-16 w-full cursor-pointer items-center gap-3 rounded-[0.75rem] bg-surface-muted px-[0.875rem] py-[0.625rem]",
                      index === selectedIndex
                        ? // TODO(tokens): no fitting role for bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this fill is not a hover state (QUALITY 41)
                          // TODO(tokens): no fitting role for ring-zinc-300 #C5C5C9; #C5C5C9 is border-strong (a border colour) and foreground-disabled (a text colour), and this is a ring (QUALITY 41(b))
                          "bg-zinc-100 ring-1 ring-zinc-300"
                        : "hover:bg-surface-hover",
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
                            variant="small-loose"
                            as="span"
                            tone="muted"
                            // TODO(tokens): no fitting role for bg-zinc-200 #DADADC; #DADADC is surface-active, an active-state fill, and border-base (a border colour), and this fill is static (QUALITY 41)
                            className="shrink-0 rounded-[0.75rem] bg-zinc-200 px-2"
                          >
                            {nodeType}
                          </Text>
                        )}
                      </div>
                      {description && (
                        <Text
                          variant="small-loose"
                          as="span"
                          tone="muted"
                          className="line-clamp-1"
                        >
                          {description}
                        </Text>
                      )}
                    </div>
                    <div
                      // TODO(tokens): no fitting role for bg-zinc-700 #505057; #505057 is foreground-body (a text colour)
                      className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[0.5rem] bg-zinc-700"
                    >
                      <Icon
                        icon={CornerUpRightIcon}
                        // TODO(tokens): no fitting role for text-zinc-50 #F9F9FA; #F9F9FA is surface-muted (a surface fill)
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
