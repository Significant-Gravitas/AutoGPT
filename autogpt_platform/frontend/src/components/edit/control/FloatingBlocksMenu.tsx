import React, {
  useCallback,
  useMemo,
  useState,
  useEffect,
  useRef,
} from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { beautifyString } from "@/lib/utils";
import { Block, BlockUIType, SpecialBlockID } from "@/lib/autogpt-server-api";
import { MagnifyingGlassIcon, PlusIcon } from "@radix-ui/react-icons";
import { getPrimaryCategoryColor } from "@/lib/utils";
import { GraphMeta } from "@/lib/autogpt-server-api";
import jaro from "jaro-winkler";
import { CustomNode } from "@/components/CustomNode";
import { filterBlocksByConnectionType } from "@/lib/utils/connectionUtils";

type _Block = Block & {
  uiKey?: string; // Keep uiKey for agent blocks
  hardcodedValues?: Record<string, any>;
  _cached?: {
    blockName: string;
    beautifiedName: string;
    description: string;
  };
};

interface FloatingBlocksMenuProps {
  blocks: _Block[];
  position: { x: number; y: number };
  onSelectBlock: (
    id: string,
    name: string,
    hardcodedValues: Record<string, any>,
  ) => void;
  onClose: () => void;
  flows: GraphMeta[];
  nodes: CustomNode[];
  connectionType: "source" | "target";
  handleType?: string;
  sourceNodeId?: string;
  sourceHandle?: string;
}

export function FloatingBlocksMenu({
  blocks: _blocks,
  position,
  onSelectBlock,
  onClose,
  flows,
  _nodes,
  connectionType,
  handleType,
  _sourceNodeId,
  sourceHandle,
}: FloatingBlocksMenuProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const menuRef = useRef<HTMLDivElement>(null);

  // Filter blocks based on connection compatibility
  const compatibleBlocks = useMemo(() => {
    const dragDirection = connectionType === "source" ? "output" : "input";
    const filtered = filterBlocksByConnectionType(
      _blocks.filter((b) => b.uiType !== BlockUIType.AGENT),
      dragDirection,
      handleType,
      sourceHandle,
    );

    return filtered;
  }, [_blocks, connectionType, handleType, sourceHandle]);

  // Process agent blocks
  const agentBlockList = useMemo(() => {
    return flows
      .map((flow) => {
        const block: _Block = {
          id: SpecialBlockID.AGENT,
          name: flow.name,
          description:
            `Ver.${flow.version}` +
            (flow.description ? ` | ${flow.description}` : ""),
          categories: [{ category: "AGENT", description: "" }],
          inputSchema: flow.input_schema,
          outputSchema: flow.output_schema,
          staticOutput: false,
          uiType: BlockUIType.AGENT,
          costs: [],
          uiKey: flow.id, // Add uiKey for agent blocks
          hardcodedValues: {
            graph_id: flow.id,
            graph_version: flow.version,
            input_schema: flow.input_schema,
            output_schema: flow.output_schema,
          },
          _cached: {
            blockName: flow.name.toLowerCase(),
            beautifiedName: beautifyString(flow.name).toLowerCase(),
            description: (flow.description || "").toLowerCase(),
          },
        };
        return block;
      })
      .filter((agentBlock) => {
        // Filter agent blocks by compatibility
        const dragDirection = connectionType === "source" ? "output" : "input";
        const filtered = filterBlocksByConnectionType(
          [agentBlock],
          dragDirection,
          handleType,
          sourceHandle,
        );
        return filtered.length > 0;
      });
  }, [flows, connectionType, handleType, sourceHandle]);

  // Combine and filter blocks
  const filteredBlocks = useMemo(() => {
    const allBlocks = [...compatibleBlocks, ...agentBlockList];

    if (!searchQuery) {
      return allBlocks.sort((a, b) => a.name.localeCompare(b.name));
    }

    const normalizedQuery = searchQuery.toLowerCase().trim();

    return allBlocks
      .map((block) => ({
        block,
        score: blockScoreForQuery(block, normalizedQuery),
      }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score)
      .map(({ block }) => block);
  }, [compatibleBlocks, agentBlockList, searchQuery]);

  // Close menu on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    // Small delay to ensure the menu is fully rendered
    const timer = setTimeout(() => {
      // Use capture phase to catch events before React Flow
      document.addEventListener("mousedown", handleClickOutside, true);
      document.addEventListener("click", handleClickOutside, true);
    }, 100);

    return () => {
      clearTimeout(timer);
      document.removeEventListener("mousedown", handleClickOutside, true);
      document.removeEventListener("click", handleClickOutside, true);
    };
  }, [onClose]);

  // Close on Escape key
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [onClose]);

  const handleBlockSelect = useCallback(
    (block: _Block) => {
      // Call onSelectBlock first, then close
      // This ensures the parent knows a selection was made
      onSelectBlock(block.id, block.name, block.hardcodedValues || {});
      // Now call onClose to trigger cleanup
      onClose();
    },
    [onSelectBlock, onClose],
  );

  return (
    <div
      ref={menuRef}
      className="fixed z-[9999]"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        transform: "translate(-50%, -50%)",
      }}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
    >
      <Card className="w-[20rem] p-3 pb-0 shadow-lg dark:bg-slate-900">
        <CardHeader className="flex flex-col gap-x-8 gap-y-1 p-3 px-2">
          <Label className="whitespace-nowrap text-base font-bold text-black dark:text-white">
            Add Block
          </Label>
          <div className="relative flex items-center">
            <MagnifyingGlassIcon className="absolute m-2 h-5 w-5 text-gray-500 dark:text-gray-400" />
            <Input
              type="text"
              placeholder="Search compatible blocks"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="rounded-lg px-8 py-5 dark:bg-slate-800 dark:text-white"
              autoComplete="off"
              autoFocus
            />
          </div>
        </CardHeader>
        <CardContent className="overflow-scroll border-t border-t-gray-200 p-0 dark:border-t-slate-700">
          <ScrollArea className="h-[40vh] w-full">
            {filteredBlocks.length === 0 ? (
              <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                No compatible blocks found
              </div>
            ) : (
              filteredBlocks.map((block, index) => (
                <Card
                  key={`${block.id}-${index}`}
                  className="group m-2 my-4 flex h-20 cursor-pointer overflow-hidden shadow-none hover:shadow-md dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700"
                  onClick={() => handleBlockSelect(block)}
                  title={`${beautifyString(block.name).replace(/ Block$/, "")}\n${block.description}`}
                >
                  <div
                    className={`-ml-px h-full w-3 flex-shrink-0 rounded-l-xl ${getPrimaryCategoryColor(block.categories)}`}
                  />
                  <div className="mx-3 flex min-w-0 flex-1 items-center justify-between overflow-hidden">
                    <div className="mr-2 min-w-0 overflow-hidden">
                      <div className="truncate pb-1 text-sm font-semibold dark:text-white">
                        {beautifyString(block.name).replace(/ Block$/, "")}
                      </div>
                      <div className="line-clamp-2 text-xs font-normal text-gray-500 dark:text-gray-400">
                        {block.description}
                      </div>
                    </div>
                    <div className="flex flex-shrink-0 items-center gap-1">
                      <PlusIcon className="h-6 w-6 rounded-lg bg-gray-200 stroke-black stroke-[0.5px] p-1 dark:bg-gray-700 dark:stroke-white" />
                    </div>
                  </div>
                </Card>
              ))
            )}
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

function blockScoreForQuery(block: _Block, query: string): number {
  if (!query) return 1;
  const normalizedQuery = query.toLowerCase().trim();
  const queryWords = normalizedQuery.split(/\s+/);

  const { blockName, beautifiedName, description } = block._cached!;

  // Exact match in name
  if (
    blockName.includes(normalizedQuery) ||
    beautifiedName.includes(normalizedQuery)
  ) {
    return 3;
  }

  // All query words in name
  const allWordsInName = queryWords.every(
    (word) => blockName.includes(word) || beautifiedName.includes(word),
  );
  if (allWordsInName) return 2;

  // Similarity with name
  const similarityThreshold = 0.65;
  const nameSimilarity = jaro(blockName, normalizedQuery);
  const beautifiedSimilarity = jaro(beautifiedName, normalizedQuery);
  const maxSimilarity = Math.max(nameSimilarity, beautifiedSimilarity);
  if (maxSimilarity > similarityThreshold) {
    return 1 + maxSimilarity;
  }

  // All query words in description
  const allWordsInDescription = queryWords.every((word) =>
    description.includes(word),
  );
  if (allWordsInDescription) return 0.5;

  return 0;
}
