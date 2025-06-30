import React, { useState, useMemo } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { TextRenderer } from "@/components/ui/render";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CustomNode } from "@/components/CustomNode";
import { beautifyString } from "@/lib/utils";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Block, BlockUIType, SpecialBlockID } from "@/lib/autogpt-server-api";
import { MagnifyingGlassIcon, PlusIcon } from "@radix-ui/react-icons";
import { IconToyBrick } from "@/components/ui/icons";
import { getPrimaryCategoryColor } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { GraphMeta } from "@/lib/autogpt-server-api";
import jaro from "jaro-winkler";

interface BlocksControlProps {
  blocks: Block[];
  addBlock: (
    id: string,
    name: string,
    hardcodedValues: Record<string, any>,
  ) => void;
  pinBlocksPopover: boolean;
  flows: GraphMeta[];
  nodes: CustomNode[];
}

/**
 * A React functional component that displays a control for managing blocks.
 *
 * @component
 * @param {Object} BlocksControlProps - The properties for the BlocksControl component.
 * @param {Block[]} BlocksControlProps.blocks - An array of blocks to be displayed and filtered.
 * @param {(id: string, name: string) => void} BlocksControlProps.addBlock - A function to call when a block is added.
 * @returns The rendered BlocksControl component.
 */
export const BlocksControl: React.FC<BlocksControlProps> = ({
  blocks,
  addBlock,
  pinBlocksPopover,
  flows,
  nodes,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const graphHasWebhookNodes = nodes.some((n) =>
    [BlockUIType.WEBHOOK, BlockUIType.WEBHOOK_MANUAL].includes(n.data.uiType),
  );
  const graphHasInputNodes = nodes.some(
    (n) => n.data.uiType == BlockUIType.INPUT,
  );

  const filteredAvailableBlocks = useMemo(() => {
    const blockList = blocks
      .filter((b) => b.uiType !== BlockUIType.AGENT)
      .sort((a, b) => a.name.localeCompare(b.name));
    const agentBlockList = flows.map(
      (flow) =>
        ({
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
          uiKey: flow.id,
          costs: [],
          hardcodedValues: {
            graph_id: flow.id,
            graph_version: flow.version,
            input_schema: flow.input_schema,
            output_schema: flow.output_schema,
          },
        }) satisfies Block,
    );

    /**
     * Evaluates how well a block matches the search query and returns a relevance score.
     * The scoring algorithm works as follows:
     * - Returns 1 if no query (all blocks match equally)
     * - Normalized query for case-insensitive matching
     * - Returns 3 for exact substring matches in block name (highest priority)
     * - Returns 2 when all query words appear in the block name (regardless of order)
     * - Returns 1.X for blocks with names similar to query using Jaro-Winkler distance (X is similarity score)
     * - Returns 0.5 when all query words appear in the block description (lowest priority)
     * - Returns 0 for no match
     *
     * Higher scores will appear first in search results.
     */
    const matchesSearch = (block: Block, query: string): number => {
      if (!query) return 1;
      const normalizedQuery = query.toLowerCase().trim();
      const queryWords = normalizedQuery.split(/\s+/);
      const blockName = block.name.toLowerCase();
      const beautifiedName = beautifyString(block.name).toLowerCase();
      const description = block.description.toLowerCase();

      // 1. Exact match in name (highest priority)
      if (
        blockName.includes(normalizedQuery) ||
        beautifiedName.includes(normalizedQuery)
      ) {
        return 3;
      }

      // 2. All query words in name (regardless of order)
      const allWordsInName = queryWords.every(
        (word) => blockName.includes(word) || beautifiedName.includes(word),
      );
      if (allWordsInName) return 2;

      // 3. Similarity with name (Jaro-Winkler)
      const similarityThreshold = 0.65;
      const nameSimilarity = jaro(blockName, normalizedQuery);
      const beautifiedSimilarity = jaro(beautifiedName, normalizedQuery);
      const maxSimilarity = Math.max(nameSimilarity, beautifiedSimilarity);
      if (maxSimilarity > similarityThreshold) {
        return 1 + maxSimilarity; // Score between 1 and 2
      }

      // 4. All query words in description (lower priority)
      const allWordsInDescription = queryWords.every((word) =>
        description.includes(word),
      );
      if (allWordsInDescription) return 0.5;

      return 0;
    };

    return blockList
      .concat(agentBlockList)
      .map((block) => ({
        block,
        score: matchesSearch(block, searchQuery),
      }))
      .filter(
        ({ block, score }) =>
          score > 0 &&
          (!selectedCategory ||
            block.categories.some((cat) => cat.category === selectedCategory)),
      )
      .sort((a, b) => b.score - a.score)
      .map(({ block }) => ({
        ...block,
        notAvailable:
          (block.uiType == BlockUIType.WEBHOOK &&
            graphHasWebhookNodes &&
            "Agents can only have one webhook-triggered block") ||
          (block.uiType == BlockUIType.WEBHOOK &&
            graphHasInputNodes &&
            "Webhook-triggered blocks can't be used together with input blocks") ||
          (block.uiType == BlockUIType.INPUT &&
            graphHasWebhookNodes &&
            "Input blocks can't be used together with a webhook-triggered block") ||
          null,
      }));
  }, [
    blocks,
    flows,
    searchQuery,
    selectedCategory,
    graphHasInputNodes,
    graphHasWebhookNodes,
  ]);

  const resetFilters = React.useCallback(() => {
    setSearchQuery("");
    setSelectedCategory(null);
  }, []);

  // Extract unique categories from blocks
  const categories = Array.from(
    new Set([
      null,
      ...blocks
        .flatMap((block) => block.categories.map((cat) => cat.category))
        .sort(),
    ]),
  );

  return (
    <Popover
      open={pinBlocksPopover ? true : undefined}
      onOpenChange={(open) => open || resetFilters()}
    >
      <Tooltip delayDuration={500}>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              data-id="blocks-control-popover-trigger"
              data-testid="blocks-control-blocks-button"
              name="Blocks"
              className="dark:hover:bg-slate-800"
            >
              <IconToyBrick />
            </Button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="right">Blocks</TooltipContent>
      </Tooltip>
      <PopoverContent
        side="right"
        sideOffset={22}
        align="start"
        className="absolute -top-3 w-[17rem] rounded-xl border-none p-0 shadow-none md:w-[30rem]"
        data-id="blocks-control-popover-content"
      >
        <Card className="p-3 pb-0 dark:bg-slate-900">
          <CardHeader className="flex flex-col gap-x-8 gap-y-1 p-3 px-2">
            <div className="items-center justify-between">
              <Label
                htmlFor="search-blocks"
                className="whitespace-nowrap text-base font-bold text-black dark:text-white 2xl:text-xl"
                data-id="blocks-control-label"
                data-testid="blocks-control-blocks-label"
              >
                Blocks
              </Label>
            </div>
            <div className="relative flex items-center">
              <MagnifyingGlassIcon className="absolute m-2 h-5 w-5 text-gray-500 dark:text-gray-400" />
              <Input
                id="search-blocks"
                type="text"
                placeholder="Search blocks"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="rounded-lg px-8 py-5 dark:bg-slate-800 dark:text-white"
                data-id="blocks-control-search-input"
              />
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {categories.map((category) => {
                const color = getPrimaryCategoryColor([
                  { category: category || "All", description: "" },
                ]);
                const colorClass =
                  selectedCategory === category ? `${color}` : "";
                return (
                  <div
                    key={category}
                    className={`cursor-pointer rounded-xl border px-2 py-2 text-xs font-medium dark:border-slate-700 dark:text-white ${colorClass}`}
                    onClick={() =>
                      setSelectedCategory(
                        selectedCategory === category ? null : category,
                      )
                    }
                  >
                    {beautifyString((category || "All").toLowerCase())}
                  </div>
                );
              })}
            </div>
          </CardHeader>
          <CardContent className="overflow-scroll border-t border-t-gray-200 p-0 dark:border-t-slate-700">
            <ScrollArea
              className="h-[60vh] w-full"
              data-id="blocks-control-scroll-area"
            >
              {filteredAvailableBlocks.map((block) => (
                <Card
                  key={block.uiKey || block.id}
                  className={`m-2 my-4 flex h-20 shadow-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700 ${
                    block.notAvailable
                      ? "cursor-not-allowed opacity-50"
                      : "cursor-pointer hover:shadow-lg"
                  }`}
                  data-id={`block-card-${block.id}`}
                  onClick={() =>
                    !block.notAvailable &&
                    addBlock(block.id, block.name, block?.hardcodedValues || {})
                  }
                  title={block.notAvailable ?? undefined}
                >
                  <div
                    className={`-ml-px h-full w-3 rounded-l-xl ${getPrimaryCategoryColor(block.categories)}`}
                  ></div>

                  <div className="mx-3 flex flex-1 items-center justify-between">
                    <div className="mr-2 min-w-0">
                      <span
                        className="block truncate pb-1 text-sm font-semibold dark:text-white"
                        data-id={`block-name-${block.id}`}
                        data-type={block.uiType}
                        data-testid={`block-name-${block.id}`}
                      >
                        <TextRenderer
                          value={beautifyString(block.name).replace(
                            / Block$/,
                            "",
                          )}
                          truncateLengthLimit={45}
                        />
                      </span>
                      <span
                        className="block break-all text-xs font-normal text-gray-500 dark:text-gray-400"
                        data-testid={`block-description-${block.id}`}
                      >
                        <TextRenderer
                          value={block.description}
                          truncateLengthLimit={165}
                        />
                      </span>
                    </div>
                    <div
                      className="flex flex-shrink-0 items-center gap-1"
                      data-id={`block-tooltip-${block.id}`}
                      data-testid={`block-add`}
                    >
                      <PlusIcon className="h-6 w-6 rounded-lg bg-gray-200 stroke-black stroke-[0.5px] p-1 dark:bg-gray-700 dark:stroke-white" />
                    </div>
                  </div>
                </Card>
              ))}
            </ScrollArea>
          </CardContent>
        </Card>
      </PopoverContent>
    </Popover>
  );
};
