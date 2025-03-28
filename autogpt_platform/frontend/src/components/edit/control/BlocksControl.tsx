import React, { useState, useMemo, useRef, useEffect } from "react";
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
  const [isOpen, setIsOpen] = useState(false);
  const [selectedBlockIndex, setSelectedBlockIndex] = useState<number>(-1);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const blockCardsRef = useRef<HTMLDivElement[]>([]);
  const [isInputFocused, setIsInputFocused] = useState(true);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check if Ctrl+B or Cmd+B was pressed
      if ((e.ctrlKey || e.metaKey) && e.key === "b") {
        e.preventDefault();
        setIsOpen(true);
        // Focus on the search input after a brief delay to ensure the popover is open
        setTimeout(() => {
          searchInputRef.current?.focus();
          setSelectedBlockIndex(-1);
          setIsInputFocused(true);
        }, 10);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  useEffect(() => {
    // Reset selected block index when search query changes
    setSelectedBlockIndex(-1);
    setIsInputFocused(true);
  }, [searchQuery]);

  useEffect(() => {
    // Focus on the search input when popover opens
    if (isOpen) {
      setTimeout(() => {
        searchInputRef.current?.focus();
        setIsInputFocused(true);
      }, 10);
    }
  }, [isOpen]);

  const handleInputFocus = () => {
    setIsInputFocused(true);
  };

  const handleInputBlur = () => {
    setIsInputFocused(false);
  };

  const handleKeyNavigation = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();

      if (isInputFocused) {
        // When Enter is pressed in search input, select the first block
        if (filteredAvailableBlocks.length > 0) {
          setSelectedBlockIndex(0);
          setIsInputFocused(false);
        }
      } else if (
        selectedBlockIndex >= 0 &&
        selectedBlockIndex < filteredAvailableBlocks.length
      ) {
        // When Enter is pressed on a selected block, add that block
        const block = filteredAvailableBlocks[selectedBlockIndex];
        if (!block.notAvailable) {
          addBlock(block.id, block.name, block?.hardcodedValues || {});
          setIsOpen(false);
        }
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (isInputFocused && filteredAvailableBlocks.length > 0) {
        // Move from input to first block
        setSelectedBlockIndex(0);
        setIsInputFocused(false);
      } else {
        // Navigate blocks downward
        setSelectedBlockIndex((prev) =>
          Math.min(prev + 1, filteredAvailableBlocks.length - 1),
        );
      }
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (selectedBlockIndex <= 0) {
        // Go back to input when at first block
        setSelectedBlockIndex(-1);
        setIsInputFocused(true);
        searchInputRef.current?.focus();
      } else {
        // Navigate blocks upward
        setSelectedBlockIndex((prev) => Math.max(prev - 1, 0));
      }
    }
  };

  useEffect(() => {
    // Scroll selected block into view
    if (selectedBlockIndex >= 0 && blockCardsRef.current[selectedBlockIndex]) {
      blockCardsRef.current[selectedBlockIndex].scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [selectedBlockIndex]);

  // Function to focus on the newly added block
  const handleAddBlock = (
    id: string,
    name: string,
    hardcodedValues: Record<string, any>,
  ) => {
    addBlock(id, name, hardcodedValues);
    setIsOpen(false);
  };

  const graphHasWebhookNodes = nodes.some(
    (n) => n.data.uiType == BlockUIType.WEBHOOK,
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

    return blockList
      .concat(agentBlockList)
      .filter(
        (block: Block) =>
          (block.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            beautifyString(block.name)
              .toLowerCase()
              .includes(searchQuery.toLowerCase()) ||
            block.description
              .toLowerCase()
              .includes(searchQuery.toLowerCase())) &&
          (!selectedCategory ||
            block.categories.some((cat) => cat.category === selectedCategory)),
      )
      .map((block) => ({
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
    setSelectedBlockIndex(-1);
    setIsInputFocused(true);
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
      open={pinBlocksPopover ? true : isOpen}
      onOpenChange={(open) => {
        setIsOpen(open);
        if (!open) resetFilters();
      }}
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
        onKeyDown={handleKeyNavigation}
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
                onFocus={handleInputFocus}
                onBlur={handleInputBlur}
                className="rounded-lg px-8 py-5 dark:bg-slate-800 dark:text-white"
                data-id="blocks-control-search-input"
                ref={searchInputRef}
                autoFocus
                autoComplete="off"
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
              {filteredAvailableBlocks.map((block, index) => (
                <Card
                  key={block.uiKey || block.id}
                  ref={(el) => {
                    if (el) blockCardsRef.current[index] = el;
                  }}
                  className={`m-2 my-4 flex h-20 shadow-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700 ${
                    block.notAvailable
                      ? "cursor-not-allowed opacity-50"
                      : "cursor-pointer hover:shadow-lg"
                  } ${selectedBlockIndex === index ? "border-2 border-blue-500 dark:border-blue-400" : ""}`}
                  data-id={`block-card-${block.id}`}
                  onClick={() => {
                    if (!block.notAvailable) {
                      handleAddBlock(
                        block.id,
                        block.name,
                        block?.hardcodedValues || {},
                      );
                    }
                  }}
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
