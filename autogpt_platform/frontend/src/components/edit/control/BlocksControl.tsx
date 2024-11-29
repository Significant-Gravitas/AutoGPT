import React, { useState, useCallback } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { TextRenderer } from "@/components/ui/render";
import { ScrollArea } from "@/components/ui/scroll-area";
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
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const getFilteredBlockList = (): Block[] => {
    const blockList = blocks
      .filter((b) => b.uiType !== BlockUIType.AGENT)
      .sort((a, b) => a.name.localeCompare(b.name));
    const agentList = flows.map(
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
        }) as Block,
    );

    return blockList
      .concat(agentList)
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
      );
  };

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
        <Card className="p-3 pb-0">
          <CardHeader className="flex flex-col gap-x-8 gap-y-1 p-3 px-2">
            <div className="items-center justify-between">
              <Label
                htmlFor="search-blocks"
                className="whitespace-nowrap text-base font-bold text-black 2xl:text-xl"
                data-id="blocks-control-label"
              >
                Blocks
              </Label>
            </div>
            <div className="relative flex items-center">
              <MagnifyingGlassIcon className="absolute m-2 h-5 w-5 text-gray-500" />
              <Input
                id="search-blocks"
                type="text"
                placeholder="Search blocks"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="rounded-lg px-8 py-5"
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
                    className={`cursor-pointer rounded-xl border px-2 py-2 text-xs font-medium ${colorClass}`}
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
          <CardContent className="overflow-scroll border-t p-0">
            <ScrollArea
              className="h-[60vh]"
              data-id="blocks-control-scroll-area"
            >
              {getFilteredBlockList().map((block) => (
                <Card
                  key={block.uiKey || block.id}
                  className="m-2 my-4 flex h-20 cursor-pointer shadow-none hover:shadow-lg"
                  data-id={`block-card-${block.id}`}
                  onClick={() =>
                    addBlock(block.id, block.name, block?.hardcodedValues || {})
                  }
                >
                  <div
                    className={`-ml-px h-full w-3 rounded-l-xl ${getPrimaryCategoryColor(block.categories)}`}
                  ></div>

                  <div className="mx-3 flex flex-1 items-center justify-between">
                    <div className="mr-2 min-w-0">
                      <span
                        className="block truncate pb-1 text-sm font-semibold"
                        data-id={`block-name-${block.id}`}
                      >
                        <TextRenderer
                          value={beautifyString(block.name).replace(
                            / Block$/,
                            "",
                          )}
                          truncateLengthLimit={45}
                        />
                      </span>
                      <span className="block break-all text-xs font-normal text-gray-500">
                        <TextRenderer
                          value={block.description}
                          truncateLengthLimit={165}
                        />
                      </span>
                    </div>
                    <div
                      className="flex flex-shrink-0 items-center gap-1"
                      data-id={`block-tooltip-${block.id}`}
                    >
                      <PlusIcon className="h-6 w-6 rounded-lg bg-gray-200 stroke-black stroke-[0.5px] p-1" />
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
