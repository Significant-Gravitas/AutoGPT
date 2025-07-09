import React from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { TextRenderer } from "@/components/ui/render";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { MagnifyingGlassIcon, PlusIcon } from "@radix-ui/react-icons";
import { IconToyBrick } from "@/components/ui/icons";
import { getPrimaryCategoryColor, beautifyString } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Block, GraphMeta } from "@/lib/autogpt-server-api";
import { CustomNode } from "@/components/CustomNode";
import { useBlocksControl } from "./useBlocksControl";

interface Props {
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
 * Optimized for performance with debounced search, memoized data, and separated concerns.
 */
function BlocksControlComponent({
  blocks,
  addBlock,
  pinBlocksPopover,
  flows,
  nodes,
}: Props) {
  const {
    searchQuery,
    setSearchQuery,
    selectedCategory,
    filteredAvailableBlocks,
    categories,
    resetFilters,
    handleCategoryClick,
    handleAddBlock,
  } = useBlocksControl({
    blocks,
    flows,
    nodes,
    addBlock,
  });

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
                    onClick={() => handleCategoryClick(category)}
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
                  onClick={() => handleAddBlock(block)}
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
}

BlocksControlComponent.displayName = "BlocksControl";
export const BlocksControl = React.memo(BlocksControlComponent);
