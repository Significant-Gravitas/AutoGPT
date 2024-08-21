import React, { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { ToyBrick } from "lucide-react";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { beautifyString } from "@/lib/utils";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Block } from "@/lib/autogpt-server-api";
import { PlusIcon } from "@radix-ui/react-icons";
import { IconToyBrick } from "@/components/ui/icons";
import SchemaTooltip from "@/components/SchemaTooltip";
import { getPrimaryCategoryColor } from "@/lib/utils";
interface BlocksControlProps {
  blocks: Block[];
  addBlock: (id: string, name: string) => void;
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
}) => {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredBlocks = blocks.filter((block: Block) =>
    block.name.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon">
          <IconToyBrick />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="right"
        sideOffset={22}
        align="start"
        className="w-96 p-0"
      >
        <Card className="border-none shadow-md">
          <CardHeader className="flex flex-col gap-x-8 gap-y-2 p-3 px-2">
            <div className="items-center justify-between">
              <Label
                htmlFor="search-blocks"
                className="whitespace-nowrap border-b-2 border-violet-500 text-base font-semibold text-black 2xl:text-xl"
              >
                Blocks
              </Label>
            </div>
            <Input
              id="search-blocks"
              type="text"
              placeholder="Search blocks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </CardHeader>
          <CardContent className="p-1">
            <ScrollArea className="h-[60vh]">
              {filteredBlocks.map((block) => (
                <Card
                  key={block.id}
                  className={`m-2 ${getPrimaryCategoryColor(block.categories)}`}
                >
                  <div className="m-3 flex items-center justify-between">
                    <div className="mr-2 min-w-0 flex-1">
                      <span className="block truncate font-medium">
                        {beautifyString(block.name)}
                      </span>
                    </div>
                    <SchemaTooltip description={block.description} />
                    <div className="flex flex-shrink-0 items-center gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => addBlock(block.id, block.name)}
                        aria-label="Add block"
                      >
                        <PlusIcon />
                      </Button>
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
