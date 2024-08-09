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
          <CardHeader className="flex px-2 flex-col p-3 gap-x-8 gap-y-2">
            <div className="justify-between items-center ">
              <Label
                htmlFor="search-blocks"
                className="text-base 2xl:text-xl font-semibold whitespace-nowrap text-black border-b-2 border-violet-500"
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
                <Card key={block.id} className="m-2">
                  <div className="flex items-center justify-between m-3">
                    <div className="flex-1 min-w-0 mr-2">
                      <span className="font-medium truncate block">
                        {beautifyString(block.name)}
                      </span>
                    </div>
                    <div className="flex items-center gap-1 flex-shrink-0">
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
