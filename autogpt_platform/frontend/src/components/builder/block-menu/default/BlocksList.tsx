import React from "react";
import { Block } from "../Block";
import { Block as BlockType } from "@/lib/autogpt-server-api";
import { useBlockMenuContext } from "../block-menu-provider";

interface BlocksListProps {
  blocks: BlockType[];
  loading?: boolean;
}

export const BlocksList: React.FC<BlocksListProps> = ({ blocks, loading = false }) => {
  const { addNode } = useBlockMenuContext();
  return (
    <div className="w-full space-y-3 px-4 pb-4">
      {loading
        ? Array.from({ length: 7 }).map((_, index) => (
            <Block.Skeleton key={index} />
          ))
        : blocks.map((block) => (
            <Block
              key={block.id}
              title={block.name}
              description={block.description}
              onClick={() => {
                addNode(block);
              }}
            />
          ))}
    </div>
  );
};


