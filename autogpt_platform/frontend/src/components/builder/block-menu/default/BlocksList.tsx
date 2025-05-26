import React from "react";
import Block from "../Block";
import { Block as BlockType } from "@/lib/autogpt-server-api";
import { useBlockMenuContext } from "../block-menu-provider";

interface BlocksListProps {
  blocks: BlockType[];
  loading?: boolean;
}

const BlocksList: React.FC<BlocksListProps> = ({ blocks, loading = false }) => {
  const { addNode } = useBlockMenuContext();
  return (
    <div className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
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
                  addNode(
                    block.id,
                    block.name,
                    block.hardcodedValues || {},
                    block,
                  );
                }}
              />
            ))}
      </div>
    </div>
  );
};

export default BlocksList;
