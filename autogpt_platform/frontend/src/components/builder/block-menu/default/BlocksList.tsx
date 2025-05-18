import React from "react";
import Block from "../Block";
import { BlockListType } from "./BlockMenuDefaultContent";

interface BlocksListProps {
  blocks: BlockListType[];
  loading?: boolean;
}

const BlocksList: React.FC<BlocksListProps> = ({ blocks, loading = false }) => {
  return (
    <div className="scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-zinc-200 scrollbar-track-transparent h-full overflow-y-scroll pt-4">
      <div className="w-full space-y-3 px-4 pb-4">
        {loading
          ? Array.from({ length: 7 }).map((_, index) => (
              <Block.Skeleton key={index} />
            ))
          : blocks.map((block) => (
              <Block
                key={block.id}
                title={block.title}
                description={block.description}
              />
            ))}
      </div>
    </div>
  );
};

export default BlocksList;
