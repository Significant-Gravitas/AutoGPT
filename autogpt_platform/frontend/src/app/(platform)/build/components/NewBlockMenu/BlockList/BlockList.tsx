import React from "react";
import { Block } from "../Block";
import { blockMenuContainerStyle } from "../style";
import { useNodeStore } from "../../../stores/nodeStore";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";

interface BlocksListProps {
  blocks: BlockInfo[];
  loading?: boolean;
}

export const BlocksList: React.FC<BlocksListProps> = ({
  blocks,
  loading = false,
}) => {
  const { addBlock } = useNodeStore();
  if (loading) {
    return (
      <div className={blockMenuContainerStyle}>
        {Array.from({ length: 7 }).map((_, index) => (
          <Block.Skeleton key={index} />
        ))}
      </div>
    );
  }
  return blocks.map((block) => (
    <Block
      key={block.id}
      title={block.name}
      description={block.description}
      onClick={() => addBlock(block)}
    />
  ));
};
