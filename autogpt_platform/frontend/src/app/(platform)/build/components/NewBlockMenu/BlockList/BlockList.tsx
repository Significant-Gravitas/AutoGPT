import React from "react";
import { Block } from "../Block";
import { blockMenuContainerStyle } from "../style";

export interface BlockType {
  id: string;
  name: string;
  description: string;
  category?: string;
  type?: string;
  provider?: string;
}

interface BlocksListProps {
  blocks: BlockType[];
  loading?: boolean;
}

export const BlocksList: React.FC<BlocksListProps> = ({
  blocks,
  loading = false,
}) => {
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
    <Block key={block.id} title={block.name} description={block.description} />
  ));
};
