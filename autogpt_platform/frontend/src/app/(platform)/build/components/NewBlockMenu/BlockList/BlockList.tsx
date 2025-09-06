import React from "react";
import { Block } from "../Block";

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
            />
          ))}
    </div>
  );
};