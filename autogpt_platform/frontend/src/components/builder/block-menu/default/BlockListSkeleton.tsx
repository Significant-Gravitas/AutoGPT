import React from "react";
import Block from "../Block";

interface BlockListSkeletonProps {
  count?: number;
}

const BlockListSkeleton: React.FC<BlockListSkeletonProps> = ({ count = 3 }) => {
  return (
    <>
      {Array.from({ length: count }).map((_, index) => (
        <Block.Skeleton key={index} />
      ))}
    </>
  );
};

export default BlockListSkeleton;
