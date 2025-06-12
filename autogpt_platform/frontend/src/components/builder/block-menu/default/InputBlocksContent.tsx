import React from "react";
import { PaginatedBlocksContent } from "./PaginatedBlocksContent";

export const InputBlocksContent: React.FC = () => {
  return <PaginatedBlocksContent blockRequest={{ type: "input" }} />;
};
