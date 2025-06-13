import React from "react";
import { PaginatedBlocksContent } from "./PaginatedBlocksContent";

export const OutputBlocksContent: React.FC = () => {
  return <PaginatedBlocksContent blockRequest={{ type: "output" }} />;
};
