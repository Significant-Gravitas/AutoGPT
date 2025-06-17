import React from "react";
import { PaginatedBlocksContent } from "./PaginatedBlocksContent";

export const OutputBlocksContent = () => {
  return <PaginatedBlocksContent blockRequest={{ type: "output" }} />;
};
