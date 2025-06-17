import React from "react";
import { PaginatedBlocksContent } from "./PaginatedBlocksContent";

export const ActionBlocksContent = () => {
  return <PaginatedBlocksContent blockRequest={{ type: "action" }} />;
};
