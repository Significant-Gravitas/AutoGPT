import React from "react";
import { PaginatedBlocksContent } from "./PaginatedBlocksContent";

export const ActionBlocksContent: React.FC = () => {
  return <PaginatedBlocksContent blockRequest={{ type: "action" }} />;
};
