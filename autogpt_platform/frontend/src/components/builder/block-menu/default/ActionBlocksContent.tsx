import React from "react";
import PaginatedBlocksContent from "./PaginatedBlocksContent";

const ActionBlocksContent: React.FC = () => {
  return <PaginatedBlocksContent blockRequest={{ type: "action" }} />;
};

export default ActionBlocksContent;
