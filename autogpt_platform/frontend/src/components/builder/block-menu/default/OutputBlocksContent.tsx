import React from "react";
import PaginatedBlocksContent from "./PaginatedBlocksContent";

const OutputBlocksContent: React.FC = () => {
  return <PaginatedBlocksContent blockRequest={{ type: "output" }} />;
};

export default OutputBlocksContent;