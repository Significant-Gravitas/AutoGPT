import React from "react";
import PaginatedBlocksContent from "./PaginatedBlocksContent";

const InputBlocksContent: React.FC = () => {
  return <PaginatedBlocksContent blockRequest={{ type: "input" }} />;
};

export default InputBlocksContent;
