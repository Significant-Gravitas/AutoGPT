import React, { useEffect, useState } from "react";
import BlocksList from "./BlocksList";
import { BlockListType } from "./BlockMenuDefaultContent";
import { outputBlocksListData } from "../../testing_data";

const OutputBlocksContent: React.FC = () => {
  const [blocks, setBlocks] = useState<BlockListType[]>([]);
  const [loading, setLoading] = useState(true);

  // TEMPORARY FETCHING
  useEffect(() => {
    const fetchBlocks = async () => {
      setLoading(true);
      try {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        setBlocks(outputBlocksListData);
      } catch (error) {
        console.error("Error fetching blocks:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchBlocks();
  }, []);
  return <BlocksList blocks={blocks} loading={loading} />;
};

export default OutputBlocksContent;
