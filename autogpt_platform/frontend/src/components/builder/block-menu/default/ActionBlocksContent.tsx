import React, { useEffect, useState } from "react";
import { actionBlocksListData } from "../../testing_data";
import { BlockListType } from "./BlockMenuDefaultContent";
import BlocksList from "./BlocksList";

const ActionBlocksContent: React.FC = () => {
  const [blocks, setBlocks] = useState<BlockListType[]>([]);
  const [loading, setLoading] = useState(true);

  // TEMPORARY FETCHING
  useEffect(() => {
    const fetchBlocks = async () => {
      setLoading(true);
      try {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        setBlocks(actionBlocksListData);
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

export default ActionBlocksContent;
