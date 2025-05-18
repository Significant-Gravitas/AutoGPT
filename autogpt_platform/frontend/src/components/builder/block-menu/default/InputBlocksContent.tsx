import React, { useEffect, useState } from "react";
import BlocksList from "./BlocksList";
import { BlockListType } from "./BlockMenuDefaultContent";
import { inputBlocksListData } from "../../testing_data";

const InputBlocksContent: React.FC = () => {
  const [blocks, setBlocks] = useState<BlockListType[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBlocks = async () => {
      setLoading(true);
      try {
        // Simulate API call
        await new Promise((resolve) => setTimeout(resolve, 1000));
        setBlocks(inputBlocksListData);
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

export default InputBlocksContent;
