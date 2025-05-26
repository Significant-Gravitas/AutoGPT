import React, { useEffect, useState } from "react";
import BlocksList from "./BlocksList";
import { Block } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

const InputBlocksContent: React.FC = () => {
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [loading, setLoading] = useState(true);
  const api = useBackendAPI();

  useEffect(() => {
    const fetchBlocks = async () => {
      setLoading(true);
      try {
        const response = await api.getBuilderBlocks({ type: "input" });
        setBlocks(response.blocks);
      } catch (error) {
        console.error("Error fetching blocks:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchBlocks();
  }, [api]);
  return <BlocksList blocks={blocks} loading={loading} />;
};

export default InputBlocksContent;
