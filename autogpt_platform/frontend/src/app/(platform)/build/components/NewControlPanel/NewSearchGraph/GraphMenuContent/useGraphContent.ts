import { useEffect, useState } from "react";
import { SearchableNode } from "../GraphMenuSearchBar/useGraphMenuSearchBar";

interface UseGraphContentProps {
  searchQuery: string;
  filteredNodes: SearchableNode[];
  onNodeSelect: (nodeId: string) => void;
}

export const useGraphContent = ({
  searchQuery,
  filteredNodes,
  onNodeSelect,
}: UseGraphContentProps) => {
  const [selectedIndex, setSelectedIndex] = useState(0);

  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, filteredNodes.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === "Enter" && filteredNodes.length > 0) {
      e.preventDefault();
      onNodeSelect(filteredNodes[selectedIndex].id);
    }
  };

  const getNodeInputOutputSummary = (node: SearchableNode) => {
    // Safety check for node data
    if (!node || !node.data) {
      return "";
    }
    
    const inputs = Object.keys(node.data?.inputSchema?.properties || {});
    const outputs = Object.keys(node.data?.outputSchema?.properties || {});
    const parts = [];
    
    if (inputs.length > 0) {
      parts.push(`Inputs: ${inputs.slice(0, 3).join(", ")}${inputs.length > 3 ? "..." : ""}`);
    }
    if (outputs.length > 0) {
      parts.push(`Outputs: ${outputs.slice(0, 3).join(", ")}${outputs.length > 3 ? "..." : ""}`);
    }
    
    return parts.join(" | ");
  };

  return {
    selectedIndex,
    setSelectedIndex,
    handleKeyDown,
    getNodeInputOutputSummary,
  };
};