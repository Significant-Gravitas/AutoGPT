import { useEffect, useState } from "react";
import { SearchableNode } from "../GraphMenuSearchBar/useGraphMenuSearchBar";

interface UseGraphContentProps {
  searchQuery: string;
  filteredNodes: SearchableNode[];
  onNodeSelect: (nodeID: string) => void;
}

export function useGraphContent({
  searchQuery,
  filteredNodes,
  onNodeSelect,
}: UseGraphContentProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);

  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, filteredNodes.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === "Enter" && filteredNodes.length > 0) {
      e.preventDefault();
      const safeIndex = Math.max(
        0,
        Math.min(selectedIndex, filteredNodes.length - 1),
      );
      const node = filteredNodes[safeIndex];
      if (node) {
        onNodeSelect(node.id);
      }
    }
  }

  return {
    selectedIndex,
    setSelectedIndex,
    handleKeyDown,
  };
}
