import { useCallback, useEffect, useRef, useState } from "react";
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
  const itemRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  useEffect(() => {
    const el = itemRefs.current.get(selectedIndex);
    if (el) {
      el.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  const setItemRef = useCallback((index: number, el: HTMLDivElement | null) => {
    if (el) {
      itemRefs.current.set(index, el);
    } else {
      itemRefs.current.delete(index);
    }
  }, []);

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown" && filteredNodes.length > 0) {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, filteredNodes.length - 1));
    } else if (e.key === "ArrowUp" && filteredNodes.length > 0) {
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
    setItemRef,
    handleKeyDown,
  };
}
