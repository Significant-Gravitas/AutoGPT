import { useGraphSearch } from "../GraphMenuSearchBar/useGraphMenuSearchBar";
import { CustomNode } from "@/components/CustomNode";

interface UseGraphMenuProps {
  nodes: CustomNode[];
  blockMenuSelected: "save" | "block" | "search" | "";
  setBlockMenuSelected: React.Dispatch<
    React.SetStateAction<"" | "save" | "block" | "search">
  >;
  onNodeSelect: (nodeId: string) => void;
}

export const useGraphMenu = ({
  nodes,
  setBlockMenuSelected,
  onNodeSelect,
}: UseGraphMenuProps) => {
  const { open, setOpen, searchQuery, setSearchQuery, filteredNodes } = 
    useGraphSearch(nodes);

  const handleNodeSelect = (nodeId: string) => {
    onNodeSelect(nodeId);
    setOpen(false);
    setSearchQuery("");
    setBlockMenuSelected("");
  };

  const handleOpenChange = (newOpen: boolean) => {
    setOpen(newOpen);
    setBlockMenuSelected(newOpen ? "search" : "");
  };

  return {
    open,
    searchQuery,
    setSearchQuery,
    filteredNodes,
    handleNodeSelect,
    handleOpenChange,
  };
};