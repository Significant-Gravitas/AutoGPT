import { useControlPanelStore } from "@/app/(platform)/build/stores/controlPanelStore";
import { CustomNode } from "../../../FlowEditor/nodes/CustomNode/CustomNode";
import { useGraphSearch } from "../GraphMenuSearchBar/useGraphMenuSearchBar";

interface UseGraphMenuProps {
  nodes: CustomNode[];
  onNodeSelect: (nodeID: string) => void;
}

export function useGraphMenu({ nodes, onNodeSelect }: UseGraphMenuProps) {
  const { setGraphSearchOpen } = useControlPanelStore();
  const { searchQuery, setSearchQuery, filteredNodes } = useGraphSearch(nodes);

  function handleNodeSelect(nodeID: string) {
    onNodeSelect(nodeID);
    setGraphSearchOpen(false);
    setSearchQuery("");
  }

  return {
    searchQuery,
    setSearchQuery,
    filteredNodes,
    handleNodeSelect,
  };
}
