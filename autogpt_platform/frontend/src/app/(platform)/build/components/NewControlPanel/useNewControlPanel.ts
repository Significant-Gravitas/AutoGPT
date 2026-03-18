import { useState } from "react";

export interface NewControlPanelProps {
  visualizeBeads?: "no" | "static" | "animate";
}

export const useNewControlPanel = ({
  visualizeBeads: _visualizeBeads,
}: NewControlPanelProps) => {
  const [blockMenuSelected, setBlockMenuSelected] = useState<
    "save" | "block" | "search" | ""
  >("");

  return {
    blockMenuSelected,
    setBlockMenuSelected,
  };
};
