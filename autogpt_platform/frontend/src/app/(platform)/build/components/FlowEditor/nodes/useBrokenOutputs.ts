import { useMemo } from "react";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";

/**
 * Hook to get the set of broken output names for a node in resolution mode.
 */
export function useBrokenOutputs(nodeID: string): Set<string> {
  // Subscribe to the actual state values, not just methods
  const isInResolution = useNodeStore((state) =>
    state.nodesInResolutionMode.has(nodeID),
  );
  const resolutionData = useNodeStore((state) =>
    state.nodeResolutionData.get(nodeID),
  );

  return useMemo(() => {
    if (!isInResolution || !resolutionData) {
      return new Set<string>();
    }

    return new Set(resolutionData.incompatibilities.missingOutputs);
  }, [isInResolution, resolutionData]);
}
