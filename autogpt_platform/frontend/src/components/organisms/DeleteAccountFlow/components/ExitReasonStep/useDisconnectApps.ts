import {
  useDeleteIntegration,
  type DeleteIntegrationTarget,
} from "@/components/contextual/IntegrationsPanel/components/hooks/useDeleteIntegration";
import { useState } from "react";
import type { DisconnectState } from "../../helpers";
import { getDisconnectResult, getTargetNames } from "./helpers";

export function useDisconnectApps() {
  const { remove, isPending } = useDeleteIntegration();
  const [state, setState] = useState<DisconnectState>({ status: "idle" });

  async function disconnectAll(targets: DeleteIntegrationTarget[]) {
    if (isPending) return;
    // A retry after a failure finds nothing left when the refetched list no
    // longer holds the apps that failed: they are gone, so say so.
    if (targets.length === 0) {
      if (state.status === "failed") {
        setState({ status: "done", names: state.names });
      }
      return;
    }
    try {
      setState(getDisconnectResult(await remove(targets, true)));
    } catch {
      setState({ status: "failed", names: getTargetNames(targets) });
    }
  }

  return { state, isPending, disconnectAll };
}
