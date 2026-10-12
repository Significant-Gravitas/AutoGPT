import { useRouter } from "next/navigation";
import type {
  DeletionSource,
  ExitReason,
  PanelActionKind,
} from "../../helpers";
import { useConnectedApps } from "../../useConnectedApps";
import { usePlanContext } from "../../usePlanContext";
import { PANEL_ROUTES } from "./helpers";
import { getReasonPanel } from "./panel-helpers";
import { useDisconnectApps } from "./useDisconnectApps";
import { useExitFeedback } from "./useExitFeedback";

interface Args {
  reason: ExitReason | null;
  origin: DeletionSource;
  onClose: () => void;
}

export function useReasonPanel({ reason, origin, onClose }: Args) {
  const router = useRouter();
  const { apps, targets } = useConnectedApps();
  const { planState, pending, proMonthlyCents, trialEndsAt } = usePlanContext();
  const disconnect = useDisconnectApps();
  const { sendAlternative } = useExitFeedback();

  const panel = getReasonPanel({
    reason,
    planState,
    pending,
    apps,
    proMonthlyCents,
    trialEndsAt,
    origin,
    disconnect: disconnect.state,
  });

  function runAction(kind: PanelActionKind) {
    if (!reason) return;
    sendAlternative({ reason, alternative: kind });
    if (kind === "disconnect_all") {
      void disconnect.disconnectAll(targets);
      return;
    }
    onClose();
    const route = PANEL_ROUTES[kind];
    if (route) router.push(route);
  }

  return {
    panel,
    isActionPending:
      disconnect.isPending && panel?.action?.kind === "disconnect_all",
    runAction,
  };
}
