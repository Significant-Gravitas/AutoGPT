"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { TokenDevtoolPanel } from "./components/TokenDevtoolPanel";
import { useTokenDevtoolBadge } from "./useTokenDevtoolBadge";

interface Props {
  sessionId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** The token devtool readout as a dialog, opened from the composer's +. */
export function TokenDevtoolDialog({ sessionId, open, onOpenChange }: Props) {
  const devtool = useTokenDevtoolBadge(sessionId);

  return (
    <Dialog
      title="Token usage"
      controlled={{ isOpen: open, set: onOpenChange }}
      styling={{ maxWidth: "24rem" }}
    >
      <Dialog.Content>
        <TokenDevtoolPanel {...devtool} />
      </Dialog.Content>
    </Dialog>
  );
}
