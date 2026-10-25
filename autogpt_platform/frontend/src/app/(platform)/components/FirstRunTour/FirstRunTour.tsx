"use client";

import { ConnectToolsDialog } from "@/app/(platform)/copilot/components/ConnectToolsDialog/ConnectToolsDialog";
import { TourSpotlight } from "./components/TourSpotlight";
import { useFirstRunTourStore } from "./store";
import { useFirstRunTour } from "./useFirstRunTour";

export function FirstRunTour() {
  const source = useFirstRunTourStore((state) => state.source);
  const isConnectToolsOpen = useFirstRunTourStore(
    (state) => state.isConnectToolsOpen,
  );
  const setConnectToolsOpen = useFirstRunTourStore(
    (state) => state.setConnectToolsOpen,
  );
  return (
    <>
      {source !== null && <OpenTour />}
      <ConnectToolsDialog
        isOpen={isConnectToolsOpen}
        onClose={() => setConnectToolsOpen(false)}
      />
    </>
  );
}

function OpenTour() {
  const tour = useFirstRunTour();
  return <TourSpotlight tour={tour} />;
}
