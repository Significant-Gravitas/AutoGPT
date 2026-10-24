"use client";

import { ConnectToolsDialog } from "@/app/(platform)/copilot/components/ConnectToolsDialog/ConnectToolsDialog";
import { TourSpotlight } from "./components/TourSpotlight";
import { useFirstRunTourStore } from "./store";
import { useFirstRunTour } from "./useFirstRunTour";

// The guided first run over the real sidebar. Rendered once, inside the app
// chrome, and idle until something opens it: the first home visit after
// onboarding, or "Show me around" in the profile menu.
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
