"use client";

import { useState } from "react";

import { ConnectServiceDialog } from "./components/ConnectServiceDialog/ConnectServiceDialog";
import { IntegrationsHeader } from "./components/IntegrationsHeader/IntegrationsHeader";
import { IntegrationsList } from "./components/IntegrationsList/IntegrationsList";

export default function SettingsIntegrationsPage() {
  const [isConnectOpen, setIsConnectOpen] = useState(false);

  return (
    <>
      <IntegrationsHeader onConnect={() => setIsConnectOpen(true)} />
      <IntegrationsList />
      <ConnectServiceDialog
        open={isConnectOpen}
        onOpenChange={setIsConnectOpen}
      />
    </>
  );
}
