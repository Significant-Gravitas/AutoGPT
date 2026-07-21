"use client";

import { useState } from "react";
import { ConnectServiceDialog } from "./components/ConnectServiceDialog/ConnectServiceDialog";
import { IntegrationsHeader } from "./components/IntegrationsHeader/IntegrationsHeader";
import { IntegrationsList } from "./components/IntegrationsList/IntegrationsList";

interface Props {
  withHeading?: boolean;
}

export function IntegrationsPanel({ withHeading = true }: Props) {
  const [isConnectOpen, setIsConnectOpen] = useState(false);

  return (
    <>
      <IntegrationsHeader
        onConnect={() => setIsConnectOpen(true)}
        withTitle={withHeading}
      />
      <IntegrationsList />
      <ConnectServiceDialog
        open={isConnectOpen}
        onOpenChange={setIsConnectOpen}
      />
    </>
  );
}
