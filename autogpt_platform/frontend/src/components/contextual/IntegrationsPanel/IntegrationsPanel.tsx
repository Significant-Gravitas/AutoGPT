"use client";

import { useState } from "react";

import { Text } from "@/components/atoms/Text/Text";

import { AIConnectionsSection } from "./components/AIConnectionsSection/AIConnectionsSection";
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
      <AIConnectionsSection />
      <section aria-labelledby="tool-connections-heading">
        <Text
          variant="small-medium"
          as="h2"
          id="tool-connections-heading"
          className="pb-3 pl-4 uppercase tracking-[0.06em] text-[#505057]"
        >
          Tools your agents use
        </Text>
        <IntegrationsList />
      </section>
      <ConnectServiceDialog
        open={isConnectOpen}
        onOpenChange={setIsConnectOpen}
      />
    </>
  );
}
