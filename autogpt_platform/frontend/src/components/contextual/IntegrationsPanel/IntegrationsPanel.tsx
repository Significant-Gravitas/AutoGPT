"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { useState } from "react";

import { Text } from "@/components/atoms/Text/Text";

import { AIConnectionsSection } from "./components/AIConnectionsSection/AIConnectionsSection";
import { ConnectServiceDialog } from "./components/ConnectServiceDialog/ConnectServiceDialog";
import { IntegrationsHeader } from "./components/IntegrationsHeader/IntegrationsHeader";
import { IntegrationsList } from "./components/IntegrationsList/IntegrationsList";
import { AvailableIntegrations } from "./components/AvailableIntegrations/AvailableIntegrations";

const SURFACE_BACKGROUND = {
  page: "bg-[#F9F9FA]",
  dialog: "bg-white",
} as const;

interface Props {
  withHeading?: boolean;
  /** What the panel sits on, so the sticky search matches it. */
  surface?: keyof typeof SURFACE_BACKGROUND;
  preferMcp?: boolean;
  onConnected?: (credential: CredentialsMetaResponse) => void;
}

export function IntegrationsPanel({
  withHeading = true,
  surface = "page",
  preferMcp,
  onConnected,
}: Props) {
  const [isConnectOpen, setIsConnectOpen] = useState(false);
  const [selectedProviderId, setSelectedProviderId] = useState<string | null>(
    null,
  );
  const [query, setQuery] = useState("");

  function openConnect(providerId: string | null = null) {
    setSelectedProviderId(providerId);
    setIsConnectOpen(true);
  }

  return (
    <>
      <IntegrationsHeader
        onConnect={() => openConnect()}
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
        <IntegrationsList
          query={query}
          onQueryChange={setQuery}
          stickyClassName={SURFACE_BACKGROUND[surface]}
        />
      </section>
      <AvailableIntegrations query={query} onSelect={openConnect} />
      <ConnectServiceDialog
        open={isConnectOpen}
        onOpenChange={setIsConnectOpen}
        initialProviderId={selectedProviderId}
        preferMcp={preferMcp}
        onConnected={onConnected}
      />
    </>
  );
}
