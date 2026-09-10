"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { McpConnectPanel } from "./McpConnectPanel";
import { PublicMCPPanel } from "./PublicMCPPanel";

interface Props {
  server: NonNullable<ProviderMetadata["mcp_server"]>;
  onSuccess: (credential?: CredentialsMetaResponse) => void;
}

export function MCPPresetPanel({ server, onSuccess }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" className="text-zinc-600">
        {server.setup_instructions}
      </Text>
      <Link href={server.documentation_url} isExternal variant="secondary">
        Official documentation
      </Link>
      {server.connection_mode === "unavailable" ? null : server.auth_mode ===
          "none" && server.server_url ? (
        <PublicMCPPanel serverURL={server.server_url} />
      ) : (
        <McpConnectPanel
          onSuccess={onSuccess}
          initialServerURL={server.server_url ?? ""}
          lockServerURL={server.connection_mode === "hosted"}
          initialAuthMode={server.auth_mode}
        />
      )}
    </div>
  );
}
