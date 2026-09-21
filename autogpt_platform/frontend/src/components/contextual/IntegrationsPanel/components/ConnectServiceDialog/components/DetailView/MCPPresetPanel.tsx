"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
  TabsLineContent,
} from "@/components/molecules/TabsLine/TabsLine";
import { McpConnectPanel } from "./McpConnectPanel";
import { PublicMCPPanel } from "./PublicMCPPanel";

type AuthMethod = NonNullable<
  ProviderMetadata["mcp_server"]
>["auth_methods"][number];

const METHOD_LABELS: Record<AuthMethod, string> = {
  none: "No sign-in",
  oauth: "Sign in",
  bearer: "API token",
  basic: "Basic authentication",
};

interface Props {
  server: NonNullable<ProviderMetadata["mcp_server"]>;
  onSuccess: (credential?: CredentialsMetaResponse) => void;
}

export function MCPPresetPanel({ server, onSuccess }: Props) {
  const methods = server.auth_methods;
  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" className="text-zinc-600">
        {server.setup_instructions}
      </Text>
      <Link href={server.documentation_url} isExternal variant="secondary">
        Official documentation
      </Link>
      {methods.length === 1 ? (
        <PresetMethod
          server={server}
          onSuccess={onSuccess}
          method={methods[0]}
        />
      ) : (
        <TabsLine defaultValue={methods[0]}>
          <TabsLineList aria-label="Connection method">
            {methods.map((method) => (
              <TabsLineTrigger key={method} value={method}>
                {METHOD_LABELS[method]}
              </TabsLineTrigger>
            ))}
          </TabsLineList>
          {methods.map((method) => (
            <TabsLineContent key={method} value={method}>
              <PresetMethod
                server={server}
                onSuccess={onSuccess}
                method={method}
              />
            </TabsLineContent>
          ))}
        </TabsLine>
      )}
    </div>
  );
}

function PresetMethod({
  server,
  onSuccess,
  method,
}: Props & { method: AuthMethod }) {
  if (method === "none") {
    return server.server_url ? (
      <PublicMCPPanel serverURL={server.server_url} />
    ) : null;
  }
  return (
    <McpConnectPanel
      onSuccess={onSuccess}
      initialServerURL={
        (method === "oauth" ? server.oauth_server_url : null) ??
        server.server_url ??
        ""
      }
      lockServerURL={server.connection_mode === "hosted"}
      allowedAuthMethods={[method]}
      oauthScopes={server.oauth_scopes}
      oauthWriteScopes={server.oauth_write_scopes}
      serverURLOptions={server.server_url_options}
    />
  );
}
