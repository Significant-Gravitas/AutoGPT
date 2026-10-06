"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import { Button } from "@/components/atoms/Button/Button";
import { MCPPresetPanel } from "./MCPPresetPanel";

interface Props {
  server: NonNullable<ProviderMetadata["mcp_server"]>;
  /** Whether the vendor also takes an API key or other block credential. */
  hasNativeMethods: boolean;
  onUseNative: () => void;
  onSuccess: (credential?: CredentialsMetaResponse) => void;
}

/** The service's own sign-in, shown first, with the block credential one
 *  click away when the vendor also ships one. */
export function McpFirstPanel({
  server,
  hasNativeMethods,
  onUseNative,
  onSuccess,
}: Props) {
  return (
    <div className="flex flex-col gap-4">
      <MCPPresetPanel server={server} onSuccess={onSuccess} />
      {hasNativeMethods ? (
        <Button variant="ghost" size="small" onClick={onUseNative}>
          Connect with an API key instead
        </Button>
      ) : null}
    </div>
  );
}
