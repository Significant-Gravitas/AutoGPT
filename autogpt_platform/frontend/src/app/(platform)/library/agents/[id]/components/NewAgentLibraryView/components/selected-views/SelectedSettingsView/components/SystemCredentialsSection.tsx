"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { useAgentSystemCredentials } from "../../../../hooks/useAgentSystemCredentials";
import { SystemCredentialRow } from "./SystemCredentialRow";

interface Props {
  agent: LibraryAgent;
}

export function SystemCredentialsSection({ agent }: Props) {
  const { hasSystemCredentials, systemCredentials, isLoading } =
    useAgentSystemCredentials(agent);

  if (isLoading) {
    return (
      <div className="flex w-full max-w-2xl flex-col items-start gap-4 rounded-xl border border-zinc-100 bg-white p-6">
        <Text variant="large-semibold">System Credentials</Text>
        <Text variant="body" className="text-muted-foreground">
          Loading credentials...
        </Text>
      </div>
    );
  }

  if (!hasSystemCredentials) return null;

  // Group by credential field key (from schema) to show one row per field
  const credentialsByField = systemCredentials.reduce(
    (acc, item) => {
      if (!acc[item.key]) {
        acc[item.key] = item;
      }
      return acc;
    },
    {} as Record<string, (typeof systemCredentials)[0]>,
  );

  return (
    <div className="flex w-full max-w-2xl flex-col items-start gap-4 rounded-xl border border-zinc-100 bg-white p-6">
      <div>
        <Text variant="large-semibold">System Credentials</Text>
        <Text variant="body" className="mt-1 text-muted-foreground">
          These credentials are managed by AutoGPT and used by the agent to
          access various services. You can switch to your own credentials if
          preferred.
        </Text>
      </div>
      <div className="w-full space-y-4">
        {Object.entries(credentialsByField).map(([fieldKey, item]) => (
          <SystemCredentialRow
            key={fieldKey}
            credentialKey={fieldKey}
            agentId={agent.id.toString()}
            schema={item.schema}
            systemCredential={item.credential}
          />
        ))}
      </div>
    </div>
  );
}
