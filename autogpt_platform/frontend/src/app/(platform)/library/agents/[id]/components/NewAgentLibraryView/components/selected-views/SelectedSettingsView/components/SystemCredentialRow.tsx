"use client";

import { Text } from "@/components/atoms/Text/Text";
import { CredentialsMetaResponse } from "@/lib/autogpt-server-api/types";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import { useEffect, useState } from "react";
import { CredentialsInput } from "../../../../components/modals/CredentialsInputs/CredentialsInputs";
import {
  NONE_CREDENTIAL_MARKER,
  useAgentCredentialPreferencesStore,
} from "../../../../stores/agentCredentialPreferencesStore";

interface Props {
  credentialKey: string;
  agentId: string;
  schema: any;
  systemCredential: CredentialsMetaResponse;
}

export function SystemCredentialRow({
  credentialKey,
  agentId,
  schema,
  systemCredential,
}: Props) {
  const store = useAgentCredentialPreferencesStore();

  // Initialize with saved preference or default to system credential
  const savedPreference = store.getCredentialPreference(agentId, credentialKey);
  const defaultCredential = {
    id: systemCredential.id,
    type: systemCredential.type,
    provider: systemCredential.provider,
    title: systemCredential.title,
  };

  // If saved preference is the NONE marker, use undefined (which CredentialsInput interprets as "None")
  // Otherwise use saved preference or default
  const [selectedCredential, setSelectedCredential] = useState<any>(
    savedPreference === NONE_CREDENTIAL_MARKER
      ? undefined
      : savedPreference || defaultCredential,
  );

  // Update when preference changes externally
  useEffect(() => {
    const preference = store.getCredentialPreference(agentId, credentialKey);
    if (preference === NONE_CREDENTIAL_MARKER) {
      setSelectedCredential(undefined);
    } else if (preference) {
      setSelectedCredential(preference);
    } else {
      setSelectedCredential(defaultCredential);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [credentialKey, agentId]);

  const providerName = schema.credentials_provider?.[0] || "";
  const displayName = toDisplayName(providerName);

  function handleSelectCredentials(value: any) {
    setSelectedCredential(value);
    // Save preference:
    // - undefined = explicitly selected "None" (save NONE_CREDENTIAL_MARKER)
    // - null = use default system credential (fallback behavior, save null)
    // - credential object = use this specific credential
    if (value === undefined) {
      // User explicitly selected "None" - save special marker
      store.setCredentialPreference(
        agentId,
        credentialKey,
        NONE_CREDENTIAL_MARKER,
      );
    } else if (value === null) {
      // User cleared selection - use default system credential
      store.setCredentialPreference(agentId, credentialKey, null);
    } else {
      // User selected a credential
      store.setCredentialPreference(agentId, credentialKey, value);
    }
  }

  return (
    <div className="rounded-lg border border-zinc-100 bg-zinc-50/50 px-4 pb-2 pt-4">
      <Text variant="body-medium" className="mb-2 ml-2">
        {displayName}
      </Text>

      <CredentialsInput
        schema={{ ...schema, discriminator: undefined }}
        selectedCredentials={selectedCredential}
        onSelectCredentials={handleSelectCredentials}
        showTitle={false}
        isOptional
        allowSystemCredentials={true}
      />
    </div>
  );
}
