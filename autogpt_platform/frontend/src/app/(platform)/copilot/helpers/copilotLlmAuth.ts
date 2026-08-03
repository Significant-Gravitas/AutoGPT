import type { CredentialsMetaResponse } from "@/lib/autogpt-server-api";
import type { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import type { CopilotLlmAuthSelection } from "../store";

type SubsidizedLlmAuthProvider = Exclude<
  CopilotLlmAuthSelection["authProvider"],
  "platform"
>;

interface SubsidizedLlmTransportDefinition {
  authProvider: SubsidizedLlmAuthProvider;
  provider: string;
  credentialType: CredentialsMetaResponse["type"];
  label: string;
  description: string;
}

export interface ConnectedSubsidizedLlmTransport
  extends SubsidizedLlmTransportDefinition {
  credentials: CredentialsMetaResponse[];
}

const SUBSIDIZED_LLM_TRANSPORTS: SubsidizedLlmTransportDefinition[] = [
  {
    authProvider: "codex",
    provider: "codex",
    credentialType: "oauth2",
    label: "ChatGPT/Codex",
    description: "Uses your ChatGPT plan",
  },
];

export function getConnectedSubsidizedLlmTransports(
  providers: CredentialsProvidersContextType | null,
): ConnectedSubsidizedLlmTransport[] {
  if (!providers) return [];

  return SUBSIDIZED_LLM_TRANSPORTS.flatMap((transport) => {
    const credentials =
      providers[transport.provider]?.savedCredentials.filter(
        (credential) =>
          credential.provider === transport.provider &&
          credential.type === transport.credentialType,
      ) ?? [];

    return credentials.length > 0 ? [{ ...transport, credentials }] : [];
  });
}

export function getSubsidizedTransportSelection(
  transport: ConnectedSubsidizedLlmTransport,
  currentSelection: CopilotLlmAuthSelection,
): CopilotLlmAuthSelection {
  const credential =
    currentSelection.authProvider === transport.authProvider
      ? (transport.credentials.find(
          (candidate) => candidate.id === currentSelection.credentialId,
        ) ?? transport.credentials[0])
      : transport.credentials[0];

  return {
    authProvider: transport.authProvider,
    credentialId: credential.id,
  };
}

export function resolveCopilotLlmAuthSelection(
  providers: CredentialsProvidersContextType | null,
  currentSelection: CopilotLlmAuthSelection,
): CopilotLlmAuthSelection | null {
  if (providers === null) return null;

  const transports = getConnectedSubsidizedLlmTransports(providers);
  if (currentSelection.authProvider !== "platform") {
    const currentTransport = transports.find(
      (transport) => transport.authProvider === currentSelection.authProvider,
    );
    const credentialStillExists = currentTransport?.credentials.some(
      (credential) => credential.id === currentSelection.credentialId,
    );
    if (credentialStillExists) return currentSelection;
  }

  if (transports.length === 1) {
    return getSubsidizedTransportSelection(transports[0], currentSelection);
  }

  if (transports.length === 0) {
    return { authProvider: "platform", credentialId: null };
  }

  return null;
}
