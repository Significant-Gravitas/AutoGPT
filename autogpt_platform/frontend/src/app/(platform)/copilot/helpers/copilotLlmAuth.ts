import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import type { CopilotLlmAuthSelection } from "../store";

export function getAvailableLlmTransports(
  transports: ChatTransportResponse[] | null | undefined,
): ChatTransportResponse[] {
  return (transports ?? []).filter(
    (transport) =>
      transport.available &&
      (transport.auth_provider === "platform" ||
        transport.credential_id !== null),
  );
}

export function getChatTransportSelection(
  transport: ChatTransportResponse,
): CopilotLlmAuthSelection | null {
  if (transport.auth_provider === "platform") {
    return { authProvider: "platform", credentialId: null };
  }
  if (transport.credential_id === null) return null;
  return {
    authProvider: "codex",
    credentialId: transport.credential_id,
  };
}

export function findSelectedLlmTransport(
  transports: ChatTransportResponse[] | null | undefined,
  selection: CopilotLlmAuthSelection,
): ChatTransportResponse | undefined {
  return getAvailableLlmTransports(transports).find(
    (transport) =>
      transport.auth_provider === selection.authProvider &&
      (transport.auth_provider === "platform" ||
        transport.credential_id === selection.credentialId),
  );
}

export function resolveCopilotLlmAuthSelection(
  transports: ChatTransportResponse[] | null | undefined,
  currentSelection: CopilotLlmAuthSelection,
): CopilotLlmAuthSelection | null {
  if (transports == null) return null;

  const availableTransports = getAvailableLlmTransports(transports);
  const selectedTransport = findSelectedLlmTransport(
    availableTransports,
    currentSelection,
  );
  if (selectedTransport) return currentSelection;

  const defaultTransport = availableTransports.find(
    (transport) => transport.default,
  );
  if (defaultTransport) return getChatTransportSelection(defaultTransport);

  if (availableTransports.length === 1) {
    return getChatTransportSelection(availableTransports[0]);
  }

  return null;
}
