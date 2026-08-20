import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";

/**
 * The label the server sends for the platform transport on a self-hosted
 * install. It sends "AutoGPT Platform" on the hosted product instead.
 */
export const SELF_HOSTED_LABEL = "Self-hosted chat";

/**
 * Says what backs a run, which is the thing that actually differs between
 * connections. Copy lives here only until the server-owned connection offer
 * carries it — the client should not be deciding how a provider describes
 * its own billing.
 */
export function describeTransport(transport: ChatTransportResponse): string {
  if (transport.auth_provider === "codex") {
    return "New chats are backed by your ChatGPT plan, and spend no AutoGPT credits.";
  }
  // The ChatGPT line promises "no AutoGPT credits", which only means anything
  // if the row it contrasts with says credits are spent. Self-host has no
  // credits at all, so it says what it actually has instead.
  return transport.label === SELF_HOSTED_LABEL
    ? "New chats are backed by the chat provider configured on this server."
    : "New chats are backed by your AutoGPT plan, and spend AutoGPT credits.";
}

export function isSelectable(transport: ChatTransportResponse): boolean {
  return (
    transport.available &&
    (transport.auth_provider === "platform" || transport.credential_id !== null)
  );
}

/**
 * A stable key. The platform transport carries no credential id, and a user
 * can hold several ChatGPT accounts, so neither half identifies a row alone.
 */
export function transportKey(transport: ChatTransportResponse): string {
  return `${transport.auth_provider}:${transport.credential_id ?? "deployment"}`;
}
