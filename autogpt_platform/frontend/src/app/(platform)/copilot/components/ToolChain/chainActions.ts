"use client";

import { createContext } from "react";
import type {
  ConnectorRequest,
  InputsRequest,
  McpConnectorRequest,
  QuestionRequest,
} from "../ChainActionCard/helpers";

/** One user-actionable card (credential setup, clarifying questions) inside
 *  a tool chain. Cards register themselves instead of rendering their own
 *  Proceed/Answer buttons; the chain renders a single Proceed that drafts
 *  every card's message into the chat input in one go. */
export interface ChainActionEntry {
  id: string;
  ready: boolean;
  buildMessage: () => string | null;
  /** The message only confirms that credentials are in place. Several cards
   *  in one chain often ask for the same account, and the model needs to be
   *  told once. */
  credentialsOnly?: boolean;
  /** Awaited before the chain's reply goes out. The reply is what makes the
   *  tool run again, so anything it must find in place is settled here. */
  beforeSend?: () => Promise<void>;
  onSent?: () => void;
  /** This card's reply must be reviewed before it is sent, so the chain owes
   *  it a Proceed even when it asks for nothing but credentials. */
  manualProceed?: boolean;
  /** A sign-in completed on this card during this page life. The chain sends
   *  only when one has, so a chat reloaded from history stays silent. */
  justConnected?: boolean;
  /** Every credential this card asked for is in place. Distinct from `ready`,
   *  which also waits on run inputs the user is still typing. */
  credentialsReady?: boolean;
  /** Credentials this card needs. The chain merges every entry's request
   *  into the single connectors table it renders underneath itself. */
  connectors?: ConnectorRequest;
  /** MCP server this card needs connected — rendered as a row in the same
   *  connectors table, driven by the hidden MCPSetupCard's state machine. */
  mcp?: McpConnectorRequest;
  /** Editable run inputs this card collects, rendered in the same card as
   *  the connectors instead of inside the chain rows. */
  inputs?: InputsRequest;
  /** Clarifying questions this card asks, rendered in the same card as the
   *  connectors so everything the user owes the chain is in one place. */
  questions?: QuestionRequest;
}

/** A registration belongs to the component that made it (`owner`, its
 *  `useId()`), not to the card's id, so one mounted copy of a card unmounting
 *  can never remove an entry another copy still owns. */
export interface ChainActions {
  register: (owner: string, entry: ChainActionEntry) => void;
  unregister: (owner: string) => void;
}

export const ChainActionsContext = createContext<ChainActions | null>(null);

/** One entry per card id. Only the question card shares its id (the dock id)
 *  between copies, and only for a moment; when two are mounted, the copy that
 *  mounted last is shown. Setup and MCP cards use per-instance ids. */
export function uniqueChainEntries(
  registrations: ReadonlyMap<string, ChainActionEntry>,
): ChainActionEntry[] {
  const byId = new Map<string, ChainActionEntry>();
  registrations.forEach((entry) => byId.set(entry.id, entry));
  return [...byId.values()];
}

/** The chain's single reply. Every card contributes its line, except that
 *  cards which only confirm credentials share one confirmation: two cards
 *  asking for the same account used to say so twice in one message. */
export function buildChainReply(entries: ChainActionEntry[]): string {
  let confirmed = false;
  const lines: string[] = [];
  for (const entry of entries) {
    const line = entry.buildMessage();
    if (!line) continue;
    if (entry.credentialsOnly) {
      if (confirmed) continue;
      confirmed = true;
    }
    lines.push(line);
  }
  return lines.join("\n\n");
}
