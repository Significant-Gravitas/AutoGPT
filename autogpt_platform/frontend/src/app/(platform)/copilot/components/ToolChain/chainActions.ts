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
  onSent?: () => void;
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

export interface ChainActions {
  register: (entry: ChainActionEntry) => void;
  unregister: (id: string) => void;
}

export const ChainActionsContext = createContext<ChainActions | null>(null);
