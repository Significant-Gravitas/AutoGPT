import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import type { CredentialField } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import type { RJSFSchema } from "@rjsf/utils";
import type { ClarifyingQuestion } from "../../tools/clarifying-questions";

/** One question card's ask. The card owns the inputs; the asking component
 *  keeps the answers so it can still build its own message. */
export interface QuestionRequest {
  id: string;
  questions: ClarifyingQuestion[];
  answers: Record<string, string>;
  onAnswer: (keyword: string, value: string) => void;
  onSkip: () => void;
}

/** One setup card's editable run inputs, lifted out of the chain rows into
 *  its own card below the chain — one card per block/agent, titled by it. */
export interface InputsRequest {
  id: string;
  title?: string;
  schema: RJSFSchema | null;
  values: Record<string, unknown>;
  onChange: (values: Record<string, unknown>) => void;
  hasAdvanced: boolean;
  showAdvanced: boolean;
  onToggleAdvanced: () => void;
}

/** "SendDiscordMessageBlock" → "Send Discord Message". Agent names with
 *  regular spacing pass through unchanged. */
export function formatInputsTitle(name: string): string {
  const spaced = name
    .replace(/Block$/, "")
    .replace(/_/g, " ")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .trim();
  return spaced || name;
}

/** One MCP server's ask. Unlike ConnectorRequest, MCP servers aren't
 *  platform providers — the hidden MCPSetupCard keeps the OAuth/token state
 *  machine and hands the table row its state + callbacks. */
export interface McpConnectorRequest {
  id: string;
  service: string;
  serverUrl: string;
  connected: boolean;
  loading: boolean;
  error: string | null;
  showManualToken: boolean;
  authScheme: "basic" | "bearer";
  onConnect: () => void;
  onUseToken: (token: string) => void;
}

/** One setup card's ask, handed to the chain so every card in the chain can
 *  be answered from a single connectors table. */
export interface ConnectorRequest {
  id: string;
  fields: CredentialField[];
  selected: Record<string, CredentialsMetaInput | undefined>;
  onChange: (key: string, value?: CredentialsMetaInput) => void;
  /** The user finished a sign-in on this row. Distinct from a credential
   *  merely being present, which is also true of a card re-rendered from
   *  chat history. */
  onConnected: () => void;
}

export interface ConnectorRow {
  provider: string;
  displayName: string;
  description: string | null;
  schema: CredentialField[1];
  selected?: CredentialsMetaInput;
  select: (value?: CredentialsMetaInput) => void;
  onConnected: () => void;
}

/** Flattens every request into one row per provider: two tools asking for
 *  GitHub is one "Connect GitHub", and connecting it answers both. Fields
 *  whose provider can't be resolved are dropped — nothing to connect. */
export function toConnectorRows(
  requests: ConnectorRequest[],
  providers: ProviderMetadata[],
): ConnectorRow[] {
  const byName = new Map(providers.map((p) => [p.name, p]));
  const rows = new Map<
    string,
    {
      schema: CredentialField[1];
      selected?: CredentialsMetaInput;
      targets: { request: ConnectorRequest; key: string }[];
    }
  >();

  for (const request of requests) {
    for (const [key, schema] of request.fields) {
      const provider = schema.credentials_provider?.[0];
      if (!provider) continue;
      const row = rows.get(provider);
      if (!row) {
        rows.set(provider, {
          schema,
          targets: [{ request, key }],
          selected: request.selected[key],
        });
        continue;
      }
      // One row answers every card that asked for this provider, so it must
      // request the union of their scopes — keeping only the first card's
      // leaves the others permanently unsatisfiable. Scopes only: merging
      // `credentials_types` would offer a method some cards cannot accept, and
      // that needs a per-card row rather than a wider one.
      row.schema = withUnionedScopes(row.schema, schema);
      row.targets.push({ request, key });
      row.selected = row.selected ?? request.selected[key];
    }
  }

  return [...rows.entries()].map(([provider, row]) => ({
    provider,
    displayName: formatProviderName(provider),
    description:
      byName.get(provider)?.description ?? row.schema.description ?? null,
    schema: row.schema,
    selected: row.selected,
    select: (value?: CredentialsMetaInput) =>
      row.targets.forEach(({ request, key }) => request.onChange(key, value)),
    onConnected: () =>
      row.targets.forEach(({ request }) => request.onConnected()),
  }));
}

/** Merges `incoming`'s scopes into `kept`, leaving every other schema field
 *  as the first card set it. */
function withUnionedScopes(
  kept: CredentialField[1],
  incoming: CredentialField[1],
): CredentialField[1] {
  const scopes = [
    ...new Set([
      ...(kept.credentials_scopes ?? []),
      ...(incoming.credentials_scopes ?? []),
    ]),
  ];
  return scopes.length > 0 ? { ...kept, credentials_scopes: scopes } : kept;
}
