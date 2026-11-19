import type { PickedCredential } from "@/features/connections/connectedThisRun";
import {
  connectMethod,
  credentialLabel,
  fittingCredentials,
  hostFromURL,
  MCP_PROVIDER,
  parseCredentialRequests,
  requestHost,
  type ConnectMethod,
  type CredentialRequest,
} from "@/features/connections/helpers";
import type { CredentialMeta } from "@/lib/api/integrations";

export const SETUP_REQUIREMENTS = "setup_requirements";

const MCP_TOOL = "run_mcp_tool";
const CONNECT_TOOL = "connect_integration";
const TRIGGER_TOOL = "setup_agent_webhook_trigger";
// The web's CAPABILITY_RUN_TOOLS (ToolChain/ToolResult.tsx).
const CAPABILITY_TOOLS = ["run_capability", "resume_capability", "describe_capability"];
// Their inputs live in the workflow and are only shown, never sent.
const PREVIEW_TOOLS = ["run_agent", "schedule_agent", TRIGGER_TOOL];

/** A `setup_requirements` tool result that asks for at least one service. */
export interface SetupRequest {
  toolName: string;
  /** `setup_info.agent_name`: the service, for MCP and connect cards. */
  serviceName: string | null;
  /** `setup_info.agent_id`: the server URL on an MCP card. */
  serverURL: string | null;
  requests: CredentialRequest[];
  /** Inputs the card also sends, with their values; null when it has none. */
  inputValues: Record<string, unknown> | null;
  /** A required input has no value yet, which only the web can collect. */
  needsInputs: boolean;
  /** A saved account the provider refused, to reconnect. */
  rejection: { provider: string; credentialID: string } | null;
}

function toRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value) as unknown;
    } catch {
      return null;
    }
  }
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function text(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function isEmpty(value: unknown) {
  return value === undefined || value === null || value === "";
}

function readInputs(toolName: string, requirements: Record<string, unknown> | null) {
  const inputs = Array.isArray(requirements?.inputs) ? requirements.inputs : [];
  if (PREVIEW_TOOLS.includes(toolName) || toolName === MCP_TOOL || inputs.length === 0) {
    return { inputValues: null, needsInputs: false };
  }
  const values: Record<string, unknown> = {};
  let needsInputs = false;
  inputs.forEach((raw, index) => {
    const input = toRecord(raw);
    if (!input) return;
    const name = text(input.name) ?? `input-${index}`;
    if (!isEmpty(input.value)) values[name] = input.value;
    else if (input.required === true && input.advanced !== true) needsInputs = true;
  });
  return { inputValues: values, needsInputs };
}

function readRejection(value: unknown) {
  const record = toRecord(value);
  const provider = text(record?.provider);
  const credentialID = text(record?.credential_id);
  return provider && credentialID ? { provider, credentialID } : null;
}

/** The card a tool result asks for, or null when it isn't a setup card. */
export function parseSetupRequirements(output: unknown, toolName: string): SetupRequest | null {
  const record = toRecord(output);
  if (!record || record.type !== SETUP_REQUIREMENTS) return null;
  const setupInfo = toRecord(record.setup_info);
  const readiness = toRecord(setupInfo?.user_readiness);
  const requests = parseCredentialRequests(readiness?.missing_credentials);
  const inputs = readInputs(toolName, toRecord(setupInfo?.requirements));
  // A card that only needs an input (a Drive file, say) still sends you to the web.
  if (requests.length === 0 && !inputs.needsInputs) return null;
  return {
    toolName,
    serviceName: text(setupInfo?.agent_name),
    serverURL: text(setupInfo?.agent_id),
    requests,
    ...inputs,
    rejection: readRejection(record.rejection),
  };
}

/** The web's MCP card: `run_mcp_tool`, or a capability call to an MCP server. */
export function isMCPSetup(setup: SetupRequest) {
  return (
    setup.toolName === MCP_TOOL ||
    (CAPABILITY_TOOLS.includes(setup.toolName) &&
      setup.requests.every((request) => request.provider === MCP_PROVIDER))
  );
}

/** The name a row shows: an MCP card names its service, not "Mcp". */
export function displayName(request: CredentialRequest, setup: SetupRequest) {
  return request.provider === MCP_PROVIDER && setup.serviceName
    ? setup.serviceName
    : request.providerName;
}

export type RowStatus =
  | { kind: "connected"; credentialID: string; label: string | null }
  | { kind: "allow"; accounts: { id: string; title: string }[] }
  | { kind: "choose"; accounts: { id: string; title: string }[] }
  | { kind: "reconnect"; credentialID: string }
  | { kind: "connect"; method: ConnectMethod };

export interface RowContext {
  /** The account this row got in this app run. */
  pick: PickedCredential | undefined;
  /** The user's saved accounts. */
  credentials: CredentialMeta[];
  /** Accounts the expert may use; null outside an expert's chat. */
  grantedIDs: string[] | null;
  rejection: SetupRequest["rejection"];
}

function account(credential: CredentialMeta) {
  return { id: credential.id, title: credentialLabel(credential) ?? credential.provider };
}

/** Where a row stands and what its button does. */
export function rowStatus(request: CredentialRequest, context: RowContext): RowStatus {
  const { pick, credentials, grantedIDs } = context;
  if (pick) return { kind: "connected", credentialID: pick.id, label: pick.title };
  const rejectedID =
    context.rejection?.provider === request.provider ? context.rejection.credentialID : null;
  const refused = [
    request.gap?.kind === "reconnect_required" ? request.gap.credentialID : null,
    rejectedID,
  ].filter((id): id is string => !!id);
  const fitting = fittingCredentials(credentials, request, refused);
  if (request.expertGrant) {
    const granted = fitting.find((credential) => grantedIDs?.includes(credential.id));
    if (granted) {
      return { kind: "connected", credentialID: granted.id, label: credentialLabel(granted) };
    }
    // An account the backend also marked for reconnecting is not offered as is.
    const offered = request.expertGrant.credentials
      .filter(({ id }) => !refused.includes(id))
      .map(({ id, title }) => ({ id, title }));
    const accounts = [
      ...offered,
      ...fitting.map(account).filter((item) => !offered.some((o) => o.id === item.id)),
    ];
    if (accounts.length > 0) return { kind: "allow", accounts };
  } else if (fitting.length === 1) {
    return { kind: "connected", credentialID: fitting[0].id, label: credentialLabel(fitting[0]) };
  } else if (fitting.length > 1) {
    return { kind: "choose", accounts: fitting.map(account) };
  }
  const method = connectMethod(request);
  const onFile = (id: string | null) => !!id && credentials.some((c) => c.id === id);
  const reconnectID = request.gap?.credentialID ?? (onFile(rejectedID) ? rejectedID : null);
  if (reconnectID && (method === "oauth" || method === "mcp")) {
    return { kind: "reconnect", credentialID: reconnectID };
  }
  return { kind: "connect", method };
}

const CONNECT_LABELS: Record<ConnectMethod, string | null> = {
  oauth: "Sign in",
  mcp: "Sign in",
  api_key: "Add key",
  user_password: "Add login",
  host_scoped: "Add",
  web: null,
};

/** The row's button, or null when it has none. */
export function actionLabel(status: RowStatus) {
  switch (status.kind) {
    case "connected":
      return null;
    case "allow":
      return "Allow";
    case "choose":
      return "Choose account";
    case "reconnect":
      return "Reconnect";
    case "connect":
      return CONNECT_LABELS[status.method];
  }
}

const CONNECT_INTROS: Partial<Record<ConnectMethod, (name: string) => string>> = {
  oauth: (name) => `Sign in to ${name} to continue.`,
  mcp: (name) => `Sign in to ${name} to continue.`,
  api_key: (name) => `Add your ${name} key to continue.`,
  user_password: (name) => `Add your ${name} login to continue.`,
  web: (name) => `Finish connecting ${name} on the web.`,
};

/**
 * The line under the card's title, from its rows. The tool's own `message`
 * is written for Otto, so it is never shown.
 */
export function cardIntro(rows: { name: string; status: RowStatus; usesToken?: boolean }[]) {
  const open = rows.filter((row) => row.status.kind !== "connected");
  if (open.length === 0) return null;
  if (open.length > 1) return "Connect these services to continue.";
  const [{ name, status, usesToken }] = open;
  if (usesToken) return `Add a token for ${name} to continue.`;
  switch (status.kind) {
    case "allow":
      return `Allow access to your ${name} account to continue.`;
    case "choose":
      return `Choose which ${name} account to use.`;
    case "reconnect":
      return `Reconnect ${name} to continue.`;
    case "connect":
      return CONNECT_INTROS[status.method]?.(name) ?? `Connect ${name} to continue.`;
    case "connected":
      return null;
  }
}

/** The small line under the service's name. */
export function rowHint(request: CredentialRequest, status: RowStatus, setup: SetupRequest) {
  switch (status.kind) {
    case "connected":
      return status.label;
    case "allow":
      return "Already connected · needs your OK";
    case "choose":
      return "More than one account";
    case "reconnect":
      return request.gap?.kind === "missing_scopes" ? "Needs more access" : "Needs reconnecting";
    case "connect":
      return connectHint(request, status.method, setup);
  }
}

function connectHint(request: CredentialRequest, method: ConnectMethod, setup: SetupRequest) {
  if (method === "api_key") return "API key";
  if (method === "user_password") return "Username and password";
  if (method === "host_scoped") return requestHost(request) ?? "Request header";
  if (method === "web") return "Finish this on the web";
  if (method === "mcp") {
    const url = request.discriminatorValues[0] ?? setup.serverURL;
    return url ? hostFromURL(url) : null;
  }
  return null;
}

export interface ConnectedRow {
  request: CredentialRequest;
  credentialID: string;
}

/** What the chat records as picked: per provider, or `provider:host`. */
export function buildSelections(rows: ConnectedRow[]) {
  const selections: Record<string, string> = {};
  for (const { request, credentialID } of rows) {
    if (request.provider === MCP_PROVIDER) continue;
    const host = requestHost(request);
    selections[host ? `${request.provider}:${host}` : request.provider] = credentialID;
  }
  return selections;
}

/** A trigger card's reply names the accounts, so it waits for a tap, as on the web. */
export function needsManualContinue(setup: SetupRequest) {
  return setup.toolName === TRIGGER_TOOL;
}

const CREDENTIALS_DONE = "I've configured the required credentials.";

/** The same line the web sends for this tool once everything is connected. */
export function followUpMessage(setup: SetupRequest, rows: ConnectedRow[]) {
  if (isMCPSetup(setup)) {
    return "I've connected the integration. Please retry the same call.";
  }
  if (setup.toolName === TRIGGER_TOOL) {
    const picked = Object.fromEntries(rows.map((row) => [row.request.key, row.credentialID]));
    return (
      "I've selected the credentials to use. Call setup_agent_webhook_trigger " +
      `with credentials=${JSON.stringify(picked)} to finish setting up the trigger.`
    );
  }
  if (PREVIEW_TOOLS.includes(setup.toolName)) {
    return `${CREDENTIALS_DONE} Please check if everything is ready and proceed with running the workflow.`;
  }
  if (setup.inputValues) {
    return `${CREDENTIALS_DONE} Run with these inputs: ${JSON.stringify(setup.inputValues, null, 2)}`;
  }
  if (setup.toolName === CONNECT_TOOL) {
    return `${CREDENTIALS_DONE} I've connected my account. Please continue.`;
  }
  return `${CREDENTIALS_DONE} Please re-run this step now.`;
}
