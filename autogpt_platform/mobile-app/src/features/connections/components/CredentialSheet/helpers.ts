import { ApiError } from "@/lib/api/client";
import type { NewCredential } from "@/lib/api/integrations";

export type SheetKind = "api_key" | "user_password" | "host_scoped" | "mcp_token";

/** Which service the sheet saves a key, login, header or token for. */
export interface SheetTarget {
  /** Handed back with the saved account, e.g. the card row it answers. */
  rowKey: string;
  kind: SheetKind;
  provider: string;
  providerName: string;
  /** The host a host-scoped header is for, when the tool named one. */
  host: string | null;
  /** The MCP server a token is for. */
  serverURL: string | null;
  /** Said instead of the usual intro, e.g. why a sign-in became a token. */
  notice?: string;
}

/** Exactly what was typed in each field. */
export interface SheetValues {
  title: string;
  secret: string;
  username: string;
  host: string;
  headerName: string;
  server: string;
  scheme: string;
}

export interface Scheme {
  value: string;
  label: string;
}

// The web's SERVER_SCOPE_SCHEMES: logins for these say which server they are for.
const SERVER_SCHEMES: Record<string, Scheme[]> = {
  smtp: [{ value: "smtp", label: "SMTP" }],
  database: [
    { value: "postgresql", label: "PostgreSQL" },
    { value: "mysql", label: "MySQL" },
    { value: "mssql", label: "SQL Server" },
  ],
};

const SERVER_PATTERN = /^(\[[0-9a-fA-F:.]+\]|[A-Za-z0-9._-]+)(?::([0-9]{1,5}))?$/;

export function serverSchemes(provider: string) {
  return SERVER_SCHEMES[provider] ?? null;
}

export function initialValues(target: SheetTarget): SheetValues {
  return {
    title: "",
    secret: "",
    username: "",
    host: target.host ?? "",
    headerName: "Authorization",
    server: "",
    scheme: serverSchemes(target.provider)?.[0].value ?? "",
  };
}

function isValidServer(server: string) {
  const match = SERVER_PATTERN.exec(server.trim());
  if (!match) return false;
  return match[2] === undefined || (Number(match[2]) >= 1 && Number(match[2]) <= 65535);
}

/** A bare host, maybe with a port: no scheme and no path, as the web asks. */
export function isValidHost(host: string) {
  const value = host.trim();
  return !!value && !/^[a-z][a-z\d+.-]*:\/\//i.test(value) && !value.includes("/");
}

/** The body to save, or null while something required is missing or invalid. */
export function buildCredentialBody(
  target: SheetTarget,
  values: SheetValues,
): NewCredential | null {
  const secret = values.secret.trim();
  if (!secret) return null;
  const title = values.title.trim();
  if (target.kind === "api_key") {
    return { type: "api_key", title: title || `${target.providerName} key`, api_key: secret };
  }
  if (target.kind === "user_password") {
    const username = values.username.trim();
    if (!username) return null;
    const schemes = serverSchemes(target.provider);
    if (schemes && !isValidServer(values.server)) return null;
    return {
      type: "user_password",
      title: title || `${target.providerName} login`,
      username,
      password: values.secret,
      ...(schemes
        ? { scope: `${values.scheme}://${values.server.trim().toLowerCase()}` }
        : {}),
    };
  }
  if (target.kind === "host_scoped") {
    const host = values.host.trim();
    const headerName = values.headerName.trim();
    if (!isValidHost(host) || !headerName) return null;
    return { type: "host_scoped", title: host, host, headers: { [headerName]: secret } };
  }
  return null;
}

export function canSave(target: SheetTarget, values: SheetValues) {
  if (target.kind === "mcp_token") {
    return !!values.secret.trim() && !!target.serverURL;
  }
  return buildCredentialBody(target, values) !== null;
}

export function sheetIntro(kind: SheetKind) {
  switch (kind) {
    case "api_key":
      return "Paste your API key. It's stored encrypted and never shown in the chat.";
    case "user_password":
      return "Your login is stored encrypted and never shown in the chat.";
    case "host_scoped":
      return "Requests to this host get this header. It's stored encrypted and never shown in the chat.";
    case "mcp_token":
      return "This server takes a token instead of a sign-in. Paste a token or a full Authorization header.";
  }
}

export function saveErrorText(error: unknown) {
  if (error instanceof ApiError && error.status === 402) {
    return "Adding a service needs an active plan. Finish this on the web.";
  }
  if (error instanceof ApiError && [400, 422].includes(error.status)) {
    return error.message;
  }
  return "Couldn't save it. Check your connection and try again.";
}
