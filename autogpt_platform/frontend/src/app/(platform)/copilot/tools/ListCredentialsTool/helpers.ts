import type { ToolUIPart } from "ai";

interface CredentialMeta {
  id: string;
  provider: string;
  type: string;
  title?: string | null;
  scopes?: string[] | null;
  username?: string | null;
  host?: string | null;
  is_managed?: boolean;
}

interface CredentialListOutput {
  type: "credential_list";
  message: string;
  credentials: CredentialMeta[];
  providers: string[];
  count: number;
}

interface ErrorOutput {
  type: "error" | "need_login";
  message: string;
  error?: string;
}

export type ListCredentialsOutput = CredentialListOutput | ErrorOutput;

export type { CredentialMeta };

function parseOutput(output: unknown): ListCredentialsOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return parseOutput(JSON.parse(trimmed) as unknown);
    } catch {
      return null;
    }
  }
  if (typeof output === "object") {
    const obj = output as Record<string, unknown>;
    if (typeof obj.type === "string") {
      return output as ListCredentialsOutput;
    }
  }
  return null;
}

export function getListCredentialsOutput(part: {
  output?: unknown;
}): ListCredentialsOutput | null {
  return parseOutput(part.output);
}

export function isCredentialList(
  o: ListCredentialsOutput,
): o is CredentialListOutput {
  return o.type === "credential_list";
}

export function isErrorOutput(o: ListCredentialsOutput): o is ErrorOutput {
  return o.type === "error" || o.type === "need_login";
}

export function formatProviderName(provider: string): string {
  return provider
    .split(/[_-]/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function getCredentialTypeLabel(type: string): string {
  switch (type) {
    case "oauth2":
      return "OAuth";
    case "api_key":
      return "API key";
    case "user_password":
      return "Username / password";
    case "host_scoped":
      return "Host-scoped";
    default:
      return type;
  }
}

export function getAnimationText(part: {
  state: ToolUIPart["state"];
  output?: unknown;
}): string {
  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return "Checking connected integrations…";
    case "output-available": {
      const output = getListCredentialsOutput(part);
      if (!output) return "Done";
      if (isErrorOutput(output))
        return "Could not check connected integrations";
      return output.message;
    }
    case "output-error":
      return "Could not check connected integrations";
    default:
      return "Checking connected integrations…";
  }
}
