import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertLlmRouteUpdate } from "@/app/api/__generated__/models/expertLlmRouteUpdate";
import type { SelectOption } from "@/components/atoms/Select/Select";

export const ACCOUNT_DEFAULT_VALUE = "account-default";
const ACCOUNT_DEFAULT_LABEL = "Account default";

type RouteFields = Pick<Expert, "llm_auth_provider" | "llm_credential_id">;

export function routeValue(route: RouteFields): string {
  if (!route.llm_auth_provider) return ACCOUNT_DEFAULT_VALUE;
  return `${route.llm_auth_provider}:${route.llm_credential_id ?? ""}`;
}

export function parseRouteValue(value: string): ExpertLlmRouteUpdate {
  if (value === ACCOUNT_DEFAULT_VALUE) return { auth_provider: null };
  const separator = value.indexOf(":");
  const provider = value.slice(0, separator);
  const credentialId = value.slice(separator + 1);
  if (
    provider !== "platform" &&
    provider !== "codex" &&
    provider !== "microsoft_365_copilot"
  ) {
    return { auth_provider: null };
  }
  return {
    auth_provider: provider,
    credential_id: credentialId || null,
  };
}

/**
 * One row per connection the owner can chat over, after "Account default".
 * Two linked accounts of one provider share a label, so the account each runs
 * as is appended when the owner's credentials know it.
 */
export function routeOptions(
  transports: ChatTransportResponse[],
  credentials: CredentialsMetaResponse[],
  expert: Expert,
): SelectOption[] {
  const usernameById = new Map(
    credentials.map((credential) => [credential.id, credential.username]),
  );
  const options: SelectOption[] = [
    { value: ACCOUNT_DEFAULT_VALUE, label: ACCOUNT_DEFAULT_LABEL },
    ...transports
      .filter((transport) => transport.available)
      .map((transport) => {
        const username = transport.credential_id
          ? usernameById.get(transport.credential_id)
          : null;
        return {
          value: routeValue({
            llm_auth_provider: transport.auth_provider,
            llm_credential_id: transport.credential_id,
          }),
          label: username
            ? `${transport.label} · ${username}`
            : transport.label,
        };
      }),
  ];
  const currentValue = routeValue(expert);
  if (options.some((option) => option.value === currentValue)) return options;
  // A pin to a connection that is gone still has to show as the current
  // choice, or the control would read as blank while the server still holds
  // the pin. It can be re-chosen; the note beside it says why not to.
  return [
    ...options,
    {
      value: currentValue,
      label: `${expert.llm_route_label ?? expert.llm_auth_provider} · not connected`,
    },
  ];
}

export function isOnSubscription(expert: RouteFields): boolean {
  return (
    expert.llm_auth_provider !== null &&
    expert.llm_auth_provider !== undefined &&
    expert.llm_auth_provider !== "platform"
  );
}

export function getRouteNote(expert: Expert): {
  text: string;
  warning: boolean;
} {
  const label = expert.llm_route_label ?? "that connection";
  if (!expert.llm_auth_provider) {
    return {
      text: "Follows the default AI connection chosen in Settings.",
      warning: false,
    };
  }
  if (expert.llm_route_available === false) {
    return {
      text: `${label} connection missing. ${expert.name} runs on your account default until you reconnect it or pick another.`,
      warning: true,
    };
  }
  if (expert.llm_auth_provider === "microsoft_365_copilot") {
    return {
      text: `Chats run on ${label}, which cannot run tools. Routines, follow-ups and delegations fall back to the account default.`,
      warning: true,
    };
  }
  return {
    text: `New threads, routines and follow-ups run on ${label}.`,
    warning: false,
  };
}
