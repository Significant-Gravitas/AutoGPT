"use client";

import {
  AuthType,
  type AuthMethod,
  type ConnectableProvider,
} from "../../helpers";
import { ApiKeyConnectForm } from "./ApiKeyConnectForm";
import { OAuthConnectButton } from "./OAuthConnectButton";
import { UnsupportedNotice } from "./UnsupportedNotice";

const TAB_LABEL: Record<AuthMethod, string> = {
  [AuthType.oauth2]: "OAuth",
  [AuthType.api_key]: "API key",
  [AuthType.user_password]: "User / password",
  [AuthType.host_scoped]: "Host",
};

interface Props {
  method: AuthMethod;
  provider: ConnectableProvider;
  onSuccess: () => void;
}

export function MethodPanel({ method, provider, onSuccess }: Props) {
  const authProvider = getAuthProvider(provider, method);
  if (method === AuthType.oauth2) {
    const isChatGPT = authProvider === "codex";
    return (
      <OAuthConnectButton
        provider={authProvider}
        providerName={isChatGPT ? "ChatGPT" : provider.name}
        buttonLabel={isChatGPT ? "Sign in with ChatGPT" : undefined}
        onSuccess={onSuccess}
      />
    );
  }
  if (method === AuthType.api_key) {
    return (
      <ApiKeyConnectForm
        provider={authProvider}
        providerName={provider.name}
        onSuccess={onSuccess}
      />
    );
  }
  return (
    <UnsupportedNotice
      providerName={provider.name}
      detail={`${TAB_LABEL[method]} sign-in for ${provider.name} is not yet wired up in this dialog.`}
    />
  );
}

export function getAuthProvider(
  provider: ConnectableProvider,
  method: AuthMethod,
): string {
  return provider.authProviderByType?.[method] ?? provider.id;
}

export function getAuthMethodLabel(
  provider: ConnectableProvider,
  method: AuthMethod,
): string {
  if (
    method === AuthType.oauth2 &&
    getAuthProvider(provider, method) === "codex"
  ) {
    return "ChatGPT";
  }
  return TAB_LABEL[method];
}

export { TAB_LABEL };
