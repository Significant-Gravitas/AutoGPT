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
  if (method === AuthType.oauth2) {
    return (
      <OAuthConnectButton
        provider={provider.id}
        providerName={provider.name}
        onSuccess={onSuccess}
      />
    );
  }
  if (method === AuthType.api_key) {
    return (
      <ApiKeyConnectForm
        provider={provider.id}
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

export { TAB_LABEL };
