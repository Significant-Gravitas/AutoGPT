"use client";

import {
  AuthType,
  type AuthMethod,
  type ConnectableProvider,
} from "../../helpers";
import { ApiKeyConnectForm } from "./ApiKeyConnectForm";
import { DeviceAuthConnectButton } from "@/components/contextual/DeviceAuth/DeviceAuthConnectButton";
import { useProviderConnectCopy } from "./useProviderConnectCopy";
import { SubscriptionConnectExplainer } from "./SubscriptionConnectExplainer";
import { OAuthConnectButton } from "./OAuthConnectButton";
import { UnsupportedNotice } from "./UnsupportedNotice";

const TAB_LABEL: Record<AuthMethod, string> = {
  [AuthType.oauth2]: "OAuth",
  [AuthType.api_key]: "API key", // pragma: allowlist secret
  [AuthType.user_password]: "User / password", // pragma: allowlist secret
  [AuthType.host_scoped]: "Host",
  [AuthType.device_code]: "Device auth",
};

interface Props {
  method: AuthMethod;
  provider: ConnectableProvider;
  onSuccess: () => void;
}

export function MethodPanel({ method, provider, onSuccess }: Props) {
  const authProvider = getAuthProvider(provider, method);
  const connectCopy = useProviderConnectCopy(authProvider);
  if (method === AuthType.oauth2) {
    return (
      <div className="flex flex-col gap-4">
        {connectCopy.isSubscription && connectCopy.displayName && (
          <SubscriptionConnectExplainer
            providerName={connectCopy.displayName}
            modelsSentence={connectCopy.modelsSentence}
          />
        )}
        <OAuthConnectButton
          provider={authProvider}
          providerName={connectCopy.displayName ?? provider.name}
          buttonLabel={connectCopy.buttonLabel ?? undefined}
          termsNotice={connectCopy.termsCompany ?? undefined}
          onSuccess={onSuccess}
        />
      </div>
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
  if (method === AuthType.device_code) {
    return (
      <DeviceAuthConnectButton
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
  // The tab for a subscription sign-in is named after the account, not the
  // mechanism: "ChatGPT" tells someone what they are about to connect,
  // where "OAuth" tells them how. Names come from the server so a new
  // provider does not need a branch here; the mechanism label is the
  // fallback for anything it does not describe.
  const authProvider = getAuthProvider(provider, method);
  if (method === AuthType.oauth2 && authProvider) {
    return SUBSCRIPTION_TAB_LABELS[authProvider] ?? TAB_LABEL[method];
  }
  return TAB_LABEL[method];
}

// Sync fallback for the tab strip, which renders before the provider
// descriptions have loaded. Kept deliberately short: it is a label, and a
// provider missing from it degrades to "OAuth" rather than to something
// wrong.
const SUBSCRIPTION_TAB_LABELS: Record<string, string> = {
  codex: "ChatGPT",
  github_copilot: "GitHub Copilot",
  grok: "Grok",
};

export { TAB_LABEL };
