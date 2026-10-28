"use client";

import { Button } from "@/components/atoms/Button/Button";
import { credentialService } from "@/components/contextual/CredentialsInput/service-identity";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { ProviderAvatar } from "../DetailView/ProviderAvatar";
import type { ApiKeyConnectFormValues } from "../DetailView/schema";
import { AuthType, type ConnectableProvider } from "../../helpers";
import type { UseFormReturn } from "react-hook-form";
import { InlineApiKeyForm } from "./InlineApiKeyForm";
import { ServiceKeyForm } from "./ServiceKeyForm";

export interface WorkflowConnectOptions {
  onSignIn: () => void;
  isConnecting: boolean;
  service?: ReturnType<typeof credentialService>;
}

interface Props {
  provider: ConnectableProvider;
  options: WorkflowConnectOptions;
  form: UseFormReturn<ApiKeyConnectFormValues>;
  onApiKeySubmit: (values: ApiKeyConnectFormValues) => void;
  onConnected: (credential?: CredentialsMetaResponse) => void;
}

export function WorkflowConnectionView({
  provider,
  options,
  form,
  onApiKeySubmit,
  onConnected,
}: Props) {
  const hasSignIn = provider.supportedAuthTypes.includes(AuthType.oauth2);
  const hasKey = provider.supportedAuthTypes.includes(AuthType.api_key);
  return (
    <div className="flex flex-col gap-5">
      <div className="flex size-14 items-center justify-center rounded-2xl border border-separator p-3">
        <ProviderAvatar
          id={provider.iconId ?? provider.id}
          name={provider.name}
        />
      </div>
      <div>
        <h3 className="text-xl font-medium text-foreground-strong">
          Connect {provider.name}
        </h3>
        <p className="mt-2 text-sm text-foreground-tertiary">
          Choose the account Otto should use for this workflow.
        </p>
      </div>
      {hasSignIn ? (
        <>
          <div className="rounded-xl bg-surface-muted p-4 text-sm text-foreground-tertiary">
            Review the requested access on {provider.name}&apos;s sign-in page
            before connecting.
          </div>
          <Button
            variant="primary"
            size="small"
            loading={options.isConnecting}
            disabled={options.isConnecting}
            onClick={options.onSignIn}
          >
            {options.isConnecting
              ? "Connecting…"
              : `Continue with ${provider.name}`}
          </Button>
          <p className="text-center text-xs text-foreground-tertiary">
            Sign in on the app&apos;s own page, then return here.
          </p>
          {hasKey && (
            <details className="border-t border-separator pt-4">
              <summary className="cursor-pointer text-sm text-foreground-tertiary">
                Use an API key instead
              </summary>
              <div className="mt-4 space-y-3">
                <InlineApiKeyForm
                  form={form}
                  providerName={provider.name}
                  onSubmit={onApiKeySubmit}
                />
                <Button
                  variant="primary"
                  size="small"
                  loading={options.isConnecting}
                  disabled={!form.formState.isValid || options.isConnecting}
                  onClick={form.handleSubmit(onApiKeySubmit)}
                >
                  Save and connect
                </Button>
              </div>
            </details>
          )}
        </>
      ) : options.service ? (
        <ServiceKeyForm
          provider={provider.id}
          service={options.service}
          onConnected={onConnected}
        />
      ) : null}
    </div>
  );
}
