"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { DeviceAuthConnectButton } from "@/components/contextual/DeviceAuth/DeviceAuthConnectButton";
import { MICROSOFT_PROVIDER } from "@/lib/microsoft-setup";
import { MicrosoftServiceList } from "./components/MicrosoftServiceList";
import { MicrosoftSignIn } from "./components/MicrosoftSignIn";
import { needsWorkIQSetup, workIQToAllow } from "./helpers";
import { MicrosoftAdminSetup } from "./MicrosoftAdminSetup";
import { useMicrosoftConnection } from "./useMicrosoftConnection";
import { WorkIQAdminConsent } from "./WorkIQAdminConsent";

interface Props {
  browserAvailable: boolean;
  /** What a first sign-in asks for. Omit for the basics. */
  scopes?: string[];
  doneLabel?: string;
  onDone: (credential?: CredentialsMetaResponse) => void;
  /** Offers "Not now" until an account is connected. */
  onSkip?: () => void;
}

export function MicrosoftConnectionPanel({
  browserAvailable,
  scopes,
  doneLabel = "Done",
  onDone,
  onSkip,
}: Props) {
  const connection = useMicrosoftConnection({ scopes });
  const { credential, check } = connection;
  const workIQ = workIQToAllow(check);

  if (!credential || connection.adding) {
    return (
      <MicrosoftSignIn
        browserAvailable={browserAvailable}
        scopes={scopes}
        loading={connection.isLoading}
        pending={connection.oauth.isPending}
        recovery={connection.recovery}
        onSignIn={connection.signIn}
        onDeviceConnected={connection.handleConnected}
        onDeviceError={connection.handleError}
        onSkip={credential ? connection.cancelAddAccount : onSkip}
      />
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {connection.credentials.length > 1 ? (
        <Select
          id="microsoft-account"
          label="Microsoft account"
          hideLabel
          size="small"
          wrapperClassName="mb-0"
          value={credential.id}
          onValueChange={connection.select}
          options={connection.credentials.map((item) => ({
            value: item.id,
            label: item.title || item.username || "Microsoft account",
          }))}
        />
      ) : (
        <Text variant="body-medium">
          {credential.title || credential.username}
        </Text>
      )}
      <MicrosoftServiceList
        check={check}
        checking={connection.checking}
        browserAvailable={browserAvailable}
        onAllow={connection.allow}
      />
      {connection.pending.length > 0 &&
        (browserAvailable ? (
          <Button
            variant="primary"
            loading={connection.oauth.isPending}
            onClick={() => connection.allow(connection.pending)}
          >
            Allow everything listed
          </Button>
        ) : (
          <DeviceAuthConnectButton
            provider={MICROSOFT_PROVIDER}
            providerName="Microsoft"
            scopes={connection.pending}
            credentialID={credential.id}
            onSuccess={connection.handleConnected}
            onError={connection.handleError}
          />
        ))}
      {!browserAvailable && workIQ && (
        <DeviceAuthConnectButton
          provider={MICROSOFT_PROVIDER}
          providerName="Work IQ"
          scopes={workIQ.scopes}
          credentialID={credential.id}
          onSuccess={connection.handleConnected}
          onError={connection.handleError}
        />
      )}
      {check?.admin_consent_url && (
        <div className="flex flex-col gap-2">
          <Button
            as="NextLink"
            variant="secondary"
            href={check.admin_consent_url}
            target="_blank"
            rel="noopener noreferrer"
          >
            Approve for my whole organization
          </Button>
          <Text variant="small" tone="muted">
            Only a Microsoft admin can do this. Afterwards everyone in your
            organization gets these services in one click.
          </Text>
        </div>
      )}
      {needsWorkIQSetup(check) && (
        <WorkIQAdminConsent
          tenantID={credential.microsoft_identity?.tenant_id}
          onRetry={connection.recheck}
        />
      )}
      {connection.recovery && (
        <MicrosoftAdminSetup
          reason={connection.recovery}
          onRetry={connection.recheck}
        />
      )}
      <div className="flex flex-wrap gap-2">
        <Button
          variant="ghost"
          size="small"
          loading={connection.checking}
          onClick={connection.recheck}
        >
          Check again
        </Button>
        <Button
          variant="ghost"
          size="small"
          onClick={browserAvailable ? connection.signIn : connection.addAccount}
        >
          Connect another account
        </Button>
        <Button
          variant="primary"
          size="small"
          onClick={() => onDone(credential)}
        >
          {doneLabel}
        </Button>
      </div>
    </div>
  );
}
