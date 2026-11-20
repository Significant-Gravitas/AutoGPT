"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { AccessConnectProps } from "../../accessProviders";
import { GoogleLookupFailed } from "./components/GoogleLookupFailed";
import { GoogleServiceList } from "./components/GoogleServiceList";
import { useGoogleAccess } from "./useGoogleAccess";

export function GoogleAccess({ onDone, onSkip }: AccessConnectProps) {
  const access = useGoogleAccess();
  const { credential, missing } = access;

  if (access.isLoading)
    return (
      <Text variant="body" role="status" tone="muted">
        Loading…
      </Text>
    );

  if (access.lookupFailed && !credential)
    return (
      <GoogleLookupFailed
        error={access.lookupError}
        onRetry={access.retry}
        onSkip={onSkip}
      />
    );

  return (
    <div className="flex flex-col gap-4">
      {credential && (
        <Text variant="body-medium">
          {credential.username || credential.title || "Google account"}
        </Text>
      )}
      <GoogleServiceList
        credential={credential}
        pending={access.isPending}
        onAllow={(service) => access.allow([service])}
      />
      {!credential && (
        <Button
          variant="primary"
          loading={access.isPending}
          disabled={access.isPending}
          onClick={access.connect}
        >
          Connect Google
        </Button>
      )}
      {missing.length > 1 && (
        <Button
          variant="primary"
          loading={access.isPending}
          disabled={access.isPending}
          onClick={() => access.allow(missing)}
        >
          Allow everything listed
        </Button>
      )}
      {credential ? (
        <Button
          variant={missing.length > 1 ? "secondary" : "primary"}
          onClick={onDone}
        >
          Continue
        </Button>
      ) : (
        <Button variant="ghost" onClick={onSkip}>
          Not now
        </Button>
      )}
    </div>
  );
}
