"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { AccessConnectProps } from "../../accessProviders";
import { GoogleServiceList } from "./components/GoogleServiceList";
import { useGoogleAccess } from "./useGoogleAccess";

export function GoogleAccess({ onDone, onSkip }: AccessConnectProps) {
  const access = useGoogleAccess();
  const { credential } = access;

  if (access.isLoading)
    return (
      <Text variant="body" role="status" tone="muted">
        Loading…
      </Text>
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
        pending={access.pending}
        onAllow={access.allow}
      />
      {credential ? (
        <Button variant="primary" onClick={onDone}>
          Continue
        </Button>
      ) : (
        <>
          <Button
            variant="primary"
            loading={access.pending}
            disabled={access.pending}
            onClick={access.connect}
          >
            Connect Google
          </Button>
          <Button variant="ghost" onClick={onSkip}>
            Not now
          </Button>
        </>
      )}
    </div>
  );
}
