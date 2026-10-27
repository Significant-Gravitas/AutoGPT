"use client";

import { Text } from "@/components/atoms/Text/Text";
import { MicrosoftConnectionPanel } from "@/components/contextual/MicrosoftConnection/MicrosoftConnectionPanel";
import { useMicrosoftBrowserSignIn } from "@/components/contextual/MicrosoftConnection/useMicrosoftBrowserSignIn";
import type { AccessConnectProps } from "../accessProviders";

export function MicrosoftAccess({ onDone }: AccessConnectProps) {
  const { browserAvailable, isLoading } = useMicrosoftBrowserSignIn();

  // Until the server says which sign-in it supports, either button could be wrong.
  if (isLoading)
    return (
      <Text variant="body" role="status" tone="muted">
        Loading…
      </Text>
    );

  return (
    <MicrosoftConnectionPanel
      browserAvailable={browserAvailable}
      doneLabel="Continue"
      onDone={() => onDone()}
    />
  );
}
