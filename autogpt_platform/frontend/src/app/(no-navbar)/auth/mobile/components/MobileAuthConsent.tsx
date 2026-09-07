"use client";

import { AuthCard } from "@/components/auth/AuthCard";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useMobileAuthConsent } from "./useMobileAuthConsent";

interface Props {
  userID: string;
  email: string;
  codeChallenge: string;
  state: string;
}

export function MobileAuthConsent({
  userID,
  email,
  codeChallenge,
  state,
}: Props) {
  const { isConnecting, callbackURL, error, connect, returnToApp, cancel } =
    useMobileAuthConsent(codeChallenge, state, userID);

  return (
    <AuthCard
      title="Connect AutoGPT on your phone"
      className="gap-6 p-6 sm:p-8"
    >
      <div className="space-y-3 text-center">
        <Text variant="body">Sign in to the AutoGPT app as</Text>
        <Text variant="body" unmask={false} className="break-all font-semibold">
          {email}
        </Text>
        <Text variant="small" tone="secondary">
          Your chats, files, and connected tools will be available in the app.
          Continue only if you started this sign-in on your phone.
        </Text>
      </div>
      {error ? (
        <Text variant="small" tone="danger" role="alert">
          {error}
        </Text>
      ) : null}
      <div className="flex w-full flex-col items-center gap-4">
        {callbackURL ? (
          <Button className="w-full" onClick={returnToApp}>
            Return to AutoGPT
          </Button>
        ) : (
          <Button className="w-full" loading={isConnecting} onClick={connect}>
            Connect AutoGPT
          </Button>
        )}
        <Button variant="link" disabled={isConnecting} onClick={cancel}>
          Cancel
        </Button>
      </div>
    </AuthCard>
  );
}
