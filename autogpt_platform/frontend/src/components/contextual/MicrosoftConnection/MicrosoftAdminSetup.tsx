"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  MICROSOFT_SETUP_MESSAGES,
  type MicrosoftSetupReason,
} from "@/lib/microsoft-setup";

interface Props {
  reason: MicrosoftSetupReason;
  onRetry?: () => void;
}

export function MicrosoftAdminSetup({ reason, onRetry }: Props) {
  const [copyMessage, setCopyMessage] = useState("");

  async function copySetupLink() {
    const link = `${window.location.origin}/microsoft-setup`;
    try {
      await navigator.clipboard.writeText(link);
      setCopyMessage("Setup link copied.");
    } catch {
      setCopyMessage(`Share this link with your Microsoft admin: ${link}`);
    }
  }

  return (
    <section className="flex flex-col gap-3 rounded-lg border border-zinc-200 p-4">
      <Text variant="body" role="status">
        {MICROSOFT_SETUP_MESSAGES[reason]}
      </Text>
      <Text variant="small" tone="muted">
        {reason === "app_setup_required"
          ? "Browser sign-in isn't set up on this AutoGPT server yet. Whoever runs it needs to add the Microsoft app's client ID, secret and callback address."
          : "Anything already connected keeps working. Send the setup link to your Microsoft admin, or approve it yourself if you're the admin."}
      </Text>
      <div className="flex flex-wrap gap-2">
        {reason !== "app_setup_required" && (
          <Button variant="secondary" size="small" onClick={copySetupLink}>
            Copy setup link for my admin
          </Button>
        )}
        {onRetry && (
          <Button variant="ghost" size="small" onClick={onRetry}>
            Try again
          </Button>
        )}
      </div>
      {copyMessage && (
        <Text variant="small" role="status">
          {copyMessage}
        </Text>
      )}
    </section>
  );
}
