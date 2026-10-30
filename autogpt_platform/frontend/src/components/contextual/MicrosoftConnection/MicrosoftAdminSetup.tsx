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
  retryLabel?: string;
}

// The person's own to fix: they declined, or Microsoft wants a new sign-in.
const OWN_REASONS: ReadonlySet<MicrosoftSetupReason> = new Set([
  "access_denied",
  "reconnect_required",
]);

export function MicrosoftAdminSetup({
  reason,
  onRetry,
  retryLabel = "Try again",
}: Props) {
  const forAdmin = reason !== "app_setup_required" && !OWN_REASONS.has(reason);
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
    <section className="flex flex-col gap-3 rounded-lg border border-border-base p-4">
      <Text variant="body" role="status">
        {MICROSOFT_SETUP_MESSAGES[reason]}
      </Text>
      {reason === "app_setup_required" && (
        <Text variant="small" tone="muted">
          Browser sign-in isn&apos;t set up on this AutoGPT server yet. Whoever
          runs it needs to add the Microsoft app&apos;s client ID, secret and
          callback address.
        </Text>
      )}
      {forAdmin && (
        <Text variant="small" tone="muted">
          Anything already connected keeps working. Send the setup link to your
          Microsoft admin, or approve it yourself if you&apos;re the admin.
        </Text>
      )}
      <div className="flex flex-wrap gap-2">
        {forAdmin && (
          <Button variant="secondary" size="small" onClick={copySetupLink}>
            Copy setup link for my admin
          </Button>
        )}
        {onRetry && (
          <Button variant="ghost" size="small" onClick={onRetry}>
            {retryLabel}
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
