"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { getWorkIQAdminConsentURL } from "@/lib/microsoft-setup";
import { WorkIQInstructions } from "./WorkIQInstructions";

interface Props {
  tenantID?: string;
  onRetry: () => void;
}

export function WorkIQAdminConsent({ tenantID, onRetry }: Props) {
  const approvalURL = tenantID ? getWorkIQAdminConsentURL(tenantID) : null;

  return (
    <section className="flex flex-col gap-3 rounded-lg border border-border-base p-4">
      <Text as="h3" variant="body-medium">
        Set up Work IQ for your organization
      </Text>
      <Text variant="small" tone="muted">
        Microsoft turns Work IQ on per organization, and it bills by usage. Your
        other Microsoft services keep working while this is pending.
      </Text>
      {approvalURL && (
        <Button
          as="NextLink"
          variant="secondary"
          size="small"
          href={approvalURL}
          target="_blank"
          rel="noopener noreferrer"
        >
          Turn on Work IQ (admins)
        </Button>
      )}
      <WorkIQInstructions />
      <Button variant="ghost" size="small" onClick={onRetry}>
        Check again
      </Button>
    </section>
  );
}
