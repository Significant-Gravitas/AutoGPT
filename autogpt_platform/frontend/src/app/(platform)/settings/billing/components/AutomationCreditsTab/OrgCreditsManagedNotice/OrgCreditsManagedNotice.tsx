"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ORG_CREDITS_MANAGED_MESSAGE } from "@/services/org-team/org-wallet";

// What a team organization's member without billing rights sees where the
// balance and its actions would be.
export function OrgCreditsManagedNotice() {
  return (
    <section
      className="flex w-full flex-col gap-2"
      data-testid="org-credits-managed-notice"
    >
      <div className="px-4">
        <Text variant="body-medium" as="h2">
          Automation credits
        </Text>
      </div>
      <div className="rounded-large border border-border-base bg-surface p-5 shadow-message-table">
        <Text variant="body" tone="muted">
          {ORG_CREDITS_MANAGED_MESSAGE}
        </Text>
      </div>
    </section>
  );
}
