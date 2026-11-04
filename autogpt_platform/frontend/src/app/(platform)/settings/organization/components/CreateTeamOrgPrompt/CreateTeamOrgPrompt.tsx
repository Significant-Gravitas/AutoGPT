"use client";

import { useState } from "react";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CreateOrgDialog } from "@/components/contextual/CreateOrgDialog/CreateOrgDialog";

export function CreateTeamOrgPrompt() {
  const [isCreateOpen, setIsCreateOpen] = useState(false);

  return (
    <section
      className="flex flex-wrap items-center justify-between gap-3 rounded-large border border-border-base p-4"
      data-testid="create-team-org-prompt"
    >
      <div className="flex min-w-0 flex-col gap-1">
        <Text variant="body-medium" as="h2">
          Want to work with others?
        </Text>
        <Text variant="small" tone="muted">
          A team organization has its own members, roles and credits.
        </Text>
      </div>
      <Button
        variant="secondary"
        size="small"
        onClick={() => setIsCreateOpen(true)}
      >
        Create a team organization
      </Button>
      <CreateOrgDialog open={isCreateOpen} onOpenChange={setIsCreateOpen} />
    </section>
  );
}
