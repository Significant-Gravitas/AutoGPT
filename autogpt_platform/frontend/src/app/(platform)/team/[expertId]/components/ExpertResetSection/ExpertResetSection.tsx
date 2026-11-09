"use client";

import { ArrowReloadHorizontalIcon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ResetExpertDialog } from "./ResetExpertDialog";

interface Props {
  expert: Expert;
}

/** The factory reset: unlike firing, it erases what the expert learned. */
export function ExpertResetSection({ expert }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  // A created expert has no template to go back to, so its soul and routines
  // stay as they are.
  const hasTemplate = Boolean(expert.source_template_id);

  return (
    <section
      aria-label={`Reset ${expert.name}`}
      className="overflow-hidden rounded-2xl bg-surface smooth-shadow-ring-sm"
    >
      <div className="flex items-start gap-3 px-5 py-4">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-surface-muted text-foreground-tertiary">
          <Icon icon={ArrowReloadHorizontalIcon} size={20} aria-hidden />
        </span>
        <div className="min-w-0">
          <Text variant="large-medium" tone="primary">
            Reset {expert.name}
          </Text>
          <Text variant="body-md" tone="secondary" className="mt-0.5">
            {hasTemplate
              ? `Start over with ${expert.name} as they were the day you hired them: their memory, computer and your edits are erased, and your chats with them are archived.`
              : `Start over with ${expert.name}: their memory and computer are erased, and your chats with them are archived.`}{" "}
            Unlike firing, this can&apos;t be undone.
          </Text>
        </div>
      </div>
      <div className="flex justify-end border-t border-separator bg-surface-muted/60 px-5 py-3">
        <Button
          variant="secondary"
          size="small"
          textSize="body-md"
          onClick={() => setIsOpen(true)}
          data-testid="expert-reset-button"
        >
          Reset {expert.name}
        </Button>
      </div>
      <ResetExpertDialog
        expertId={expert.id}
        expertName={expert.name}
        hasTemplate={hasTemplate}
        open={isOpen}
        onClose={() => setIsOpen(false)}
      />
    </section>
  );
}
